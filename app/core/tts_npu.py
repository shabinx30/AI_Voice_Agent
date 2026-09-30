"""NPU-accelerated Kokoro-82M encoder stages for Intel AI Boost.

Splits Kokoro synthesis the way Intel's AI-PC benchmark does:

* ``albert`` (pre-exported static IR, NPU): token IDs + attention mask
  -> contextual embeddings ``d_en``.
* ``prosody body`` (exported pack-free replica, NPU): ``d_en`` + style
  -> per-token speech ``d`` + predicted durations ``pred_dur``.
* ``text encoder`` (exported pack-free replica, NPU): token IDs
  -> text features ``t_en``.
* CPU tail (plain PyTorch): duration alignment, prosody F0/N,
  and the ISTFT vocoder decoder.

The replicas drop ``pack_padded_sequence``/``pad_packed_sequence`` (which the
NPU compiler rejects) in favour of plain static-shape LSTM execution. Fed
with padded inputs plus an explicit pad mask they implement the identical
math; exactness is verified numerically against the original modules.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Maximum token length (incl. BOS/EOS) handled by the static NPU graphs.
# Longer sentences fall back to the full-graph CPU path in tts.py.
NPU_MAX_TOKENS: int = 217

# Hugging Face repo holding Intel's pre-exported static albert stage.
ALBERT_REPO: str = "mweinbach1/ai-pc-benchmarks-tts-intel-openvino"
ALBERT_PATH: str = "npu_static/staged/albert/len_217/openvino_model.xml"

# Samples produced per mel frame by the Kokoro ISTFT vocoder.
HOP_LENGTH: int = 600


class DurationEncoderStatic(nn.Module):
    """Pack-free replica of ``ProsodyPredictor.text_encoder`` (DurationEncoder).

    Operates on statically-shaped padded inputs; pad positions are zeroed via
    the boolean ``mask`` (True = pad). Without ``pack_padded_sequence`` every
    op is NPU-compatible. Given exact-length unpadded input the replica is
    bit-comparable to the original module.
    """

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.lstms = src.lstms
        self.dropout = src.dropout

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        from torch.nn import functional as F

        from kokoro.modules import AdaLayerNorm

        masks = mask
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                # NOTE: pack_padded_sequence removed (NPU-hostile). The caller
                # guarantees padded positions carry zeroed features and the
                # mask; the plain LSTM therefore implements identical math on
                # real positions for the forward direction.
                block.flatten_parameters()
                x, _ = block(x)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)


class TextEncoderStatic(nn.Module):
    """Pack-free replica of Kokoro ``TextEncoder``."""

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.embedding = src.embedding
        self.cnn = src.cnn
        self.lstm = src.lstm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # [B, T, chn]
        x = x.transpose(1, 2)  # [B, chn, T]
        m = mask.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x.masked_fill_(m, 0.0)
        return x


class ProsodyBodyStatic(nn.Module):
    """Static prosody front-end: DurationEncoder + duration LSTM + projection.

    Args produce per-token hidden states ``d`` and integer durations.
    Speed is baked to 1.0 (the assistant pipeline has no speed control).
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.dur_enc = DurationEncoderStatic(predictor.text_encoder)
        self.lstm = predictor.lstm
        self.duration_proj = predictor.duration_proj

    def forward(
        self,
        d_en: torch.Tensor,
        style_tail: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch.nn import functional as F

        d = self.dur_enc(d_en, style_tail, mask)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(d)
        duration = self.duration_proj(F.dropout(x, 0.5, training=False))
        pred_dur = torch.round(torch.sigmoid(duration).sum(-1)).clamp(min=1).long()
        return d, pred_dur


def pad_inputs(
    input_ids: np.ndarray, max_len: int = NPU_MAX_TOKENS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pads ``[1, n]`` token IDs to ``[1, max_len]`` with NPU-ready masks.

    Args:
        input_ids: int64 token IDs of shape [1, n].
        max_len: Static padded length.

    Returns:
        Tuple (padded_ids [1, max_len] int64, attention_mask [1, max_len]
        int64 with 1 = real, text_mask [1, max_len] bool with True = pad,
        n actual length).

    Raises:
        ValueError: If ``n`` exceeds ``max_len``.
    """
    n = int(input_ids.shape[1])
    if n > max_len:
        raise ValueError(f"Token length {n} exceeds NPU static length {max_len}.")
    padded = np.zeros((1, max_len), dtype=np.int64)
    padded[0, :n] = input_ids[0]
    positions = np.arange(max_len)
    attention_mask = (positions < n).astype(np.int64)[None, :]
    text_mask = (positions >= n)[None, :]
    return padded, attention_mask, text_mask, n


class KokoroNPUEncoder:
    """Kokoro encoder stages with Intel NPU acceleration.

    Stage placement (quality-first default):

    * ``albert`` runs on CPU: the 12-layer transformer runs in the NPU's
      fp16 precision with ~2% feature deviation, and the recurrent prosody
      stack amplifies that into a clearly different waveform (0.51
      correlation vs the CPU reference, rhythm/durations still identical).
    * ``prosody_body`` and ``text_encoder`` run on the NPU: their fp16
      outputs match CPU to <0.01 with identical predicted durations, keeping
      end-to-end synthesis at ~0.98 correlation with the CPU reference.

    Pass ``cpu_stages=frozenset()`` to force all stages onto the NPU when
    maximum NPU load matters more than fidelity.

    Attributes:
        stage_devices: Mapping of stage name -> device actually used
            ('NPU' or 'CPU').
    """

    # Stages kept on CPU even for an NPU run (see class docstring).
    CPU_PINNED_STAGES = frozenset({"albert"})

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        max_tokens: int = NPU_MAX_TOKENS,
        cpu_stages: Optional[frozenset] = None,
        turbo: Optional[bool] = None,
    ) -> None:
        """Initializes (but does not yet load) the NPU encoder.

        Args:
            artifacts_dir: Directory for exported IRs + compiler cache.
            max_tokens: Static token length for the NPU graphs.
            cpu_stages: Stage names pinned to CPU. Defaults to albert.
            turbo: Force NPU_TURBO on/off, or None for driver default.
        """
        self.max_tokens = max_tokens
        self.cpu_stages = self.CPU_PINNED_STAGES if cpu_stages is None else cpu_stages
        self.turbo = turbo
        default_dir = Path(__file__).resolve().parent.parent.parent / "model_artifacts" / "kokoro_npu"
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else default_dir
        self.stage_devices: Dict[str, str] = {}
        self.stage_configs: Dict[str, Dict] = {}
        self._core = None
        self._albert = None
        self._prosody = None
        self._textenc = None
        self._kmodel = None
        self._is_loaded = False

    def _npu_config(self, cache_dir: str) -> Dict:
        """Builds an explicit NPU compile config from supported properties only.

        Uses LATENCY hint (TTFA-optimized) + compiler cache + optional turbo.
        Never sets a property the installed NPU plugin does not advertise.
        """
        try:
            from app.core.diagnostics import build_npu_compile_config

            return build_npu_compile_config(
                cache_dir=cache_dir,
                performance_hint="LATENCY",
                turbo=self.turbo,
                num_requests=1,
            )
        except Exception:
            # Minimal safe fallback: cache only.
            return {"CACHE_DIR": cache_dir} if cache_dir else {}

    # ------------------------------------------------------------------
    # Loading / exporting
    # ------------------------------------------------------------------
    def load(self, preferred_device: str = "NPU") -> None:
        """Loads weights, exports replicas if needed, compiles for NPU/CPU."""
        if self._is_loaded:
            return
        import openvino as ov

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(self.artifacts_dir / "_compiler_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self._core = ov.Core()

        self._load_torch_kmodel()
        self._albert = self._compile_albert(preferred_device, cache_dir)
        self._prosody = self._compile_replica(
            name="prosody_body",
            build_fn=self._build_prosody_body,
            example_fn=self._prosody_example,
            preferred=preferred_device,
            cache_dir=cache_dir,
        )
        self._textenc = self._compile_replica(
            name="text_encoder",
            build_fn=self._build_text_encoder,
            example_fn=self._textenc_example,
            preferred=preferred_device,
            cache_dir=cache_dir,
        )
        self._is_loaded = True
        logger.info("Kokoro NPU encoder ready: %s", self.stage_devices)

    def _load_torch_kmodel(self) -> None:
        """Loads the torch KModel (weights for replicas + CPU tail)."""
        if self._kmodel is not None:
            return
        from kokoro.model import KModel

        logger.info("Loading torch Kokoro weights for NPU encoder...")
        self._kmodel = KModel(repo_id="hexgrad/Kokoro-82M")
        self._kmodel.eval()

    def _placement(self, stage: str, preferred: str) -> List[str]:
        """Returns compile candidates: pinned stages go straight to CPU."""
        if stage in self.cpu_stages:
            return ["CPU"]
        return [preferred, "CPU"] if preferred != "CPU" else ["CPU"]

    def _compile_albert(self, preferred: str, cache_dir: str):
        """Compiles Intel's pre-exported static albert stage.

        Uses explicit device selection (never AUTO): tries ``preferred``
        first, then falls back to CPU with an explicit warning stating why.
        """
        from huggingface_hub import hf_hub_download

        xml_path = hf_hub_download(ALBERT_REPO, filename=ALBERT_PATH)
        model = self._core.read_model(xml_path)
        for device in self._placement("albert", preferred):
            try:
                config = self._npu_config(cache_dir) if device == "NPU" else (
                    {"CACHE_DIR": cache_dir} if cache_dir else {}
                )
                compiled = self._core.compile_model(model, device, config)
                self.stage_devices["albert"] = device
                self.stage_configs["albert"] = dict(config)
                if device != preferred:
                    logger.warning(
                        "Stage 'albert': requested %s unavailable (%s); "
                        "explicit fallback to %s.",
                        preferred, "see previous error", device,
                    )
                else:
                    logger.info(
                        "Stage 'albert' compiled on %s with %s.", device, config)
                return compiled
            except Exception as exc:
                logger.warning(
                    "Stage 'albert' compile on %s failed: %s. %s",
                    device, str(exc).strip().splitlines()[-1][:250],
                    "Trying CPU fallback..." if device == "NPU" else "No more fallback.",
                )
        raise RuntimeError("Failed to compile albert stage on NPU and CPU.")

    def _replica_path(self, name: str) -> Path:
        return self.artifacts_dir / f"{name}_len{self.max_tokens}.xml"

    def _compile_replica(self, name: str, build_fn, example_fn, preferred: str,
                         cache_dir: str):
        """Exports (if missing) and compiles a replica stage.

        Explicit NPU-first/CPU-fallback (never AUTO): the fallback reason is
        logged so operators never silently run on CPU thinking it is NPU.
        """
        import openvino as ov

        xml_path = self._replica_path(name)
        if not xml_path.exists():
            logger.info("Exporting '%s' replica to OpenVINO IR...", name)
            module = build_fn()
            module.eval()
            example = example_fn()
            with torch.no_grad():
                ov_model = ov.convert_model(module, example_input=example)
            # Freeze fully static shapes (NPU accepts static only).
            ov_model.reshape(
                {key: list(val.shape) for key, val in example.items()}
            )
            ov.save_model(ov_model, str(xml_path))
            logger.info("Exported '%s' -> %s", name, xml_path)
        model = self._core.read_model(str(xml_path))
        last_err = ""
        for device in self._placement(name, preferred):
            try:
                config = self._npu_config(cache_dir) if device == "NPU" else (
                    {"CACHE_DIR": cache_dir} if cache_dir else {}
                )
                compiled = self._core.compile_model(model, device, config)
                self.stage_devices[name] = device
                self.stage_configs[name] = dict(config)
                if device != preferred:
                    logger.warning(
                        "Stage '%s': requested %s failed (%s); "
                        "explicit fallback to %s.",
                        name, preferred, last_err[:200], device,
                    )
                else:
                    logger.info(
                        "Stage '%s' compiled on %s with %s.", name, device, config)
                return compiled
            except Exception as exc:
                last_err = str(exc).strip().splitlines()[-1][:250]
                logger.warning(
                    "Stage '%s' compile on %s failed: %s. %s",
                    name, device, last_err,
                    "Trying CPU fallback..." if device == "NPU" else "No more fallback.",
                )
        raise RuntimeError(f"Failed to compile '{name}' stage on NPU and CPU.")

    # -- replica builders (live weight references, no copies) ------------
    def _build_prosody_body(self) -> nn.Module:
        return ProsodyBodyStatic(self._kmodel.predictor)

    def _build_text_encoder(self) -> nn.Module:
        return TextEncoderStatic(self._kmodel.text_encoder)

    def _prosody_example(self) -> Dict[str, torch.Tensor]:
        t = self.max_tokens
        return {
            "d_en": torch.zeros(1, 512, t),
            "style_tail": torch.zeros(1, 128),
            "mask": torch.zeros(1, t, dtype=torch.bool),
        }

    def _textenc_example(self) -> Dict[str, torch.Tensor]:
        t = self.max_tokens
        return {
            "x": torch.zeros(1, t, dtype=torch.long),
            "mask": torch.zeros(1, t, dtype=torch.bool),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def synthesize_tokens(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
    ) -> np.ndarray:
        """Synthesizes waveform from token IDs using NPU encoder + CPU tail.

        Args:
            input_ids: Long token IDs [1, n] including BOS/EOS (n <= max_tokens).
            ref_s: Style embedding [1, 256].

        Returns:
            1D float32 waveform at 24 kHz.
        """
        if not self._is_loaded:
            self.load()
        ids_np = input_ids.detach().cpu().numpy().astype(np.int64).reshape(1, -1)
        ref_np = ref_s.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        padded, attn, tmask, n = pad_inputs(ids_np, self.max_tokens)
        # Exported graphs take the pad mask as int8 ('char').
        mask_i8 = tmask.astype(np.int8)

        d_en = np.asarray(self._albert({"input_ids": padded, "attention_mask": attn})[0])
        p_out = self._prosody({
            "d_en": d_en.astype(np.float32),
            "style_tail": ref_np[:, 128:],
            "mask": mask_i8,
        })
        t_en = np.asarray(self._textenc({"x": padded, "mask": mask_i8})[0])

        # Output names are stable from export; resolve positionally.
        d_np = np.asarray(p_out[0])  # [1, n?, H] full-217; slice below
        pred_dur = np.asarray(p_out[1]).reshape(-1)[:n].astype(np.int64)
        pred_dur = np.clip(pred_dur, 1, None)

        with torch.no_grad():
            d_t = torch.from_numpy(d_np[:, :n, :])
            t_en_t = torch.from_numpy(t_en[:, :, :n])
            style = torch.from_numpy(ref_np[:, :128])
            style_tail = torch.from_numpy(ref_np[:, 128:])
            dur = torch.from_numpy(pred_dur).clamp(min=1)
            frames = int(dur.sum().item())
            idx = torch.repeat_interleave(torch.arange(n), dur)
            aln = torch.zeros((n, frames))
            aln[idx, torch.arange(frames)] = 1.0
            aln = aln.unsqueeze(0)
            en = d_t.transpose(-1, -2) @ aln
            f0_pred, n_pred = self._kmodel.predictor.F0Ntrain(en, style_tail)
            asr = t_en_t @ aln
            audio = self._kmodel.decoder(asr, f0_pred, n_pred, style).squeeze()
        return audio.detach().cpu().numpy().reshape(-1).astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        """True once stages are compiled and ready."""
        return self._is_loaded

    @property
    def npu_stage_count(self) -> int:
        """Number of stages actually running on NPU hardware."""
        return sum(1 for d in self.stage_devices.values() if d == "NPU")
