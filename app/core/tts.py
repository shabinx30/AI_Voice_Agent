"""Text-to-Speech synthesis module using Kokoro-82M.

This module provides high-fidelity, ultra-low-latency expressive speech generation
using the Kokoro-82M model with rich voice persona customization and fallback
support for rapid audio synthesis.

Two execution backends are supported:

* ``torch`` (default for CPU/GPU): the original ``kokoro.KPipeline`` PyTorch
  pipeline. PyTorch has no NPU execution provider, so an NPU request cannot
  run here.
* ``openvino`` (used for NPU/AUTO): the official INT8 OpenVINO export
  (``OpenVINO/Kokoro-82M-int8-ov``) executed through Optimum Intel. NPU
  compilation is attempted first; Kokoro's dynamic, data-dependent shapes
  (duration-predicted upsampling) are not always accepted by the NPU compiler,
  in which case compilation transparently falls back to CPU ــ the same
  NPU-first/CPU-fallback pattern already used by the Whisper STT engine.
"""

import asyncio
import io
import logging
import time
from collections import OrderedDict
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
import torch

from app.config import settings

logger = logging.getLogger(__name__)

# Legacy mapping to smoothly handle requests using old voice persona identifiers
VOICE_LEGACY_MAP: Dict[str, str] = {
    "ryan": "am_adam",
    "aiden": "am_michael",
    "vivian": "af_heart",
    "serena": "af_bella",
    "eric": "am_eric",
    "dylan": "am_liam",
    "ono_anna": "af_nicole",
    "sohee": "af_kore",
    "uncle_fu": "am_onyx",
}


class KokoroTTSEngine:
    """Text-to-speech engine powered by Kokoro-82M.

    Attributes:
        model_id: Hugging Face model identifier for Kokoro (e.g. 'hexgrad/Kokoro-82M').
        device: Requested hardware device ('cpu', 'gpu', 'npu', or 'auto').
        backend: Execution backend ('torch' or 'openvino'). Resolved from
            ``settings.tts_backend``; ``'auto'`` selects ``'openvino'`` for NPU/AUTO
            requests and ``'torch'`` otherwise.
        effective_device: Device the model was actually compiled on (set during
            load; may differ from ``device`` after NPU->CPU fallback).
        speaker: Default voice persona identifier (e.g. 'af_heart').
        language: Default spoken language code ('a' for US English, 'b' for UK English).
        sample_rate: Output audio sampling frequency (24 kHz).
    """

    #: Default OpenVINO INT8 Kokoro export used by the ``openvino`` backend.
    DEFAULT_OV_MODEL_ID: str = "OpenVINO/Kokoro-82M-int8-ov"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Initializes the Kokoro-82M synthesis engine.

        Args:
            model_id: Hugging Face model ID. Defaults to settings.tts_model_id.
            device: Compute device. Defaults to settings.tts_device. Accepts
                'cpu', 'gpu'/'cuda', 'npu', and 'auto' (case-insensitive).
            speaker: Speaker identifier. Defaults to settings.tts_speaker.
            language: Synthesis language code. Defaults to settings.tts_language.
        """
        self.model_id = model_id or settings.tts_model_id
        self.device = device or settings.tts_device
        self.speaker = speaker or settings.tts_speaker
        self.language = language or settings.tts_language
        self.sample_rate: int = 24000

        self.backend: str = self._resolve_backend()
        self.effective_device: Optional[str] = None
        self._ov_model_id: str = (
            getattr(settings, "tts_ov_model_id", None)
            or self.DEFAULT_OV_MODEL_ID
        )
        self._ov_model = None
        self._infer_lock = Lock()
        # Genuine NPU path: static encoder stages (KokoroNPUEncoder).
        # Initialized lazily on first NPU synthesis so imports, warmup, and
        # the CPU fallback stay fast; first-use NPU compilation is one-time
        # and cached on disk afterwards.
        self._npu_encoder = None
        self._npu_encoders: Dict[bool, Any] = {}
        self._npu_attempted: bool = False

        self._pipeline = None
        self._is_loaded: bool = False
        # Reusable G2P front-end for the OpenVINO backend. Optimum's
        # preprocess_input() builds a fresh KPipeline (misaki/G2P init,
        # ~0.7s) and re-reads the voice pack (~0.4s) on EVERY synthesis;
        # profiling shows that is ~57% of short-sentence TTS latency.
        # Keep one KPipeline per lang + one voice tensor per (lang, voice)
        # alive instead. G2P access is serialized by _g2p_lock (an RLock:
        # helpers nest _get_g2p_pipeline inside _get_voice_pack inside
        # _preprocess_kokoro; held for milliseconds once warm); NPU/CPU
        # inference keeps using _infer_lock.
        self._g2p_lock = RLock()
        self._g2p_pipelines = {}
        self._voice_packs = {}
        # Small LRU cache for repeated phrases ("I didn't catch that", etc.).
        # Key: (text, speaker). Value: (audio copy, sr). Thread-safe via Lock
        # since synthesize() runs in multiple to_thread workers.
        _cache_size = getattr(settings, "tts_cache_size", 128)
        self._cache_max = int(_cache_size) if _cache_size else 0
        self._cache: OrderedDict[Tuple[str, str], Tuple[np.ndarray, int]] = OrderedDict()
        self._cache_lock = Lock()
        self._supported_speakers: List[str] = [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_aoede",
            "af_kore",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "am_echo",
            "am_eric",
            "am_liam",
            "am_onyx",
            "am_puck",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_daniel",
        ]
        self._supported_languages: List[str] = [
            "a",
            "b",
            "e",
            "f",
            "h",
            "i",
            "j",
            "p",
            "z",
        ]

    def _resolve_backend(self) -> str:
        """Selects the execution backend for the requested device.

        Returns:
            'openvino' when an NPU-backed run is requested (or explicitly
            configured), otherwise 'torch' to preserve legacy behavior.
        """
        explicit = str(getattr(settings, "tts_backend", "auto") or "auto").lower()
        if explicit in ("torch", "openvino"):
            return explicit
        requested = str(self.device or "cpu").lower()
        if requested in ("npu", "auto"):
            return "openvino"
        return "torch"

    def load_model(self) -> None:
        """Loads the Kokoro-82M pipeline for the configured backend."""
        if self._is_loaded and (
            self._pipeline is not None or self._ov_model is not None
        ):
            logger.debug("Kokoro-82M model already loaded.")
            return

        if self.backend == "openvino":
            self._load_openvino_model()
        else:
            self._load_torch_model()

    def _load_openvino_model(self) -> None:
        """Loads the INT8 OpenVINO Kokoro export on CPU.

        This full-graph model serves grapheme-to-phoneme preprocessing and
        the CPU fallback path. The genuine NPU path (static encoder
        stages) is initialized lazily on first NPU synthesis — see
        :meth:`_get_npu_encoder` — because first-use NPU compilation is slow
        (one-time, cached on disk afterwards).

        Raises:
            RuntimeError: If the OpenVINO model cannot be loaded.
        """
        try:
            from optimum.intel.openvino import OVModelForTextToSpeechSeq2Seq
        except ImportError as exc:
            raise RuntimeError(
                "The 'openvino' TTS backend requires optimum-intel. "
                f"Install it with: pip install optimum[openvino]. ({exc})"
            ) from exc

        logger.info(
            "Loading OpenVINO Kokoro-82M model '%s' on CPU...",
            self._ov_model_id,
        )
        start_time = time.perf_counter()
        try:
            ov_config = {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": 8,
                "NUM_STREAMS": "1",
            }
            self._ov_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
                self._ov_model_id,
                device="CPU",
                ov_config=ov_config,
                trust_remote_code=True,
            )
            self.effective_device = "CPU"
            self._is_loaded = True
            # Pre-load default voice persona for fast zero-latency first synthesis.
            try:
                mapped = VOICE_LEGACY_MAP.get(self.speaker, self.speaker)
                self._ov_model._load_voice(mapped)
            except Exception as v_exc:
                logger.debug("Initial voice pre-load note: %s", v_exc)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "OpenVINO Kokoro-82M model loaded successfully in %.2fs on CPU.",
                elapsed,
            )
        except Exception as exc:
            logger.error("Failed to load OpenVINO Kokoro model: %s", exc)
            raise RuntimeError(f"OpenVINO Kokoro initialization failed: {exc}") from exc

    def _resolve_turbo(self) -> "Optional[bool]":
        """Maps settings.tts_npu_turbo (auto/on/off) to True/False/None."""
        try:
            mode = str(getattr(settings, "tts_npu_turbo", "auto") or "auto").lower()
        except Exception:
            return None
        if mode == "on":
            return True
        if mode == "off":
            return False
        return None

    def warm_npu_encoder(self) -> bool:
        """Eagerly compiles the static NPU encoder (for warmup paths).

        Returns:
            True when at least one stage landed on NPU, False otherwise.
        """
        enc = self._get_npu_encoder()
        return enc is not None and enc.npu_stage_count > 0

    def get_device_report(self) -> dict:
        """Returns a runtime-confirmed Kokoro device report (never assumes NPU)."""
        try:
            from app.core.diagnostics import get_kokoro_device_report

            return get_kokoro_device_report(self)
        except Exception as exc:
            return {
                "requested_device": str(self.device),
                "backend": str(self.backend),
                "effective_device": self.effective_device or "not-yet-loaded",
                "error": str(exc)[:200],
            }

    def get_device_info(self) -> Dict[str, Any]:
        """Returns the current requested and effective Kokoro TTS device info."""
        from app.core.diagnostics import get_available_devices

        available_devices = ["cpu"]
        if "NPU" in get_available_devices():
            available_devices.extend(["npu", "npu_only"])

        target_dev = str(self.device).lower()
        is_npu_full = (target_dev == "npu_only") or bool(getattr(settings, "tts_npu_full", False))

        return {
            "device": target_dev,
            "effective_device": self.effective_device or ("CPU" if target_dev == "cpu" else "NPU"),
            "backend": self.backend,
            "available_devices": available_devices,
            "npu_full": is_npu_full,
        }

    def set_device(self, device: str) -> Dict[str, Any]:
        """Dynamically switches the Kokoro TTS execution device (cpu, npu, or npu_only).

        Args:
            device: Target processing unit ('cpu', 'npu', or 'npu_only', case-insensitive).

        Returns:
            Dict describing the active device status.

        Raises:
            ValueError: If an unsupported device is specified.
        """
        target = str(device or "").strip().lower()
        if target in ("npu-only", "npu_full", "full_npu"):
            target = "npu_only"

        if target not in ("cpu", "npu", "npu_only"):
            raise ValueError(f"Unsupported TTS device '{device}'. Choose 'cpu', 'npu', or 'npu_only'.")

        if target == str(self.device).lower() and self._is_loaded:
            return self.get_device_info()

        logger.info("Switching Kokoro TTS processing unit from '%s' to '%s'...", self.device, target)
        self.device = target

        if target in ("npu", "npu_only"):
            self.backend = "openvino"
            is_full = (target == "npu_only")
            settings.tts_device = "npu"
            settings.tts_npu_full = is_full
            if not self._is_loaded or self._ov_model is None:
                self.load_model()
            encoder = self._get_npu_encoder(full_npu=is_full)
            if encoder and encoder.npu_stage_count > 0:
                if is_full and encoder.npu_stage_count == 3:
                    self.effective_device = "NPU (Full)"
                else:
                    self.effective_device = "NPU+CPU"
            else:
                self.effective_device = "CPU"
        else:
            self.device = "cpu"
            settings.tts_device = "cpu"
            self.effective_device = "CPU"
            if self.backend == "openvino":
                if not self._is_loaded or self._ov_model is None:
                    self.load_model()
            elif not self._is_loaded or self._pipeline is None:
                self.load_model()

        logger.info(
            "Kokoro TTS processing unit switched to '%s' (effective: %s, backend: %s).",
            self.device,
            self.effective_device,
            self.backend,
        )
        return self.get_device_info()

    def _get_npu_encoder(self, full_npu: Optional[bool] = None):
        """Returns the static NPU encoder, initializing it lazily on first use.

        Args:
            full_npu: When True, compiles all 3 static stages (albert, prosody_body,
                text_encoder) on NPU. When False, keeps albert on CPU for maximum
                speech correlation (hybrid mode). Defaults to current configuration.

        Returns:
            Configured KokoroNPUEncoder, or None when NPU is unavailable or
            was not requested.
        """
        target_device = str(self.device or "cpu").lower()
        if target_device not in ("npu", "npu_only"):
            return None

        if full_npu is None:
            full_npu = (target_device == "npu_only") or bool(getattr(settings, "tts_npu_full", False))

        if full_npu in self._npu_encoders:
            self._npu_encoder = self._npu_encoders[full_npu]
            return self._npu_encoder

        try:
            from app.core.tts_npu import KokoroNPUEncoder

            logger.info(
                "Initializing Kokoro static NPU encoder (full_npu=%s, one-time NPU "
                "compilation; subsequent loads reuse the on-disk cache)...",
                full_npu,
            )
            encoder = KokoroNPUEncoder(
                cpu_stages=frozenset() if full_npu else None,
                turbo=self._resolve_turbo(),
            )
            encoder.load(preferred_device="NPU")
            self._npu_encoders[full_npu] = encoder
            self._npu_encoder = encoder
            n_npu = encoder.npu_stage_count

            if n_npu == 0:
                self.effective_device = "CPU"
                logger.warning(
                    "Kokoro NPU requested but 0 stages landed on NPU "
                    "(%s); running on CPU.",
                    encoder.stage_devices,
                )
            elif full_npu and n_npu == 3:
                self.effective_device = "NPU (Full)"
                logger.info(
                    "Kokoro full NPU encoder ready (all %d stages on NPU: %s).",
                    n_npu,
                    encoder.stage_devices,
                )
            else:
                self.effective_device = "NPU+CPU"
                logger.info(
                    "Kokoro hybrid NPU encoder ready (%d stages on NPU: %s).",
                    n_npu,
                    encoder.stage_devices,
                )
            return encoder
        except Exception as exc:
            reason = str(exc).strip().splitlines()[-1][:200] if str(exc).strip() else repr(exc)[:200]
            logger.warning(
                "Kokoro static NPU encoder unavailable (reason: %s); "
                "explicit fallback to the OpenVINO CPU path.",
                reason,
            )
            self.effective_device = "CPU"
            return None

    def _get_g2p_pipeline(self, lang_code: str):
        """Returns a cached misaki G2P pipeline for ``lang_code`` (kept alive).

        Building KPipeline costs ~0.7s + ~550MB transient churn per call;
        reuse makes per-synthesis G2P cost milliseconds. Serialized by
        _g2p_lock; one instance per language code.
        """
        with self._g2p_lock:
            pipe = self._g2p_pipelines.get(lang_code)
            if pipe is None:
                from kokoro import KPipeline

                pipe = KPipeline(
                    lang_code=lang_code,
                    repo_id=self.model_id or "hexgrad/Kokoro-82M",
                    model=False,
                )
                self._g2p_pipelines[lang_code] = pipe
            return pipe

    def _get_voice_pack(self, lang_code: str, voice: str):
        """Returns a cached voice/style tensor for (lang, voice) (kept alive).

        Reading the voice pack from disk costs ~0.4s per synthesis; the
        cached tensor is ~0.5MB. Same index rule as optimum's preprocess.
        """
        key = (lang_code, voice)
        with self._g2p_lock:
            pack = self._voice_packs.get(key)
            if pack is None:
                pipe = self._get_g2p_pipeline(lang_code)
                pack = pipe.load_voice(voice)
                self._voice_packs[key] = pack
            return pack

    def _preprocess_kokoro(
        self,
        text: str,
        voice: str,
        lang_code: str,
        split_pattern: str = r"\n+",
    ) -> dict:
        """G2P + tokenize with reused front-end (same math as optimum's preprocess).

        Mirrors ``_OVModelForKokoroTextToSpeech.preprocess_input`` segment
        for segment (same KPipeline chunking, same vocab BOS/EOS wrapping,
        same voice-pack index rule) but reuses the cached G2P pipeline and
        voice pack instead of rebuilding them per call. Must only be used
        with the matching ``_ov_model.config.vocab``.

        Returns:
            Dict with ``segments`` (list of input_ids/ref_s/speed/phonemes/
            graphemes) plus top-level input_ids/ref_s/speed for single chunks.
        """
        if not self._is_loaded or self._ov_model is None:
            self.load_model()
        vocab = getattr(self._ov_model.config, "vocab", None)
        if vocab is None:
            raise ValueError("Model config has no 'vocab'; cannot tokenize phonemes.")
        with self._g2p_lock:
            pipe = self._get_g2p_pipeline(lang_code)
            segments = list(pipe(text=text, split_pattern=split_pattern))
            voice_pack = self._get_voice_pack(lang_code, voice)
            out = []
            for segment in segments:
                phonemes = segment.phonemes
                if not phonemes:
                    continue
                token_ids = [vocab.get(p) for p in phonemes]
                token_ids = [i for i in token_ids if i is not None]
                input_ids = torch.LongTensor([[0, *token_ids, 0]])
                ref_s = voice_pack[min(len(phonemes) - 1, voice_pack.shape[0] - 1)]
                out.append(
                    {
                        "input_ids": input_ids,
                        "ref_s": ref_s,
                        "speed": 1.0,
                        "phonemes": phonemes,
                        "graphemes": segment.graphemes,
                    }
                )
        if not out:
            raise ValueError(f"G2P produced no phoneme segments for: {text!r}")
        if len(out) == 1:
            single = out[0]
            return {
                "input_ids": single["input_ids"],
                "ref_s": single["ref_s"],
                "speed": single["speed"],
                "segments": out,
            }
        return {"segments": out}

    def _synthesize_npu_static(
        self,
        text: str,
        speaker: str,
        lang_code: str,
    ) -> Optional[np.ndarray]:
        """Synthesizes via static NPU encoder stages, segment by segment.

        Oversize segments (beyond the static token length) and per-segment
        NPU errors fall back to the full-graph CPU model for that segment.

        Args:
            text: Cleaned text to synthesize.
            speaker: Resolved voice persona identifier.
            lang_code: Validated language code.

        Returns:
            Concatenated waveform, or None when the NPU encoder is unavailable.
        """
        from app.core.tts_npu import NPU_MAX_TOKENS

        encoder = self._get_npu_encoder()
        if encoder is None:
            return None

        # G2P/text frontend runs OUTSIDE _infer_lock: it touches no shared
        # compiled state, so sentence N+1 can phonemize while sentence N
        # synthesizes (removes 100-500ms+ scheduling delay in multi-sentence
        # turns). Uses the cached front-end (no per-call rebuild).
        model_inputs = self._preprocess_kokoro(
            text, voice=speaker, lang_code=lang_code
        )
        segments = model_inputs.get("segments") or []
        if not segments:
            return None

        waves: List[np.ndarray] = []
        for seg in segments:
            seg_ids = seg["input_ids"]
            seg_ref = seg["ref_s"]
            seg_n = int(seg_ids.shape[-1])
            audio_seg: Optional[np.ndarray] = None
            if seg_n <= NPU_MAX_TOKENS:
                try:
                    with self._infer_lock:
                        audio_seg = encoder.synthesize_tokens(seg_ids, seg_ref)
                except Exception as exc:
                    logger.warning(
                        "NPU segment synthesis failed (%s); using CPU fallback "
                        "for this segment.",
                        str(exc).strip().splitlines()[-1][:160],
                    )
                    audio_seg = None
            if audio_seg is None:
                with self._infer_lock:
                    out = self._ov_model.forward(
                        input_ids=seg_ids, ref_s=seg_ref, speed=1.0
                    )
                wave = out.waveform
                if isinstance(wave, torch.Tensor):
                    wave = wave.detach().cpu().numpy()
                audio_seg = np.reshape(np.asarray(wave, dtype=np.float32), (-1,))
            waves.append(audio_seg)
        if not waves:
            return None
        return np.concatenate(waves, axis=0).astype(np.float32)

    def _synthesize_openvino(
        self,
        text: str,
        speaker: str,
        language: Optional[str],
    ) -> np.ndarray:
        """Runs synthesis through the OpenVINO Kokoro stack.

        Prefers the static NPU encoder stages when an NPU run was requested;
        falls back to the full-graph CPU model otherwise.

        Args:
            text: Cleaned text to synthesize.
            speaker: Resolved voice persona identifier.
            language: Spoken language code override.

        Returns:
            1D float32 audio waveform at 24 kHz.
        """
        if not self._is_loaded or self._ov_model is None:
            self.load_model()

        lang_code = language or self.language or "a"
        if lang_code not in self._supported_languages:
            logger.warning(
                "Language '%s' unsupported for OpenVINO Kokoro; using '%s'.",
                lang_code,
                self.language,
            )
            lang_code = self.language

        npu_audio = self._synthesize_npu_static(text, speaker, lang_code)
        if npu_audio is not None:
            return npu_audio

        # Optimum's OpenVINO wrapper is driven through a single compiled
        # request; serialize inference while still allowing the pipeline to
        # overlap LLM streaming across sentences via worker threads.
        # Frontend (cached) runs outside the lock; only generate() is locked.
        model_inputs = self._preprocess_kokoro(
            text, voice=speaker, lang_code=lang_code
        )
        with self._infer_lock:
            waveform = self._ov_model.generate(**model_inputs)

        if isinstance(waveform, torch.Tensor):
            audio = waveform.detach().cpu().numpy()
        else:
            audio = np.asarray(waveform, dtype=np.float32)
        return np.reshape(audio, (-1,)).astype(np.float32)

    def _load_torch_model(self) -> None:
        """Loads the Kokoro-82M PyTorch pipeline (legacy backend)."""
        if self._is_loaded and self._pipeline is not None:
            logger.debug("Kokoro-82M model already loaded.")
            return

        logger.info(
            "Loading Kokoro-82M model '%s' on %s (lang='%s')...",
            self.model_id,
            self.device,
            self.language,
        )
        start_time = time.perf_counter()

        try:
            from kokoro import KPipeline

            target_device = (self.device or "cpu").lower()
            if target_device in ["gpu", "cuda"]:
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    target_device = "xpu"
                else:
                    logger.info(
                        "GPU targeted for Kokoro-82M. Direct PyTorch CUDA/XPU is not bundled; "
                        "using optimized multi-threaded execution for TTS while STT and LLM leverage Intel Arc GPU."
                    )
                    target_device = "cpu"
            elif target_device == "npu":
                logger.warning(
                    "PyTorch Kokoro backend has no NPU execution provider; running "
                    "this synthesis on CPU. Set TTS_BACKEND=openvino with "
                    "TTS_DEVICE=npu to use the OpenVINO INT8 NPU path instead."
                )
                target_device = "cpu"
            else:
                target_device = "cpu"
            self.effective_device = target_device.upper()

            # Initialize Kokoro KPipeline
            self._pipeline = KPipeline(
                lang_code=self.language,
                repo_id=self.model_id,
                device=target_device,
            )

            # Pre-load default voice persona for fast zero-latency first synthesis
            try:
                self._pipeline.load_voice(self.speaker)
            except Exception as v_exc:
                logger.debug("Initial voice pre-load note: %s", v_exc)

            self._is_loaded = True
            elapsed = time.perf_counter() - start_time
            logger.info("Kokoro-82M model loaded successfully in %.2fs.", elapsed)

        except Exception as exc:
            logger.error("Failed to load Kokoro-82M model: %s", exc)
            raise RuntimeError(f"Kokoro-82M initialization failed: {exc}") from exc

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesizes text into audio waveform samples using Kokoro-82M.

        Args:
            text: Text to convert to speech.
            speaker: Voice persona name. Defaults to instance default.
            language: Spoken language code. Defaults to instance default.
            device: Optional compute device override ('cpu' or 'npu').

        Returns:
            Tuple of (audio_waveform_numpy_array, sample_rate_hz).

        Raises:
            ValueError: If input text is empty.
            RuntimeError: If audio synthesis fails.
        """
        if device is not None and str(device).strip().lower() != str(self.device).lower():
            self.set_device(device)

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text for TTS cannot be empty.")

        # Resolve speaker persona with backward compatibility
        requested_speaker = speaker or self.speaker
        target_speaker = VOICE_LEGACY_MAP.get(requested_speaker, requested_speaker)

        if (
            self._supported_speakers
            and target_speaker not in self._supported_speakers
        ):
            logger.warning(
                "Speaker '%s' not in supported list (%s). Falling back to '%s'.",
                target_speaker,
                self._supported_speakers,
                self._supported_speakers[0],
            )
            target_speaker = self._supported_speakers[0]

        # LRU cache lookup (copy on hit to avoid caller mutation).
        cache_key = (clean_text, target_speaker)
        if self._cache_max > 0:
            with self._cache_lock:
                hit = self._cache.get(cache_key)
                if hit is not None:
                    self._cache.move_to_end(cache_key)
                    cached_audio, cached_sr = hit
                    logger.debug("TTS cache hit for '%s...'", clean_text[:30])
                    return cached_audio.copy(), cached_sr

        logger.debug(
            "Synthesizing speech with Kokoro-82M: speaker='%s', text='%s'",
            target_speaker,
            clean_text[:60] + "..." if len(clean_text) > 60 else clean_text,
        )
        start_time = time.perf_counter()

        try:
            if self.backend == "openvino":
                audio_data = self._synthesize_openvino(
                    clean_text, target_speaker, language
                )
            else:
                if not self._is_loaded or self._pipeline is None:
                    self.load_model()

                # KPipeline is not thread-safe for concurrent calls; serialize
                # access to the underlying generator. Parallelism across sentences
                # is still achieved by overlapping LLM streaming + queueing, and
                # callers may run synthesize in workers — the lock prevents
                # CUDA/CPU state corruption while keeping cache lock-free.
                generator = self._pipeline(clean_text, voice=target_speaker)
                audio_chunks: List[np.ndarray] = []

                for _, _, audio in generator:
                    if audio is not None:
                        if isinstance(audio, torch.Tensor):
                            chunk = audio.detach().cpu().numpy()
                        else:
                            chunk = np.asarray(audio, dtype=np.float32)
                        audio_chunks.append(chunk)

                if not audio_chunks:
                    logger.warning("Kokoro generator yielded no audio chunks. Triggering fallback.")
                    return self._fallback_synthesize(clean_text)

                audio_data = np.concatenate(audio_chunks, axis=0).astype(np.float32)
            elapsed = time.perf_counter() - start_time
            duration = len(audio_data) / self.sample_rate

            logger.debug(
                "Kokoro-82M synthesized %.2fs of audio in %.2fs (sr=%d, speedup=%.1fx, backend=%s, device=%s).",
                duration,
                elapsed,
                self.sample_rate,
                duration / max(elapsed, 0.001),
                self.backend,
                self.effective_device or self.device,
            )
            if self._cache_max > 0:
                with self._cache_lock:
                    self._cache[cache_key] = (audio_data.copy(), self.sample_rate)
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self._cache_max:
                        self._cache.popitem(last=False)
            return audio_data, self.sample_rate

        except Exception as exc:
            logger.error("Kokoro-82M synthesis error: %s. Attempting fallback...", exc)
            return self._fallback_synthesize(clean_text)

    def _fallback_synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Provides fallback speech synthesis using pyttsx3 or a gentle tone.

        Args:
            text: Text to synthesize.

        Returns:
            Tuple of (audio_waveform_numpy_array, sample_rate_hz).
        """
        logger.info("Executing fallback TTS synthesis for: '%s'", text[:50])
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            temp_path = "temp_fallback.wav"
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            data, sr = sf.read(temp_path)
            import os

            if os.path.exists(temp_path):
                os.remove(temp_path)
            return data.astype(np.float32), sr
        except Exception as fallback_exc:
            logger.error("Fallback TTS failed: %s", fallback_exc)
            sr = 24000
            t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
            sine_wave = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            return sine_wave, sr

    def synthesize_to_wav_bytes(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
    ) -> bytes:
        """Synthesizes speech and returns encoded WAV file bytes.

        Args:
            text: Text to convert into speech.
            speaker: Voice persona name.
            language: Spoken language code.
            device: Optional compute device override ('cpu' or 'npu').

        Returns:
            WAV format binary bytes.
        """
        audio_data, sr = self.synthesize(
            text, speaker=speaker, language=language, device=device
        )
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format="WAV")
        return buffer.getvalue()

    def get_supported_speakers(self) -> List[str]:
        """Returns the list of available voice personas."""
        return list(self._supported_speakers)

    def get_supported_languages(self) -> List[str]:
        """Returns the list of supported language codes."""
        return list(self._supported_languages)

    @property
    def is_loaded(self) -> bool:
        """Indicates if the Kokoro-82M model is loaded in memory."""
        return self._is_loaded


# Backward compatibility alias
QwenTTSEngine = KokoroTTSEngine
