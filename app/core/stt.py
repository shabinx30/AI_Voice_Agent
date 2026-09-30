"""Speech-to-Text module using OpenVINO and Whisper Base INT8.

This module encapsulates OpenVINO inference for Whisper Base INT8, delivering
low-latency, quantized speech transcription optimized for Intel CPUs, GPUs,
and NPUs.
"""

import logging
import time
from typing import Dict, List, Optional, Union
import numpy as np
import openvino as ov
import torch
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor

from app.config import settings

logger = logging.getLogger(__name__)


class OpenVINOWhisperSTT:
    """Speech-to-text service leveraging OpenVINO-quantized Whisper Base INT8.

    Attributes:
        model_id: Hugging Face repository or local path for the model.
        device: Target OpenVINO compute hardware ('CPU', 'GPU', 'NPU', 'AUTO').
        language: ISO language code for transcription (e.g., 'en').
        task: Whisper task type ('transcribe' or 'translate').
        processor: Hugging Face AutoProcessor for audio feature extraction.
        model: OpenVINO speech sequence-to-sequence model instance.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        """Initializes the OpenVINO Whisper Base INT8 speech-to-text pipeline.

        Args:
            model_id: Model repository ID. Defaults to settings.stt_model_id.
            device: OpenVINO device. Defaults to settings.stt_device.
            language: Target language. Defaults to settings.stt_language.
            task: Task type. Defaults to settings.stt_task.
        """
        self.model_id = model_id or settings.stt_model_id
        self.device = (device or settings.stt_device).upper()
        self.language = language or settings.stt_language
        self.task = task or settings.stt_task

        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[OVModelForSpeechSeq2Seq] = None
        self._is_loaded: bool = False

    def load_model(self) -> None:
        """Loads and compiles the Whisper Base INT8 model on the OpenVINO runtime.

        Raises:
            RuntimeError: If model loading or compilation fails.
        """
        if self._is_loaded:
            logger.debug("Whisper Base INT8 model already loaded.")
            return

        logger.info(
            "Loading OpenVINO Whisper model '%s' on device '%s'...",
            self.model_id,
            self.device,
        )
        start_time = time.perf_counter()

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            if "NPU" in self.device:
                logger.info(
                    "Configuring Whisper Base INT8 for NPU (Intel AI Boost)..."
                )
                self.model = OVModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    compile=False,
                )
                # Reshape encoder to static shape [1, 80, 3000] for NPU VPUX compiler
                self.model.encoder.model.reshape({"input_features": [1, 80, 3000]})
                self.model.encoder._device_override = "NPU"
                logger.info("Compiling encoder on NPU (Intel AI Boost)...")
                self.model.encoder.compile()

                # Determine decoder target device (GPU if available, else CPU)
                available = self.get_openvino_devices()
                dec_device = "GPU" if "GPU" in available else "CPU"
                self.model.decoder._device_override = dec_device
                logger.info("Compiling decoder on %s...", dec_device)
                self.model.decoder.compile()

                if self.model.decoder_with_past:
                    self.model.decoder_with_past._device_override = dec_device
                    self.model.decoder_with_past.compile()

                self.model._is_compiled = True
                self.device = f"NPU+{dec_device}"
            else:
                self.model = OVModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    compile=True,
                    device=self.device,
                )

            self._is_loaded = True
            elapsed = time.perf_counter() - start_time
            logger.info(
                "OpenVINO Whisper Base INT8 loaded successfully in %.2fs on %s.",
                elapsed,
                self.device,
            )
        except Exception as exc:
            logger.error("Failed to load OpenVINO Whisper model: %s", exc)
            # Graceful fallback to CPU if GPU or NPU fails compilation
            if self.device != "CPU":
                logger.warning("Attempting fallback compilation on CPU...")
                try:
                    self.device = "CPU"
                    self.model = OVModelForSpeechSeq2Seq.from_pretrained(
                        self.model_id,
                        compile=True,
                        device="CPU",
                    )
                    self._is_loaded = True
                    logger.info("Fallback to CPU compilation succeeded.")
                    return
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Failed both {self.device} and CPU compilation: "
                        f"{fallback_exc}"
                    ) from fallback_exc
            raise RuntimeError(
                f"Error initializing OpenVINO Whisper model: {exc}"
            ) from exc

    def transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """Transcribes raw 1D audio waveform samples to text.

        Args:
            audio_array: 1D numpy array with float32 audio samples.
            sample_rate: Audio sampling frequency in Hz (Whisper expects 16 kHz).
            language: Optional language override for this utterance.

        Returns:
            Decoded transcription string.

        Raises:
            RuntimeError: If model is not loaded or inference fails.
            ValueError: If input audio array is empty.
        """
        if not self._is_loaded or self.model is None or self.processor is None:
            self.load_model()

        if audio_array.size == 0:
            raise ValueError("Input audio array is empty.")

        target_lang = language or self.language
        logger.debug(
            "Transcribing audio: shape=%s, sr=%d, lang=%s",
            audio_array.shape,
            sample_rate,
            target_lang,
        )
        start_time = time.perf_counter()

        try:
            # Extract log-mel spectrogram features
            inputs = self.processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features

            # Execute OpenVINO inference
            predicted_ids = self.model.generate(
                input_features,
                language=target_lang,
                task=self.task,
            )

            # Decode token sequence to text
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0].strip()

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Transcription completed in %.3fs: '%s'",
                elapsed,
                transcription,
            )
            return transcription

        except Exception as exc:
            logger.error("Whisper transcription inference error: %s", exc)
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    def get_openvino_devices(self) -> List[str]:
        """Discovers all hardware accelerators available to OpenVINO.

        Returns:
            List of detected device strings (e.g. ['CPU', 'GPU', 'NPU']).
        """
        try:
            core = ov.Core()
            return core.available_devices
        except Exception as exc:
            logger.warning("Failed to query OpenVINO devices: %s", exc)
            return ["CPU"]

    @property
    def is_loaded(self) -> bool:
        """Returns True if the OpenVINO model is compiled and ready."""
        return self._is_loaded
