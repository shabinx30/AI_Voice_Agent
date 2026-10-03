"""Configuration settings for the Personal Assistant application.

This module loads and validates application configuration settings using
Pydantic Settings, providing strongly-typed access to configuration options
across all subsystems (OpenVINO STT, LM Studio LLM, Qwen TTS, and Server).
"""

from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration parameters.

    Attributes:
        server_host: Network host interface to bind the FastAPI server.
        server_port: Port number for the FastAPI server.
        debug: Flag indicating whether debug mode is enabled.
        stt_model_id: Hugging Face model identifier for OpenVINO Whisper.
        stt_device: OpenVINO target hardware device (CPU, GPU, NPU, or AUTO).
        stt_language: Default language code for Whisper transcription.
        stt_task: Whisper task type, either transcribe or translate.
        lm_studio_base_url: Base URL for LM Studio's OpenAI-compatible API.
        lm_studio_model: Identifier of the active model inside LM Studio.
        lm_studio_temperature: Sampling temperature for LLM generation.
        lm_studio_max_tokens: Maximum tokens to generate per response.
        lm_studio_system_prompt: System prompt instructing the assistant behavior.
        tts_model_id: Hugging Face repository ID for Qwen TTS.
        tts_device: Target compute device for Qwen TTS execution.
        tts_backend: Execution backend ('auto', 'torch', or 'openvino').
        tts_ov_model_id: Hugging Face ID of the INT8 OpenVINO Kokoro export.
        tts_speaker: Default voice persona identifier for Qwen TTS.
        tts_language: Synthesis language for Qwen TTS.
        audio_sample_rate: Standardized audio sample rate in Hz.
        auto_play_audio: Whether the server automatically plays synthesized audio.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Settings
    server_host: str = Field(default="0.0.0.0", description="Server bind host")
    server_port: int = Field(default=8000, description="Server bind port")
    debug: bool = Field(default=False, description="Debug mode")

    # Speech-to-Text (OpenVINO Whisper) Settings
    stt_model_id: str = Field(
        default="OpenVINO/whisper-base-int8-ov",
        description="Hugging Face OpenVINO Whisper INT8 model ID",
    )
    stt_device: str = Field(
        default="GPU",
        description="OpenVINO execution device (GPU, NPU, CPU, AUTO)",
    )
    stt_language: str = Field(
        default="en",
        description="Language code for Whisper transcription",
    )
    stt_task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Whisper task type",
    )

    # LLM (LM Studio) Settings
    lm_studio_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="LM Studio OpenAI-compatible API endpoint",
    )
    lm_studio_model: str = Field(
        default="qwen2.5-0.5b-instruct",
        description="Model identifier configured in LM Studio",
    )
    lm_studio_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature",
    )
    lm_studio_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling cutoff for LLM generation",
    )
    lm_studio_max_tokens: int = Field(
        default=2048,
        ge=-1,
        le=16384,
        description="Max output tokens for the LLM (-1 or 0 for model default / uncapped)",
    )
    lm_studio_system_prompt: str = Field(
        default=(
            "You are a helpful, intelligent voice assistant. Speak naturally in conversational speech. "
            "Provide clear, thorough, and engaging answers. Avoid raw markdown formatting, code blocks, or symbols "
            "that sound unnatural when read aloud."
        ),
        description="Default system instruction for the LLM",
    )
    lm_studio_think_mode: bool = Field(
        default=False,
        description="Whether reasoning/think mode is enabled for supported models",
    )
    lm_studio_reasoning_effort: Literal["low", "medium", "high", "max"] = Field(
        default="medium",
        description="Reasoning effort level sent to LM Studio when think mode is enabled",
    )

    # Text-to-Speech (Kokoro TTS) Settings
    tts_model_id: str = Field(
        default="hexgrad/Kokoro-82M",
        description="Hugging Face model ID for Kokoro TTS (torch backend)",
    )
    tts_device: str = Field(
        default="gpu",
        description="Compute device for Kokoro TTS execution (cpu, gpu, npu, or auto)",
    )
    tts_backend: Literal["auto", "torch", "openvino"] = Field(
        default="auto",
        description="TTS execution backend: auto (openvino for npu/auto, torch otherwise), torch, or openvino",
    )
    tts_ov_model_id: str = Field(
        default="OpenVINO/Kokoro-82M-int8-ov",
        description="Hugging Face model ID for the INT8 OpenVINO Kokoro export (openvino backend, NPU-capable)",
    )
    tts_npu_full: bool = Field(
        default=False,
        description="Run all Kokoro encoder stages on NPU (higher NPU load, lower fidelity) instead of the quality-first hybrid",
    )
    tts_npu_turbo: Literal["auto", "on", "off"] = Field(
        default="auto",
        description="NPU_TURBO mode for Kokoro encoder stages (auto=driver default, on/off force it). Only applied when supported by installed OpenVINO.",
    )
    tts_warmup_npu: bool = Field(
        default=False,
        description="Eagerly compile the static NPU encoder during warmup (slow first startup, fast first turn). False keeps NPU compile lazy on first synthesis.",
    )
    tts_speaker: str = Field(
        default="af_heart",
        description="Speaker voice persona for Kokoro TTS (e.g. af_heart, am_adam, af_bella, am_michael)",
    )
    tts_language: str = Field(
        default="a",
        description="Language code for Kokoro TTS ('a' for American English, 'b' for British English)",
    )

    # Audio Playback and Capture Settings
    audio_sample_rate: int = Field(
        default=16000,
        description="Standardized audio sampling rate (16 kHz for Whisper)",
    )
    auto_play_audio: bool = Field(
        default=False,
        description="Automatically play synthesized speech on the host audio output",
    )

    # Pipeline Performance Settings
    tts_max_workers: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Concurrent TTS synthesis workers for pipelined sentence synthesis",
    )
    tts_cache_size: int = Field(
        default=128,
        ge=0,
        le=1024,
        description="LRU cache entries for repeated TTS phrases (0 disables cache)",
    )
    max_history_turns: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of conversation turns (user+assistant pairs) kept per session",
    )
    sentence_min_chars: int = Field(
        default=12,
        ge=4,
        le=80,
        description="Minimum characters before a sentence split is emitted to TTS",
    )
    sentence_max_chars: int = Field(
        default=160,
        ge=60,
        le=400,
        description="Buffer length forcing a comma/colon phrase split",
    )
    tts_queue_max: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Bounded pending TTS sentences (backpressure, prevents overproduction)",
    )
    audio_queue_max: int = Field(
        default=8,
        ge=2,
        le=32,
        description="Bounded ready-audio chunks awaiting playback",
    )


# Singleton settings instance
settings = Settings()
