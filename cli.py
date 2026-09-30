"""Command-line interface (CLI) for the Personal Assistant.

Provides interactive terminal execution allowing users to talk directly
through their microphone, type text prompts, or process audio files with
OpenVINO Whisper STT, LM Studio LLM, and Qwen TTS audio playback.
"""

import argparse
import asyncio
import logging
import sys
import time
from typing import Optional

from app.config import settings
from app.core.audio import AudioProcessor
from app.core.pipeline import AssistantPipeline

# Setup console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("assistant-cli")


def print_banner() -> None:
    """Displays the terminal welcome banner and operational summary."""
    print("=" * 65)
    print("      NexusVoice Personal Assistant (OpenVINO + LM Studio + Qwen TTS)")
    print("=" * 65)
    print(f"  • STT Model:   {settings.stt_model_id} (OpenVINO {settings.stt_device})")
    print(f"  • LLM Target:  {settings.lm_studio_base_url} ({settings.lm_studio_model})")
    print(f"  • TTS Model:   {settings.tts_model_id} (Speaker: {settings.tts_speaker})")
    print("=" * 65)
    print()


async def run_voice_loop(
    pipeline: AssistantPipeline,
    duration: float = 4.0,
    speaker: Optional[str] = None,
) -> None:
    """Runs an interactive microphone voice loop.

    Args:
        pipeline: Initialized AssistantPipeline instance.
        duration: Duration in seconds to record per turn.
        speaker: Voice persona name.
    """
    print("Interactive Voice Mode Active.")
    print(f"Press [Enter] to record {duration:.1f}s from your microphone, or type 'q' + Enter to quit.\n")

    while True:
        try:
            user_input = input("Press [Enter] to record (or 'q' to quit) > ").strip().lower()
            if user_input in ("q", "quit", "exit"):
                print("Exiting voice assistant. Goodbye!")
                break

            print(f"\n[Recording] Speak into your microphone for {duration:.1f}s...")
            audio_array = AudioProcessor.record_microphone(
                duration_seconds=duration, sample_rate=16000
            )
            wav_bytes = AudioProcessor.to_wav_bytes(audio_array, 16000)

            print("[Processing] Transcribing with OpenVINO Whisper Base INT8...")
            def _on_voice_chunk(chunk):
                print(f"  [Speaking chunk {chunk.sentence_index + 1}]: {chunk.text}")

            res = await pipeline.process_audio_bytes(
                audio_bytes=wav_bytes,
                speaker=speaker,
                play_audio=True,
                on_chunk=_on_voice_chunk,
            )

            print(f"\n>>> You Said:       {res.user_text}")
            print(f">>> Assistant:      {res.assistant_text}")
            print(
                f"[Metrics] STT: {res.metrics.stt_latency_ms}ms | "
                f"LLM: {res.metrics.llm_latency_ms}ms | "
                f"TTS: {res.metrics.tts_latency_ms}ms | "
                f"TTFA: {res.metrics.ttfa_ms}ms | "
                f"Total: {res.metrics.total_latency_ms}ms\n"
            )

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


async def run_text_loop(
    pipeline: AssistantPipeline,
    speaker: Optional[str] = None,
) -> None:
    """Runs an interactive text chat loop with spoken voice responses.

    Args:
        pipeline: Initialized AssistantPipeline instance.
        speaker: Voice persona name.
    """
    print("Interactive Text-to-Speech Mode Active.")
    print("Type your message and press Enter (or 'q' to quit).\n")

    while True:
        try:
            prompt = input("You > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("q", "quit", "exit"):
                print("Exiting assistant. Goodbye!")
                break

            print("[Processing] Streaming reply and synthesizing speech concurrently...")
            def _on_text_chunk(chunk):
                print(f"  [Speaking chunk {chunk.sentence_index + 1}]: {chunk.text}")

            res = await pipeline.process_text_prompt(
                prompt=prompt,
                speaker=speaker,
                play_audio=True,
                on_chunk=_on_text_chunk,
            )

            print(f"\nNexusVoice > {res.assistant_text}")
            print(
                f"[Metrics] LLM: {res.metrics.llm_latency_ms}ms | "
                f"TTS: {res.metrics.tts_latency_ms}ms | "
                f"TTFA: {res.metrics.ttfa_ms}ms | "
                f"Total: {res.metrics.total_latency_ms}ms\n"
            )

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


async def run_file_transcription(
    pipeline: AssistantPipeline,
    file_path: str,
    speaker: Optional[str] = None,
) -> None:
    """Processes an audio file through the complete assistant flow.

    Args:
        pipeline: Initialized AssistantPipeline instance.
        file_path: Path to the input audio file (WAV/MP3).
        speaker: Voice persona name.
    """
    print(f"Processing audio file: {file_path}")
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        res = await pipeline.process_audio_bytes(
            audio_bytes=audio_bytes,
            speaker=speaker,
            play_audio=True,
        )

        print(f"\n>>> Transcribed: {res.user_text}")
        print(f">>> Assistant:   {res.assistant_text}")
        print(
            f"[Metrics] STT: {res.metrics.stt_latency_ms}ms | "
            f"LLM: {res.metrics.llm_latency_ms}ms | "
            f"TTS: {res.metrics.tts_latency_ms}ms | "
            f"Total: {res.metrics.total_latency_ms}ms"
        )
    except Exception as exc:
        print(f"Failed to process audio file: {exc}")


def main() -> None:
    """CLI argument parser and dispatcher entrypoint."""
    parser = argparse.ArgumentParser(
        description="Personal Voice Assistant CLI powered by OpenVINO, LM Studio, and Qwen-TTS"
    )
    parser.add_argument(
        "--mode",
        choices=["voice", "text", "file", "info", "diag", "bench"],
        default="voice",
        help="Interaction mode: voice (mic), text (typed), file (audio file), info, diag (NPU-verified devices), or bench (Kokoro CPU/GPU/NPU)",
    )
    parser.add_argument(
        "--bench-text",
        type=str,
        default="Hello there, how are you today? This is a fixed benchmark sentence for measuring Kokoro speech synthesis speed.",
        help="Fixed text for --mode bench",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=3,
        help="Measured iterations per device for --mode bench",
    )
    parser.add_argument(
        "--warm-npu",
        action="store_true",
        help="Eagerly compile the NPU encoder during warmup (slow startup, fast first turn)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to audio file for --mode file",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Microphone recording duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=settings.tts_speaker,
        help="Voice persona for Qwen TTS (e.g. ryan, aiden, vivian, serena)",
    )
    args = parser.parse_args()

    print_banner()

    pipeline = AssistantPipeline()

    if args.mode == "info":
        print("Hardware & Subsystem Information:")
        devices = pipeline.stt.get_openvino_devices()
        print(f"  • OpenVINO Accelerators: {', '.join(devices)}")
        speakers = pipeline.tts.get_supported_speakers()
        print(f"  • Qwen-TTS Voices:       {', '.join(speakers)}")
        audio_devs = AudioProcessor.get_audio_devices()
        print(f"  • Default Input Device:  {audio_devs.get('default_input')}")
        print(f"  • Default Output Device: {audio_devs.get('default_output')}")
        return

    if args.mode == "diag":
        from app.core.diagnostics import (
            format_diagnostic_text,
            get_device_details,
            get_kokoro_device_report,
        )

        print("NPU-verified device diagnostics (runtime-confirmed, never assumed):")
        print()
        # Ensure Kokoro backend/effective state is known without full warmup.
        try:
            pipeline.tts.load_model()
        except Exception as exc:
            print(f"  [TTS load note] {exc}")
        report = get_kokoro_device_report(pipeline.tts)
        print(format_diagnostic_text(report))
        print()
        details = get_device_details()
        for dev, info in details.items():
            print(f"  {dev}: {info.get('full_device_name', '')} {info.get('device_architecture', '')}")
        print()
        print(f"  STT device setting: {pipeline.stt.device} ({pipeline.stt.model_id})")
        print(f"  TTS effective device: {report.get('effective_device')}")
        return

    if args.mode == "bench":
        from app.core.benchmark import benchmark_all_devices, format_results_table

        print("Kokoro benchmark: same fixed text on CPU/GPU/NPU (steady-state).")
        print(f"  Text: {args.bench_text[:80]}...")
        results = benchmark_all_devices(
            text=args.bench_text,
            measure_iters=max(1, args.bench_iters),
        )
        print()
        print(format_results_table(results))
        return

    # Warmup pipeline components
    print("Warming up models...")
    pipeline.warmup(warm_npu=args.warm_npu)
    print("Warmup complete.\n")

    if args.mode == "voice":
        asyncio.run(
            run_voice_loop(pipeline, duration=args.duration, speaker=args.speaker)
        )
    elif args.mode == "text":
        asyncio.run(run_text_loop(pipeline, speaker=args.speaker))
    elif args.mode == "file":
        if not args.file:
            print("Error: --file <path> is required when using --mode file.")
            sys.exit(1)
        asyncio.run(
            run_file_transcription(pipeline, args.file, speaker=args.speaker)
        )


if __name__ == "__main__":
    main()
