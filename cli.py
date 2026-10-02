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


async def run_direct_tts(
    pipeline: AssistantPipeline,
    text: str,
    speaker: Optional[str] = None,
    output_file: Optional[str] = None,
    play_audio: bool = True,
    tts_device: Optional[str] = None,
) -> None:
    """Synthesizes text directly into speech with Kokoro-82M.

    Args:
        pipeline: Initialized AssistantPipeline instance.
        text: Text to synthesize into speech.
        speaker: Voice persona name.
        output_file: Optional file path to save synthesized audio as WAV.
        play_audio: Whether to play synthesized audio on host speakers.
        tts_device: Compute device override ('cpu' or 'npu').
    """
    target_speaker = speaker or pipeline.tts.speaker
    print(f"\n[TTS] Synthesizing speech with Kokoro-82M (speaker: {target_speaker})...")
    res = await pipeline.process_direct_tts(
        text=text,
        speaker=speaker,
        play_audio=play_audio,
        tts_device=tts_device,
    )

    if output_file:
        try:
            with open(output_file, "wb") as f:
                f.write(res.audio_bytes)
            print(f"[Saved] Audio successfully saved to: {output_file}")
        except Exception as io_err:
            print(f"[Warning] Failed to write audio file '{output_file}': {io_err}")

    print(f">>> Synthesized: {res.assistant_text}")
    print(
        f"[Metrics] TTS Latency: {res.metrics.tts_latency_ms}ms | "
        f"RTF: {res.metrics.tts_realtime_factor}x | "
        f"Sample Rate: {res.sample_rate}Hz\n"
    )


async def run_tts_loop(
    pipeline: AssistantPipeline,
    speaker: Optional[str] = None,
    output_file: Optional[str] = None,
    play_audio: bool = True,
    tts_device: Optional[str] = None,
) -> None:
    """Runs an interactive text-to-speech loop for direct speech synthesis.

    Args:
        pipeline: Initialized AssistantPipeline instance.
        speaker: Voice persona name.
        output_file: Optional default output file path template.
        play_audio: Whether to play synthesized audio through speakers.
        tts_device: Compute device override ('cpu' or 'npu').
    """
    print("Standalone Text-to-Speech (TTS) Mode Active.")
    print("Type text and press Enter to hear it synthesized directly by Kokoro-82M.")
    print("Commands:")
    print("  :voice <name>     Change voice persona (e.g. :voice af_bella)")
    print("  :save <file.wav>  Save subsequent utterances to WAV file")
    print("  :device <cpu/npu> Switch compute device")
    print("  'q' or 'quit'     Exit\n")

    current_speaker = speaker
    current_save = output_file
    idx = 1

    while True:
        try:
            prompt = input("TTS > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("q", "quit", "exit"):
                print("Exiting TTS mode. Goodbye!")
                break

            if prompt.startswith(":voice "):
                new_spk = prompt.split(" ", 1)[1].strip()
                if new_spk:
                    current_speaker = new_spk
                    print(f"[Voice changed to: {current_speaker}]\n")
                continue

            if prompt.startswith(":save "):
                current_save = prompt.split(" ", 1)[1].strip()
                print(f"[Subsequent audio will be saved to: {current_save}]\n")
                continue

            if prompt.startswith(":device "):
                new_dev = prompt.split(" ", 1)[1].strip()
                try:
                    pipeline.set_tts_device(new_dev)
                    print(f"[TTS device set to: {new_dev}]\n")
                except Exception as d_err:
                    print(f"[Device error: {d_err}]\n")
                continue

            save_path = current_save
            if current_save and "{n}" in current_save:
                save_path = current_save.replace("{n}", str(idx))

            await run_direct_tts(
                pipeline=pipeline,
                text=prompt,
                speaker=current_speaker,
                output_file=save_path,
                play_audio=play_audio,
                tts_device=tts_device,
            )
            idx += 1

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


def main() -> None:
    """CLI argument parser and dispatcher entrypoint."""
    parser = argparse.ArgumentParser(
        description="Personal Voice Assistant CLI powered by OpenVINO, LM Studio, and Kokoro-82M TTS"
    )
    parser.add_argument(
        "--mode",
        choices=["voice", "text", "tts", "file", "info", "diag", "bench"],
        default="voice",
        help="Interaction mode: voice (mic), text (LLM chat), tts (direct speech synthesis), file (audio file), info, diag, or bench",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Explicit text to synthesize in TTS mode (or prompt in text mode)",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output_file",
        type=str,
        default=None,
        help="Path to output WAV file to save synthesized speech (e.g. speech.wav)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Disable audio playback through speakers",
    )
    parser.add_argument(
        "--tts-device",
        type=str,
        default=None,
        choices=["cpu", "npu", "gpu", "auto"],
        help="Compute device override for Kokoro TTS (cpu, npu)",
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
        help="Voice persona for Kokoro TTS (e.g. af_heart, am_adam, af_bella, am_michael)",
    )
    args = parser.parse_args()

    # If --text is explicitly provided without specifying --mode, default to TTS synthesis
    if args.text and args.mode == "voice":
        args.mode = "tts"

    print_banner()

    pipeline = AssistantPipeline()

    if args.tts_device:
        pipeline.set_tts_device(args.tts_device)

    if args.mode == "info":
        print("Hardware & Subsystem Information:")
        devices = pipeline.stt.get_openvino_devices()
        print(f"  • OpenVINO Accelerators: {', '.join(devices)}")
        speakers = pipeline.tts.get_supported_speakers()
        print(f"  • Kokoro-TTS Voices:     {', '.join(speakers)}")
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

    # In standalone TTS mode, load only the TTS model for rapid startup without LLM requirement
    if args.mode == "tts":
        print("Loading Kokoro-82M TTS engine...")
        pipeline.tts.load_model()
        print("TTS Engine ready.\n")

        if args.text:
            asyncio.run(
                run_direct_tts(
                    pipeline,
                    text=args.text,
                    speaker=args.speaker,
                    output_file=args.output_file,
                    play_audio=not args.no_play,
                    tts_device=args.tts_device,
                )
            )
        else:
            asyncio.run(
                run_tts_loop(
                    pipeline,
                    speaker=args.speaker,
                    output_file=args.output_file,
                    play_audio=not args.no_play,
                    tts_device=args.tts_device,
                )
            )
        return

    # Full assistant warmup for voice / text / file modes
    print("Warming up models...")
    pipeline.warmup(warm_npu=args.warm_npu)
    print("Warmup complete.\n")

    if args.mode == "voice":
        asyncio.run(
            run_voice_loop(pipeline, duration=args.duration, speaker=args.speaker)
        )
    elif args.mode == "text":
        if args.text:
            # Single-turn prompt
            asyncio.run(
                pipeline.process_text_prompt(
                    prompt=args.text,
                    speaker=args.speaker,
                    play_audio=not args.no_play,
                )
            )
        else:
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
