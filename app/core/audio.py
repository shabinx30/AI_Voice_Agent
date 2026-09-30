"""Audio processing and playback utilities.

This module provides routines for recording audio from microphones, resampling
waveforms to Whisper-compatible 16 kHz mono float32 format, serializing audio to
WAV byte streams, and playing synthesized audio via sounddevice.
"""

import io
import logging
import queue
import threading
from typing import Optional, Tuple
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class StreamAudioPlayer:
    """Threaded audio queue player for sequential, gapless playback.

    Allows streaming synthesized sentences to the audio device immediately as
    they arrive. The queue is bounded (backpressure): when full, producers
    briefly block instead of dropping speech, which keeps sentences in order
    while slowing TTS submission -- never silently losing audio.

    Why threaded + bounded (latency rationale):
        ``sd.play``/``sd.wait`` are blocking. Running them on a dedicated
        thread lets Kokoro synthesize sentence N+1 and Qwen generate sentence
        N+2 while sentence N is audible, removing inter-sentence gaps.
    """

    def __init__(
        self,
        max_queue: int = 8,
        on_first_play: Optional[object] = None,
    ) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, max_queue))
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running: bool = False
        self._lock = threading.Lock()
        self._on_first_play = on_first_play
        self._first_play_fired = False
        self._played_chunks = 0

    def start(self) -> None:
        """Starts the background playback worker thread if not already running."""
        if self._is_running and self._thread is not None and self._thread.is_alive():
            return
        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="StreamAudioPlayer"
        )
        self._thread.start()

    def _playback_worker(self) -> None:
        """Worker loop executing sequential chunk playback."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            audio_data, sample_rate = item
            try:
                if not self._stop_event.is_set():
                    if not self._first_play_fired:
                        self._first_play_fired = True
                        cb = self._on_first_play
                        if cb is not None:
                            try:
                                if callable(cb):
                                    cb()
                            except Exception:
                                pass
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    self._played_chunks += 1
            except Exception as exc:
                logger.warning("Stream audio playback warning: %s", exc)
            finally:
                self._queue.task_done()

        self._is_running = False

    @property
    def depth(self) -> int:
        """Current number of queued (not yet played) audio chunks."""
        try:
            return self._queue.qsize()
        except Exception:
            return 0

    @property
    def played_chunks(self) -> int:
        """Number of chunks fully played so far."""
        return self._played_chunks

    def play_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        block: bool = True,
        timeout: float = 5.0,
    ) -> bool:
        """Enqueues an audio waveform array for sequential playback.

        Args:
            audio_data: 1D float32 audio waveform samples.
            sample_rate: Sampling frequency in Hz.
            block: When True, briefly blocks (backpressure) if the queue is
                full instead of dropping speech. When False, drops the chunk
                if the player was stopped.
            timeout: Max seconds to wait for queue space.

        Returns:
            True when queued, False when dropped (stopped/cancelled).
        """
        if self._stop_event.is_set():
            return False
        if not self._is_running:
            self.start()
        # Bounded put with backpressure: never silently drop sentences.
        # TTS workers slow down here while LLM keeps generating -- exactly the
        # "prevent TTS overproduction" behavior required for low latency
        # without unbounded audio accumulation.
        try:
            if block:
                self._queue.put((audio_data, sample_rate), timeout=timeout)
            else:
                self._queue.put_nowait((audio_data, sample_rate))
            return True
        except queue.Full:
            logger.warning("StreamAudioPlayer queue full after %.1fs; chunk held.", timeout)
            return False

    def finish(self, wait: bool = False) -> None:
        """Signals completion of chunk streaming.

        Args:
            wait: If True, blocks until all queued audio chunks finish playing.
        """
        if self._is_running:
            self._queue.put(None)
            if wait and self._thread is not None and self._thread.is_alive():
                self._thread.join()

    def stop(self) -> None:
        """Immediately aborts audio playback and drains pending chunks."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Exception:
                break
        self._is_running = False

    @property
    def is_playing(self) -> bool:
        """Returns True if the background playback worker is active."""
        return self._is_running


class AudioProcessor:
    """Utility class for audio manipulation, conversion, and playback."""

    TARGET_SAMPLE_RATE: int = 16000

    @classmethod
    def load_from_bytes(
        cls, audio_bytes: bytes, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """Loads and resamples audio data from in-memory bytes into a float32 array.

        Args:
            audio_bytes: Raw binary bytes of an audio file (e.g. WAV, MP3, OGG).
            target_sr: Desired target sample rate (default 16,000 Hz).

        Returns:
            A tuple of (audio_array, sample_rate), where audio_array is 1D float32.

        Raises:
            ValueError: If the audio bytes cannot be read or are empty.
        """
        if not audio_bytes:
            raise ValueError("Audio byte buffer is empty.")

        audio_data = None
        sr = target_sr

        # Primary decoder: soundfile (fast for standard WAV, FLAC, OGG)
        try:
            with io.BytesIO(audio_bytes) as buffer:
                audio_data, sr = sf.read(buffer)
        except Exception as sf_exc:
            logger.debug(
                "Soundfile decode failed (%s). Attempting PyAV container decode...",
                sf_exc,
            )

        # Secondary decoder: PyAV (decodes WebM, Opus, AAC, MP4, and all ffmpeg containers)
        if audio_data is None:
            try:
                import av

                container = av.open(io.BytesIO(audio_bytes))
                resampler = av.AudioResampler(
                    format="fltp", layout="mono", rate=target_sr
                )
                frames = []
                for frame in container.decode(audio=0):
                    resampled = resampler.resample(frame)
                    for r in resampled:
                        frames.append(r.to_ndarray())

                if not frames:
                    raise ValueError("No decoded audio frames extracted.")

                audio_data = np.concatenate(frames, axis=1)[0]
                sr = target_sr
            except Exception as av_exc:
                logger.error("All audio decoders failed: %s", av_exc)
                raise ValueError(
                    f"Unable to parse audio stream: {av_exc}"
                ) from av_exc


        # Convert stereo or multi-channel to mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Ensure float32 representation normalized to [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            audio_data = audio_data / max_abs

        # Resample to target sample rate if needed
        if sr != target_sr:
            logger.debug(
                "Resampling audio from %d Hz to %d Hz", sr, target_sr
            )
            audio_data = librosa.resample(
                audio_data, orig_sr=sr, target_sr=target_sr
            )
            sr = target_sr

        return audio_data, sr

    @classmethod
    def to_wav_bytes(cls, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Encodes a numpy audio array into WAV format bytes.

        Args:
            audio_data: 1D or 2D numpy array containing audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Bytes representing a valid WAV file.
        """
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        return buffer.getvalue()

    @classmethod
    def play_audio(
        cls,
        audio_data: np.ndarray,
        sample_rate: int,
        blocking: bool = False,
    ) -> None:
        """Plays audio samples through the default output audio device.

        Args:
            audio_data: Numpy array with audio waveform samples.
            sample_rate: Audio sampling frequency in Hz.
            blocking: If True, waits until playback finishes before returning.

        Raises:
            RuntimeError: If playback through sounddevice fails.
        """
        try:
            logger.info(
                "Playing audio: length=%.2fs, rate=%dHz, blocking=%s",
                len(audio_data) / sample_rate,
                sample_rate,
                blocking,
            )
            sd.play(audio_data, sample_rate)
            if blocking:
                sd.wait()
        except Exception as exc:
            logger.error("Audio playback error: %s", exc)
            raise RuntimeError(f"Failed to play audio: {exc}") from exc

    @classmethod
    def create_stream_player(
        cls,
        max_queue: int = 8,
        on_first_play: Optional[object] = None,
    ) -> StreamAudioPlayer:
        """Creates and returns a new StreamAudioPlayer instance."""
        return StreamAudioPlayer(max_queue=max_queue, on_first_play=on_first_play)

    @classmethod
    def record_microphone(
        cls,
        duration_seconds: float = 5.0,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Records a fixed duration of audio from the default microphone.

        Args:
            duration_seconds: Duration of recording in seconds.
            sample_rate: Sample rate in Hz.

        Returns:
            1D float32 numpy array containing the recorded audio.

        Raises:
            RuntimeError: If recording fails.
        """
        try:
            logger.info(
                "Recording microphone for %.1f seconds at %d Hz...",
                duration_seconds,
                sample_rate,
            )
            recording = sd.rec(
                int(duration_seconds * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return recording.flatten()
        except Exception as exc:
            logger.error("Microphone recording error: %s", exc)
            raise RuntimeError(f"Microphone recording failed: {exc}") from exc

    @classmethod
    def get_audio_devices(cls) -> dict:
        """Queries and returns available host audio input and output devices.

        Returns:
            Dictionary containing input and output device summaries.
        """
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            return {
                "default_input": default_input,
                "default_output": default_output,
                "devices": devices,
            }
        except Exception as exc:
            logger.warning("Could not query sound devices: %s", exc)
            return {"error": str(exc), "devices": []}
