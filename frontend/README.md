# NexusVoice • Next.js & Tailwind CSS Frontend

Next-generation AI Personal Assistant interface built with **Next.js 16 (App Router)**, **React 19**, and **Tailwind CSS v4**.

## Architecture & Features

- **Real-Time WebSocket Streaming**: Bi-directional communication with the FastAPI backend (`/ws/assistant`) supporting real-time speech-to-text transcription, word-by-word LLM token streaming, sentence-by-sentence TTS audio chunking, and latency metrics.
- **Microphone Capture & Visualization**:
  - Live sinusoidal canvas visualizer during idle state.
  - Real-time time-domain waveform rendering via Web Audio API `AnalyserNode` during voice recording.
  - Recording duration timer with pulsing recording state.
- **Audio Playback Engine**:
  - `StreamAudioQueue`: Web Audio API timeline scheduler for gapless, sequential sentence-by-sentence playback.
  - Built-in HTML5 Audio fallback queue.
  - In-bubble replay functionality for completed assistant voice responses.
- **Subsystem & Hardware Telemetry**:
  - Live status indicators and model information for **OpenVINO Whisper Base INT8 STT** (Intel Arc GPU / NPU).
  - **LM Studio LLM** connectivity and active model monitoring.
  - **Kokoro-82M TTS** status and sample rate display (24,000 Hz).
- **Voice Configuration**:
  - Select from 18+ Kokoro-82M voice personas (American & British male/female voices).
  - Hardware host speaker toggle (`Host Speaker Output`).
- **Telemetry & Latency Pills**:
  - STT latency (`stt_ms`)
  - LLM latency (`llm_ms`)
  - TTS latency (`tts_ms`)
  - TTFA (Time To First Audio) highlighted metric (`ttfa_ms`)
  - Total end-to-end round trip latency (`total_ms`)
- **HTTP Fallback System**:
  - Seamless fallback to HTTP endpoints (`/api/interact`, `/api/chat/tokens`, `/api/tts`, `/api/health`) if WebSocket disconnects.

## Getting Started

### 1. Ensure the Backend Server is Running
In the root directory, start the FastAPI assistant server:
```bash
python run_server.py
```
*(Runs on `http://127.0.0.1:8000`)*

### 2. Start the Next.js Development Server
In the `frontend` directory:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Build for Production

```bash
npm run build
npm run start
```
