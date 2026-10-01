/**
 * NexusVoice Frontend Application
 *
 * Coordinates client-side microphone capture, Web Audio visualizer,
 * real-time bidirectional WebSocket streaming with the backend assistant server,
 * sentence-by-sentence streaming audio playback, and latency metrics tracking.
 */

// DOM Elements
const micBtn = document.getElementById('mic-btn');
const micIcon = document.getElementById('mic-icon');
const stopIcon = document.getElementById('stop-icon');
const textInput = document.getElementById('text-prompt-input');
const sendBtn = document.getElementById('send-btn');
const chatContainer = document.getElementById('chat-messages');
const waveformCanvas = document.getElementById('waveform-canvas');
const recordingTimer = document.getElementById('recording-timer');
const speakerSelect = document.getElementById('speaker-select');
const playHostAudio = document.getElementById('play-host-audio');
const refreshStatusBtn = document.getElementById('refresh-status-btn');
const audioPlayer = document.getElementById('audio-player');

const modelSelect = document.getElementById('model-select');
const refreshModelsBtn = document.getElementById('refresh-models-btn');
const ejectModelBtn = document.getElementById('eject-model-btn');
const modelStatusBadge = document.getElementById('model-status-badge');
const modelParamText = document.getElementById('model-param-text');
const modelSizeText = document.getElementById('model-size-text');

let availableModelsList = [];
let activeModelId = '';

const sttDot = document.getElementById('stt-dot');
const sttModelName = document.getElementById('stt-model-name');
const sttDeviceName = document.getElementById('stt-device-name');
const llmDot = document.getElementById('llm-dot');
const llmModelName = document.getElementById('llm-model-name');
const ttsDot = document.getElementById('tts-dot');
const ttsModelName = document.getElementById('tts-model-name');
const systemStatusText = document.getElementById('system-status-text');

const metricSTT = document.getElementById('metric-stt');
const metricLLM = document.getElementById('metric-llm');
const metricTTS = document.getElementById('metric-tts');
const metricTTFA = document.getElementById('metric-ttfa');
const metricTotal = document.getElementById('metric-total');

// Canvas Context & Audio API State
const canvasCtx = waveformCanvas.getContext('2d');
let micAudioCtx = null;
let analyser = null;
let micStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;
let animFrameId = null;

// ============================================================================
// StreamAudioQueue: Web Audio API timeline scheduler for sequential playback
// ============================================================================
class StreamAudioQueue {
  constructor() {
    this.audioCtx = null;
    this.nextStartTime = 0;
    this.activeSources = [];
    this.htmlFallbackQueue = [];
    this.isHtmlPlaying = false;
  }

  getAudioContext() {
    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContextClass();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume().catch(() => {});
    }
    return this.audioCtx;
  }

  reset() {
    this.nextStartTime = 0;
    this.activeSources.forEach((src) => {
      try {
        src.stop();
        src.disconnect();
      } catch (e) {}
    });
    this.activeSources = [];
    this.htmlFallbackQueue = [];
    this.isHtmlPlaying = false;
  }

  async enqueue(audioBase64, onStart = null, onEnded = null) {
    if (!audioBase64) return;

    try {
      const ctx = this.getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      // Decode base64 WAV into ArrayBuffer
      const binaryString = window.atob(audioBase64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Decode PCM/WAV buffer
      const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));

      const currentTime = ctx.currentTime;
      // Start immediately (+20ms buffer) if idle, or seamless transition after previous chunk
      const startTime = Math.max(currentTime + 0.02, this.nextStartTime);
      this.nextStartTime = startTime + audioBuffer.duration;

      const sourceNode = ctx.createBufferSource();
      sourceNode.buffer = audioBuffer;
      sourceNode.connect(ctx.destination);
      this.activeSources.push(sourceNode);

      sourceNode.onended = () => {
        const idx = this.activeSources.indexOf(sourceNode);
        if (idx !== -1) this.activeSources.splice(idx, 1);
        if (onEnded) onEnded();
      };

      sourceNode.start(startTime);

      if (onStart) {
        const delayMs = Math.max(0, (startTime - currentTime) * 1000);
        setTimeout(onStart, delayMs);
      }
    } catch (err) {
      console.warn('Web Audio decode failed, falling back to HTML5 audio queue:', err);
      this.enqueueHtmlAudioFallback(audioBase64, onStart, onEnded);
    }
  }

  enqueueHtmlAudioFallback(audioBase64, onStart, onEnded) {
    const audioUrl = `data:audio/wav;base64,${audioBase64}`;
    this.htmlFallbackQueue.push({ audioUrl, onStart, onEnded });
    if (!this.isHtmlPlaying) {
      this.playNextHtmlAudio();
    }
  }

  playNextHtmlAudio() {
    if (this.htmlFallbackQueue.length === 0) {
      this.isHtmlPlaying = false;
      return;
    }
    this.isHtmlPlaying = true;
    const item = this.htmlFallbackQueue.shift();
    const audio = new Audio(item.audioUrl);
    if (item.onStart) item.onStart();
    audio.onended = () => {
      if (item.onEnded) item.onEnded();
      this.playNextHtmlAudio();
    };
    audio.onerror = () => {
      this.playNextHtmlAudio();
    };
    audio.play().catch(() => {
      this.playNextHtmlAudio();
    });
  }
}

const streamAudioQueue = new StreamAudioQueue();

// ============================================================================
// WebSocket Client for Real-time Streaming
// ============================================================================
let ws = null;
let wsReconnectTimer = null;
let activeUserBubble = null;
let activeAssistantBubble = null;
let currentSentenceCount = 0;

// ============================================================================
// Word-by-word token renderer (rAF-batched)
// ============================================================================
// Raw LLM tokens arrive as sub-word deltas. They accumulate in tokenBuffer;
// complete words flush as animated spans while the trailing partial word
// renders with a blinking caret — the ChatGPT-style streaming effect.
let tokenBuffer = '';
let tokenFlushScheduled = false;
let liveTextEl = null;

function resetTokenStream() {
  tokenBuffer = '';
  tokenFlushScheduled = false;
  liveTextEl = null;
}

function pushToken(text) {
  if (!liveTextEl || !text) return;
  if (liveTextEl.querySelector('.status-placeholder')) {
    liveTextEl.innerHTML = '';
  }
  tokenBuffer += text;
  if (!tokenFlushScheduled) {
    tokenFlushScheduled = true;
    requestAnimationFrame(() => {
      tokenFlushScheduled = false;
      flushTokenBuffer(false);
    });
  }
}

function appendWordSpan(frag, word) {
  const span = document.createElement('span');
  span.className = 'streamed-word';
  span.textContent = word;
  frag.appendChild(span);
}

function flushTokenBuffer(isFinal) {
  if (!liveTextEl) {
    tokenBuffer = '';
    return;
  }
  const oldCaret = liveTextEl.querySelector('.stream-caret');
  if (oldCaret) oldCaret.remove();
  if (!tokenBuffer) return;

  const frag = document.createDocumentFragment();
  // Split keeping whitespace: [word, spaces, word, ..., trailing]
  const parts = tokenBuffer.split(/(\s+)/);
  // Everything except the trailing segment is complete (word or spacing).
  // The trailing segment is a partial word unless this is the final flush.
  const complete = isFinal ? parts : parts.slice(0, -1);
  complete.forEach((part) => {
    if (!part) return;
    if (/^\s+$/.test(part)) {
      frag.appendChild(document.createTextNode(part));
    } else {
      appendWordSpan(frag, part);
    }
  });

  if (isFinal) {
    tokenBuffer = '';
  } else {
    const pending = parts[parts.length - 1];
    tokenBuffer = pending || '';
    if (pending) {
      const caret = document.createElement('span');
      caret.className = 'stream-caret';
      caret.textContent = pending;
      frag.appendChild(caret);
    }
  }

  liveTextEl.appendChild(frag);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function getWebSocketUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/assistant`;
}

function initWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    ws = new WebSocket(getWebSocketUrl());

    ws.onopen = () => {
      console.log('Assistant WebSocket connected.');
      systemStatusText.textContent = 'All Pipelines Active • Ready for Voice Input';
      if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
        wsReconnectTimer = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleWebSocketMessage(msg);
      } catch (err) {
        console.error('WebSocket frame parse error:', err);
      }
    };

    ws.onclose = () => {
      console.log('Assistant WebSocket disconnected. Reconnecting in 3s...');
      if (!wsReconnectTimer) {
        wsReconnectTimer = setTimeout(initWebSocket, 3000);
      }
    };

    ws.onerror = (err) => {
      console.warn('Assistant WebSocket error:', err);
    };
  } catch (err) {
    console.error('Could not initialize WebSocket:', err);
  }
}

// Heartbeat ping every 25 seconds
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 25000);

function updateMetrics(metrics) {
  if (!metrics) return;
  if (metrics.stt_ms !== undefined) metricSTT.textContent = `${metrics.stt_ms}ms`;
  if (metrics.llm_ms !== undefined) metricLLM.textContent = `${metrics.llm_ms}ms`;
  if (metrics.tts_ms !== undefined) metricTTS.textContent = `${metrics.tts_ms}ms`;
  if (metricTTFA && metrics.ttfa_ms !== undefined) metricTTFA.textContent = `${metrics.ttfa_ms}ms`;
  if (metrics.total_ms !== undefined) metricTotal.textContent = `${metrics.total_ms}ms`;
}

function handleWebSocketMessage(msg) {
  if (msg.type === 'pong') {
    return;
  }

  if (msg.type === 'status') {
    systemStatusText.textContent = msg.message;
    return;
  }

  if (msg.type === 'models') {
    if (msg.models && msg.models.length > 0) {
      availableModelsList = msg.models;
    }
    if (msg.current_model) {
      activeModelId = msg.current_model;
    }
    renderModelSelectOptions();
    return;
  }

  if (msg.type === 'model_changed') {
    if (msg.model) {
      activeModelId = msg.model;
      if (modelSelect) modelSelect.value = msg.model;
      if (llmModelName) llmModelName.textContent = msg.model;
      updateModelDetailsDisplay();
    }
    if (msg.message) {
      systemStatusText.textContent = msg.message;
    }
    return;
  }

  if (msg.type === 'transcription') {
    // STT completed: update user speech bubble immediately
    if (activeUserBubble) {
      activeUserBubble.querySelector('.msg-text').textContent = msg.user_text;
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    systemStatusText.textContent = 'Speech transcribed • Streaming reply from LM Studio...';
    return;
  }

  if (msg.type === 'token') {
    // Raw LLM token delta: reveal word-by-word in the live bubble
    if (!activeAssistantBubble) {
      activeAssistantBubble = appendMessage('assistant', '<span class="status-placeholder">Thinking...</span>');
    }
    const textEl = activeAssistantBubble.querySelector('.msg-text');
    if (liveTextEl !== textEl) {
      resetTokenStream();
      liveTextEl = textEl;
    }
    pushToken(msg.text || '');
    return;
  }

  if (msg.type === 'chunk') {
    // Sentence audio completed and streamed from server. Text is already
    // revealed word-by-word via token frames; chunks only drive audio.
    if (!activeAssistantBubble) {
      activeAssistantBubble = appendMessage('assistant', '');
    }
    currentSentenceCount++;
    chatContainer.scrollTop = chatContainer.scrollHeight;

    systemStatusText.textContent = `Speaking sentence ${msg.index + 1}...`;

    // Immediately play this sentence's audio and queue next
    if (!playHostAudio.checked && msg.audio_base64) {
      streamAudioQueue.enqueue(msg.audio_base64);
    }
    return;
  }

  if (msg.type === 'result') {
    // Full generation complete: flush remaining tokens, then reconcile
    // with the authoritative reply text
    flushTokenBuffer(true);
    if (activeAssistantBubble) {
      const textEl = activeAssistantBubble.querySelector('.msg-text');
      if (msg.assistant_text) {
        textEl.textContent = msg.assistant_text;
      }

      // Attach complete audio play button
      if (msg.audio_base64) {
        const fullAudioUrl = `data:audio/wav;base64,${msg.audio_base64}`;
        attachAudioButton(activeAssistantBubble, fullAudioUrl);
      }
    }

    if (activeUserBubble && msg.user_text) {
      activeUserBubble.querySelector('.msg-text').textContent = msg.user_text;
    }

    if (msg.metrics) {
      updateMetrics(msg.metrics);
    }

    systemStatusText.textContent = 'All Pipelines Active • Ready for Voice Input';
    resetTokenStream();
    activeUserBubble = null;
    activeAssistantBubble = null;
    currentSentenceCount = 0;
    return;
  }

  if (msg.type === 'error') {
    systemStatusText.textContent = `Error: ${msg.message}`;
    if (activeAssistantBubble) {
      activeAssistantBubble.querySelector('.msg-text').textContent = `Error: ${msg.message}`;
    }
    resetTokenStream();
    activeUserBubble = null;
    activeAssistantBubble = null;
    currentSentenceCount = 0;
  }
}

// ============================================================================
// System Status & Health Check
// ============================================================================
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('Health check error');
    const data = await res.json();

    // STT
    sttDot.className = 'status-indicator online';
    sttModelName.textContent = data.stt_model;
    sttDeviceName.textContent = `${data.stt_device} (Intel Arc 130V)`;

    // LLM
    if (data.lm_studio_connected) {
      llmDot.className = 'status-indicator online';
      llmModelName.textContent = data.lm_studio_model;
      if (!activeModelId) {
        activeModelId = data.lm_studio_model;
      }
    } else {
      llmDot.className = 'status-indicator offline';
      llmModelName.textContent = 'Offline (Check LM Studio)';
    }

    // TTS
    ttsDot.className = 'status-indicator online';
    ttsModelName.textContent = data.tts_model;

    // Speakers (Kokoro-82M Voice Personas)
    if (data.tts_speakers && data.tts_speakers.length > 0) {
      const currentSelection = speakerSelect.value;
      speakerSelect.innerHTML = '';
      data.tts_speakers.forEach((spk) => {
        const opt = document.createElement('option');
        opt.value = spk;

        // Friendly label: e.g. af_heart -> Heart (US Female)
        const parts = spk.split('_');
        if (parts.length === 2) {
          const nat = parts[0].startsWith('a') ? 'US' : 'UK';
          const gen = parts[0].endsWith('f') ? 'Female' : 'Male';
          const name = parts[1].charAt(0).toUpperCase() + parts[1].slice(1);
          opt.textContent = `${name} (${nat} ${gen})`;
        } else {
          opt.textContent = spk.charAt(0).toUpperCase() + spk.slice(1);
        }

        if (spk === currentSelection || (!currentSelection && spk === 'af_heart')) {
          opt.selected = true;
        }
        speakerSelect.appendChild(opt);
      });
    }

    systemStatusText.textContent = 'All Pipelines Active • Ready for Voice Input';
  } catch (err) {
    console.error('Failed to query system health:', err);
    systemStatusText.textContent = 'System Disconnected • Ensure server is running';
    llmDot.className = 'status-indicator offline';
  }
}

// ============================================================================
// Visualizer Routine
// ============================================================================
function drawIdleVisualizer() {
  const width = waveformCanvas.width;
  const height = waveformCanvas.height;

  canvasCtx.clearRect(0, 0, width, height);
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
  canvasCtx.beginPath();

  const sliceWidth = width / 60;
  let x = 0;

  for (let i = 0; i < 60; i++) {
    const y = height / 2 + Math.sin(i * 0.2 + Date.now() * 0.003) * 3;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }
  canvasCtx.stroke();

  if (!isRecording) {
    animFrameId = requestAnimationFrame(drawIdleVisualizer);
  }
}

function drawLiveWaveform() {
  if (!isRecording || !analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);

  const width = waveformCanvas.width;
  const height = waveformCanvas.height;

  canvasCtx.clearRect(0, 0, width, height);
  canvasCtx.lineWidth = 2.5;

  const gradient = canvasCtx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, '#f43f5e');
  gradient.addColorStop(0.5, '#6366f1');
  gradient.addColorStop(1, '#06b6d4');
  canvasCtx.strokeStyle = gradient;

  canvasCtx.beginPath();
  const sliceWidth = width / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * height) / 2;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }

  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();

  animFrameId = requestAnimationFrame(drawLiveWaveform);
}

// ============================================================================
// Microphone Capture & Recording
// ============================================================================
async function startRecording() {
  try {
    cancelAnimationFrame(animFrameId);
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Unlock Web Audio Context
    streamAudioQueue.getAudioContext();

    micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = micAudioCtx.createAnalyser();
    const source = micAudioCtx.createMediaStreamSource(micStream);
    source.connect(analyser);
    analyser.fftSize = 1024;

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(micStream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = handleRecordingComplete;

    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add('recording');
    micIcon.classList.add('hidden');
    stopIcon.classList.remove('hidden');

    recordingTimer.classList.remove('hidden');
    recordStartTime = Date.now();
    timerInterval = setInterval(updateTimer, 500);

    systemStatusText.textContent = 'Listening to your microphone...';
    drawLiveWaveform();
  } catch (err) {
    console.error('Microphone access denied:', err);
    alert('Please grant microphone permission in your browser.');
    drawIdleVisualizer();
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  clearInterval(timerInterval);
  recordingTimer.classList.add('hidden');

  micBtn.classList.remove('recording');
  stopIcon.classList.add('hidden');
  micIcon.classList.remove('hidden');

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }

  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
  }

  drawIdleVisualizer();
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
  const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const secs = String(elapsed % 60).padStart(2, '0');
  recordingTimer.textContent = `${mins}:${secs}`;
}

function audioBufferToWavBlob(audioBuffer) {
  const numOfChan = audioBuffer.numberOfChannels;
  const length = audioBuffer.length * numOfChan * 2 + 44;
  const outBuffer = new ArrayBuffer(length);
  const view = new DataView(outBuffer);
  const channels = [];
  let sample = 0;
  let offset = 0;
  let pos = 0;

  function setUint16(data) {
    view.setUint16(pos, data, true);
    pos += 2;
  }
  function setUint32(data) {
    view.setUint32(pos, data, true);
    pos += 4;
  }

  setUint32(0x46464952); // "RIFF"
  setUint32(length - 8); // file length - 8
  setUint32(0x45564157); // "WAVE"

  setUint32(0x20746d66); // "fmt " chunk
  setUint32(16); // length = 16
  setUint16(1); // PCM
  setUint16(numOfChan);
  setUint32(audioBuffer.sampleRate);
  setUint32(audioBuffer.sampleRate * 2 * numOfChan);
  setUint16(numOfChan * 2);
  setUint16(16);

  setUint32(0x61746164); // "data" chunk
  setUint32(length - pos - 4);

  for (let i = 0; i < numOfChan; i++) {
    channels.push(audioBuffer.getChannelData(i));
  }

  while (offset < audioBuffer.length) {
    for (let i = 0; i < numOfChan; i++) {
      sample = Math.max(-1, Math.min(1, channels[i][offset]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      view.setInt16(pos, sample, true);
      pos += 2;
    }
    offset++;
  }

  return new Blob([outBuffer], { type: 'audio/wav' });
}

// ============================================================================
// Interaction Handlers (Voice & Text)
// ============================================================================
async function handleRecordingComplete() {
  const mimeType = (mediaRecorder && mediaRecorder.mimeType) ? mediaRecorder.mimeType : 'audio/webm';
  const rawBlob = new Blob(recordedChunks, { type: mimeType });
  systemStatusText.textContent = 'Transcribing speech with OpenVINO Whisper Base INT8...';

  activeUserBubble = appendMessage('user', '<span class="status-placeholder">Processing speech...</span>');
  activeAssistantBubble = appendMessage('assistant', '<span class="status-placeholder">Listening...</span>');
  resetTokenStream();
  currentSentenceCount = 0;

  // Unlock and clear audio queue
  streamAudioQueue.getAudioContext();
  streamAudioQueue.reset();

  let audioBlobToSend = rawBlob;
  try {
    const arrayBuf = await rawBlob.arrayBuffer();
    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decodedBuffer = await decodeCtx.decodeAudioData(arrayBuf);
    audioBlobToSend = audioBufferToWavBlob(decodedBuffer);
    decodeCtx.close();
  } catch (convErr) {
    console.debug('Direct WebM upload used:', convErr);
  }

  // Convert to Base64
  const arrayBuf = await audioBlobToSend.arrayBuffer();
  let binary = '';
  const bytes = new Uint8Array(arrayBuf);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const base64Audio = window.btoa(binary);

  // Send via streaming WebSocket if connected
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'audio',
      data: base64Audio,
      speaker: speakerSelect.value,
      play_audio: playHostAudio.checked,
      model: modelSelect ? modelSelect.value : activeModelId,
    }));
  } else {
    // HTTP Fallback
    handleHttpAudioFallback(audioBlobToSend, activeUserBubble, activeAssistantBubble);
  }
}

async function handleHttpAudioFallback(blob, userMsgEl, assistantMsgEl) {
  const formData = new FormData();
  formData.append('file', blob, 'input.wav');
  formData.append('speaker', speakerSelect.value);
  formData.append('play_audio', playHostAudio.checked);

  try {
    const res = await fetch('/api/interact', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Interaction failed');
    }

    const data = await res.json();
    userMsgEl.querySelector('.msg-text').textContent = data.user_text;
    assistantMsgEl.querySelector('.msg-text').textContent = data.assistant_text;

    if (data.audio_base64) {
      const audioUrl = `data:audio/wav;base64,${data.audio_base64}`;
      attachAudioButton(assistantMsgEl, audioUrl);
      if (!playHostAudio.checked) {
        audioPlayer.src = audioUrl;
        audioPlayer.play().catch(() => {});
      }
    }

    if (data.metrics) {
      updateMetrics(data.metrics);
    }

    systemStatusText.textContent = 'All Pipelines Active • Ready for Voice Input';
  } catch (err) {
    console.error('HTTP audio fallback error:', err);
    assistantMsgEl.querySelector('.msg-text').textContent = `Error: ${err.message}`;
    systemStatusText.textContent = 'Failed to process request.';
  }
}

// Text Input Handling
async function handleTextSubmit() {
  const text = textInput.value.trim();
  if (!text) return;

  textInput.value = '';
  appendMessage('user', text);
  systemStatusText.textContent = 'Streaming reply and synthesizing speech...';

  // Unlock and clear audio queue
  streamAudioQueue.getAudioContext();
  streamAudioQueue.reset();

  activeUserBubble = null;
  activeAssistantBubble = appendMessage('assistant', '<span class="status-placeholder">Thinking...</span>');
  resetTokenStream();
  currentSentenceCount = 0;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'text',
      prompt: text,
      speaker: speakerSelect.value,
      play_audio: playHostAudio.checked,
      model: modelSelect ? modelSelect.value : activeModelId,
    }));
  } else {
    // HTTP Fallback
    handleHttpTextFallback(text, activeAssistantBubble);
  }
}

/**
 * Reads the /api/chat/tokens SSE stream, feeding each token delta to onToken.
 *
 * Returns the full accumulated reply text.
 */
async function streamTokensHttp(promptText, onToken) {
  const res = await fetch('/api/chat/tokens', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: promptText }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Token stream failed (Status ${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let full = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const line of frame.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const data = trimmed.slice(5).trim();
        if (data === '[DONE]') continue;
        let evt;
        try {
          evt = JSON.parse(data);
        } catch (parseErr) {
          continue;
        }
        if (evt.error) throw new Error(evt.error);
        if (evt.token) {
          full += evt.token;
          onToken(evt.token);
        }
      }
    }
  }
  // Flush any decoder remainder
  buf += decoder.decode();
  return full;
}

async function handleHttpTextFallback(promptText, assistantMsgEl) {
  const textEl = assistantMsgEl.querySelector('.msg-text');
  resetTokenStream();
  liveTextEl = textEl;

  try {
    // Word-by-word token stream for display...
    const replyText = await streamTokensHttp(promptText, pushToken);
    // ...reconciled with the authoritative accumulated text.
    flushTokenBuffer(true);
    textEl.textContent = replyText;
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const ttsRes = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: replyText,
        speaker: speakerSelect.value,
      }),
    });

    if (ttsRes.ok) {
      const audioBlob = await ttsRes.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      attachAudioButton(assistantMsgEl, audioUrl);
      if (!playHostAudio.checked) {
        audioPlayer.src = audioUrl;
        audioPlayer.play().catch(() => {});
      }
    }

    systemStatusText.textContent = 'All Pipelines Active • Ready for Voice Input';
  } catch (err) {
    console.error('HTTP text fallback error:', err);
    assistantMsgEl.querySelector('.msg-text').textContent = `Error: ${err.message}`;
    systemStatusText.textContent = 'Error processing text.';
  } finally {
    resetTokenStream();
  }
}

// ============================================================================
// DOM Message Rendering
// ============================================================================
function appendMessage(role, text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-msg`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.innerHTML =
    role === 'assistant'
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a8 8 0 0 0-8 8v12l3-3h5a8 8 0 0 0 8-8V2z"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  const header = document.createElement('div');
  header.className = 'msg-header';
  header.textContent = role === 'assistant' ? 'NexusVoice' : 'You';

  const msgText = document.createElement('div');
  msgText.className = 'msg-text';
  msgText.innerHTML = text;

  bubble.appendChild(header);
  bubble.appendChild(msgText);
  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubble);

  chatContainer.appendChild(msgDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  return msgDiv;
}

function attachAudioButton(msgDiv, audioUrl) {
  const bubble = msgDiv.querySelector('.msg-bubble');
  if (bubble.querySelector('.audio-controls')) return;

  const controls = document.createElement('div');
  controls.className = 'audio-controls';

  const playBtn = document.createElement('button');
  playBtn.className = 'play-bubble-btn';
  playBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
    Replay Full Audio
  `;

  playBtn.addEventListener('click', () => {
    streamAudioQueue.reset();
    audioPlayer.src = audioUrl;
    audioPlayer.play();
  });

  controls.appendChild(playBtn);
  bubble.appendChild(controls);
}

// ============================================================================
// LM Studio Model Management
// ============================================================================
function renderModelSelectOptions() {
  if (!modelSelect) return;
  const currentSelection = modelSelect.value || activeModelId;
  modelSelect.innerHTML = '';

  if (availableModelsList.length === 0) {
    const opt = document.createElement('option');
    opt.value = activeModelId || 'qwen2.5-0.5b-instruct';
    opt.textContent = activeModelId || 'qwen2.5-0.5b-instruct';
    modelSelect.appendChild(opt);
  } else {
    availableModelsList.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.id;
      const loadedMarker = m.loaded ? ' • [Loaded]' : '';
      const paramMarker = m.params ? ` (${m.params})` : '';
      opt.textContent = `${m.name || m.id}${paramMarker}${loadedMarker}`;
      if (m.id === currentSelection) {
        opt.selected = true;
      }
      modelSelect.appendChild(opt);
    });
  }
  updateModelDetailsDisplay();
}

async function fetchAvailableModels() {
  if (!modelSelect) return;
  if (refreshModelsBtn) refreshModelsBtn.classList.add('spin-animation');
  if (modelStatusBadge) {
    modelStatusBadge.textContent = 'Scanning...';
    modelStatusBadge.className = 'badge-subtle loading';
  }

  try {
    const res = await fetch('/api/llm/models');
    if (!res.ok) throw new Error('Failed to retrieve LM Studio models');
    const data = await res.json();
    availableModelsList = data.models || [];
    if (data.current_model) {
      activeModelId = data.current_model;
    }
    renderModelSelectOptions();

    if (modelStatusBadge) {
      const activeObj = availableModelsList.find((m) => m.id === activeModelId);
      modelStatusBadge.textContent = activeObj && activeObj.loaded ? 'Loaded' : 'Ready';
      modelStatusBadge.className = 'badge-subtle';
    }
  } catch (err) {
    console.warn('Could not fetch models:', err);
    if (modelStatusBadge) {
      modelStatusBadge.textContent = 'Offline';
      modelStatusBadge.className = 'badge-subtle';
    }
  } finally {
    if (refreshModelsBtn) {
      setTimeout(() => refreshModelsBtn.classList.remove('spin-animation'), 400);
    }
  }
}

function updateModelDetailsDisplay() {
  const currentVal = modelSelect ? modelSelect.value : activeModelId;
  const activeObj = availableModelsList.find((m) => m.id === currentVal);
  if (activeObj) {
    if (modelParamText) modelParamText.textContent = activeObj.params ? `Parameters: ${activeObj.params}` : 'Parameters: Local';
    if (modelSizeText) modelSizeText.textContent = activeObj.size_formatted ? `Size: ${activeObj.size_formatted}` : '';
    if (llmModelName) llmModelName.textContent = activeObj.name || activeObj.id;
  }
}

async function handleModelChange(e) {
  const newModel = e.target.value;
  if (!newModel) return;

  if (modelStatusBadge) {
    modelStatusBadge.textContent = 'Switching...';
    modelStatusBadge.className = 'badge-subtle loading';
  }
  systemStatusText.textContent = `Activating model ${newModel}...`;

  try {
    const res = await fetch('/api/llm/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: newModel, load: true }),
    });
    if (!res.ok) throw new Error('Switch request failed');
    const data = await res.json();

    activeModelId = data.model;
    const ejectedInfo = data.ejected_models && data.ejected_models.length > 0 ? ` (ejected: ${data.ejected_models.join(', ')})` : '';
    systemStatusText.textContent = data.message || `Switched model to ${data.model}${ejectedInfo}`;
    if (llmModelName) llmModelName.textContent = data.model;

    // Send through WebSocket as well
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'set_model', model: data.model }));
    }

    if (modelStatusBadge) {
      modelStatusBadge.textContent = data.loaded ? 'Loaded' : 'Ready';
      modelStatusBadge.className = 'badge-subtle';
    }
    updateModelDetailsDisplay();
    // Refresh models to update loaded indicators
    fetchAvailableModels();
  } catch (err) {
    console.error('Failed to change model:', err);
    systemStatusText.textContent = `Model switch error: ${err.message}`;
    if (modelStatusBadge) {
      modelStatusBadge.textContent = 'Error';
      modelStatusBadge.className = 'badge-subtle';
    }
  }
}

async function handleEjectModel() {
  if (ejectModelBtn) ejectModelBtn.disabled = true;
  if (modelStatusBadge) {
    modelStatusBadge.textContent = 'Ejecting...';
    modelStatusBadge.className = 'badge-subtle loading';
  }
  systemStatusText.textContent = 'Ejecting resident model from LM Studio memory...';

  try {
    const targetModel = modelSelect ? modelSelect.value : activeModelId;
    const res = await fetch('/api/llm/eject', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: targetModel }),
    });
    if (!res.ok) throw new Error('Eject request failed');
    const data = await res.json();
    const ejected = (data.ejected_models || []).join(', ');
    systemStatusText.textContent = ejected ? `Ejected from memory: ${ejected}` : 'Model memory released';
    fetchAvailableModels();
  } catch (err) {
    console.error('Failed to eject model:', err);
    systemStatusText.textContent = `Model eject error: ${err.message}`;
  } finally {
    if (ejectModelBtn) ejectModelBtn.disabled = false;
  }
}

// ============================================================================
// Event Listeners & Startup
// ============================================================================
micBtn.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

sendBtn.addEventListener('click', handleTextSubmit);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleTextSubmit();
});

refreshStatusBtn.addEventListener('click', () => {
  checkSystemHealth();
  fetchAvailableModels();
});

if (modelSelect) modelSelect.addEventListener('change', handleModelChange);
if (refreshModelsBtn) refreshModelsBtn.addEventListener('click', fetchAvailableModels);
if (ejectModelBtn) ejectModelBtn.addEventListener('click', handleEjectModel);

window.addEventListener('DOMContentLoaded', () => {
  drawIdleVisualizer();
  checkSystemHealth();
  fetchAvailableModels();
  initWebSocket();
  setInterval(() => {
    checkSystemHealth();
    fetchAvailableModels();
  }, 15000);
});
