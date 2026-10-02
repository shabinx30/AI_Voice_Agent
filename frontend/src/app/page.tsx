"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Topbar } from "@/components/Topbar";
import { ChatArea } from "@/components/ChatArea";
import { Dock } from "@/components/Dock";
import {
  HealthStatus,
  LatencyMetrics,
  ChatMessage,
  WSInboundMessage,
  LMStudioModelInfo,
  ModelsApiResponse,
} from "@/lib/types";
import { StreamAudioQueue } from "@/lib/audio-queue";
import { audioBufferToWavBlob } from "@/lib/wav-encoder";
import { getApiBaseUrl, getWebSocketUrl } from "@/lib/config";

export default function NexusVoiceApp() {
  // Subsystem & voice state
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState<boolean>(false);
  const [selectedSpeaker, setSelectedSpeaker] = useState<string>("af_heart");
  const [playHostAudio, setPlayHostAudio] = useState<boolean>(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState<boolean>(false);

  // Model selection state
  const [availableModels, setAvailableModels] = useState<LMStudioModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("qwen2.5-0.5b-instruct");
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(false);
  const [isEjectingModel, setIsEjectingModel] = useState<boolean>(false);

  // Kokoro TTS Processing Unit state (CPU / NPU)
  const [selectedTTSDevice, setSelectedTTSDevice] = useState<string>("cpu");
  const [ttsEffectiveDevice, setTtsEffectiveDevice] = useState<string>("CPU");
  const [availableTTSDevices, setAvailableTTSDevices] = useState<string[]>(["cpu", "npu"]);
  const [isSwitchingTTSDevice, setIsSwitchingTTSDevice] = useState<boolean>(false);

  // Status & metrics
  const [statusText, setStatusText] = useState<string>(
    "System Ready • Listening Pipeline Active"
  );
  const [metrics, setMetrics] = useState<LatencyMetrics>({});

  // Chat conversation state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingTokenBuffer, setStreamingTokenBuffer] = useState<string>("");
  const streamingTokenBufferRef = useRef<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [inputText, setInputText] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [interactionMode, setInteractionMode] = useState<"assistant" | "tts">("assistant");

  // Recording state
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingSeconds, setRecordingSeconds] = useState<number>(0);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  // Audio queue and refs
  const audioQueueRef = useRef<StreamAudioQueue | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const wsReconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const micAudioCtxRef = useRef<AudioContext | null>(null);

  // Initialize Audio Queue on client
  useEffect(() => {
    audioQueueRef.current = new StreamAudioQueue();
    return () => {
      audioQueueRef.current?.reset();
    };
  }, []);

  // System Health Checker
  const checkHealth = useCallback(async () => {
    setIsCheckingHealth(true);
    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/health`);
      if (!res.ok) throw new Error("Health check failed");
      const data: HealthStatus = await res.json();
      setHealth(data);

      if (data.tts_speakers && data.tts_speakers.length > 0) {
        if (!selectedSpeaker || !data.tts_speakers.includes(selectedSpeaker)) {
          setSelectedSpeaker(data.tts_speakers[0]);
        }
      }
      if (data.lm_studio_model) {
        setSelectedModel((prev) => prev || data.lm_studio_model);
      }
      if (data.tts_device) {
        setSelectedTTSDevice(data.tts_device.toLowerCase());
      }
      if (data.tts_effective_device) {
        setTtsEffectiveDevice(data.tts_effective_device);
      }
      if (data.tts_available_devices && data.tts_available_devices.length > 0) {
        setAvailableTTSDevices(data.tts_available_devices);
      }
      setStatusText("All Pipelines Active • Ready for Voice Input");
    } catch (err) {
      console.warn("Health check error:", err);
      setStatusText("System Disconnected • Ensure backend server is running");
      setHealth((prev) =>
        prev ? { ...prev, lm_studio_connected: false } : null
      );
    } finally {
      setIsCheckingHealth(false);
    }
  }, [selectedSpeaker]);

  // LM Studio Models Fetcher
  const fetchModels = useCallback(async () => {
    setIsLoadingModels(true);
    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/llm/models`);
      if (!res.ok) throw new Error("Failed to fetch LM Studio models");
      const data: ModelsApiResponse = await res.json();
      if (data.models && data.models.length > 0) {
        setAvailableModels(data.models);
      }
      if (data.current_model) {
        setSelectedModel(data.current_model);
      }
    } catch (err) {
      console.warn("Could not fetch models:", err);
    } finally {
      setIsLoadingModels(false);
    }
  }, []);

  // Handle Model Switch (guarantees prior model is ejected before loading new model)
  const handleSelectModel = useCallback(
    async (modelId: string) => {
      if (!modelId) return;
      setSelectedModel(modelId);
      setStatusText(`Ejecting other models and activating ${modelId}...`);
      try {
        const baseUrl = getApiBaseUrl();
        const res = await fetch(`${baseUrl}/api/llm/model`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: modelId, load: true }),
        });
        if (!res.ok) throw new Error("Failed to switch model");
        const data = await res.json();
        const ejectedNote =
          data.ejected_models && data.ejected_models.length > 0
            ? ` (ejected: ${data.ejected_models.join(", ")})`
            : "";
        setStatusText(data.message || `Switched model to ${modelId}${ejectedNote}`);

        // Update local health state
        setHealth((prev) =>
          prev ? { ...prev, lm_studio_model: modelId } : null
        );

        // Notify WebSocket if connected
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({ type: "set_model", model: modelId })
          );
        }

        // Re-scan models to reflect in-memory status
        fetchModels();
      } catch (err) {
        console.error("Model switch error:", err);
        setStatusText(`Failed to switch model: ${err}`);
      }
    },
    [fetchModels]
  );

  // Handle Model Eject (unloads model from memory to free VRAM)
  const handleEjectModel = useCallback(
    async (modelId?: string) => {
      setIsEjectingModel(true);
      setStatusText(
        modelId
          ? `Ejecting model '${modelId}' from memory...`
          : "Ejecting resident models from LM Studio..."
      );
      try {
        const baseUrl = getApiBaseUrl();
        const res = await fetch(`${baseUrl}/api/llm/eject`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: modelId }),
        });
        if (!res.ok) throw new Error("Failed to eject model");
        const data = await res.json();
        const ejectedNames = (data.ejected_models || []).join(", ");
        setStatusText(
          ejectedNames
            ? `Successfully ejected from memory: ${ejectedNames}`
            : "No active models currently resident in memory"
        );
        fetchModels();
      } catch (err) {
        console.error("Eject error:", err);
        setStatusText(`Failed to eject model: ${err}`);
      } finally {
        setIsEjectingModel(false);
      }
    },
    [fetchModels]
  );

  // Handle Kokoro TTS Processing Unit Switch (CPU / NPU)
  const handleSelectTTSDevice = useCallback(
    async (device: string) => {
      const target = device.toLowerCase();
      if (!target || isSwitchingTTSDevice) return;
      setIsSwitchingTTSDevice(true);
      const targetLabel =
        target === "npu_only"
          ? "NPU Only (All Stages)"
          : target === "npu"
          ? "NPU (Hybrid)"
          : "CPU";
      setStatusText(`Switching Kokoro TTS processing unit to ${targetLabel}...`);
      setSelectedTTSDevice(target);

      try {
        const baseUrl = getApiBaseUrl();
        const res = await fetch(`${baseUrl}/api/tts/device`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device: target }),
        });
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          const errMsg =
            errData.detail ||
            (res.status === 404
              ? "Backend route not found (please restart the backend server)"
              : `Server returned status ${res.status}`);
          throw new Error(errMsg);
        }
        const data = await res.json();
        if (data.effective_device) {
          setTtsEffectiveDevice(data.effective_device);
        }
        setStatusText(data.message || `Kokoro TTS processing unit set to ${targetLabel}`);

        // Update health state
        setHealth((prev) =>
          prev
            ? {
                ...prev,
                tts_device: target,
                tts_effective_device: data.effective_device,
              }
            : null
        );

        // Notify WebSocket if connected
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({ type: "set_tts_device", device: target })
          );
        }
      } catch (err) {
        console.error("TTS device switch error:", err);
        setStatusText(`Failed to switch TTS unit: ${err}`);
      } finally {
        setIsSwitchingTTSDevice(false);
      }
    },
    [isSwitchingTTSDevice]
  );

  // WebSocket Message Dispatcher
  const handleWebSocketMessage = useCallback(
    (msg: WSInboundMessage) => {
      if (msg.type === "pong") return;

      if (msg.type === "status") {
        setStatusText(msg.message);
        return;
      }

      if (msg.type === "models") {
        if (msg.models && msg.models.length > 0) {
          setAvailableModels(msg.models);
        }
        if (msg.current_model) {
          setSelectedModel(msg.current_model);
        }
        return;
      }

      if (msg.type === "model_changed") {
        const changedModel = msg.model;
        if (changedModel) {
          setSelectedModel(changedModel);
          setHealth((prev) => (prev ? { ...prev, lm_studio_model: changedModel } : null));
        }
        if (msg.message) {
          setStatusText(msg.message);
        }
        return;
      }

      if (msg.type === "tts_device") {
        if (msg.device) {
          setSelectedTTSDevice(msg.device.toLowerCase());
        }
        if (msg.effective_device) {
          setTtsEffectiveDevice(msg.effective_device);
        }
        if (msg.available_devices && msg.available_devices.length > 0) {
          setAvailableTTSDevices(msg.available_devices);
        }
        return;
      }

      if (msg.type === "tts_device_changed") {
        if (msg.device) {
          setSelectedTTSDevice(msg.device.toLowerCase());
        }
        if (msg.effective_device) {
          setTtsEffectiveDevice(msg.effective_device);
        }
        if (msg.message) {
          setStatusText(msg.message);
        }
        return;
      }

      if (msg.type === "transcription") {
        setStatusText("Speech transcribed • Streaming reply from LM Studio...");
        // Update user message text
        setMessages((prev) => {
          const updated = [...prev];
          const lastUserIdx = updated.findLastIndex((m) => m.role === "user");
          if (lastUserIdx !== -1) {
            updated[lastUserIdx] = {
              ...updated[lastUserIdx],
              text: msg.user_text,
            };
          }
          return updated;
        });
        return;
      }

      if (msg.type === "token") {
        setIsStreaming(true);
        streamingTokenBufferRef.current += (msg.text || "");
        setStreamingTokenBuffer((prev) => prev + (msg.text || ""));
        return;
      }

      if (msg.type === "chunk") {
        setStatusText(`Speaking sentence ${(msg.index ?? 0) + 1}...`);
        if (!playHostAudio && msg.audio_base64 && audioQueueRef.current) {
          audioQueueRef.current.enqueue(msg.audio_base64);
        }
        return;
      }

      if (msg.type === "result") {
        setIsStreaming(false);
        setIsProcessing(false);
        const streamedText = streamingTokenBufferRef.current;
        streamingTokenBufferRef.current = "";
        setStreamingTokenBuffer("");

        if (msg.metrics) {
          setMetrics(msg.metrics);
        }

        setStatusText("All Pipelines Active • Ready for Voice Input");

        setMessages((prev) => {
          const updated = [...prev];
          const lastAssistantIdx = updated.findLastIndex(
            (m) => m.role === "assistant"
          );
          if (lastAssistantIdx !== -1) {
            let finalText = msg.assistant_text || updated[lastAssistantIdx].text;
            if (
              streamedText &&
              streamedText.includes("\n") &&
              (!finalText || !finalText.includes("\n"))
            ) {
              finalText = streamedText;
            }
            updated[lastAssistantIdx] = {
              ...updated[lastAssistantIdx],
              text: finalText,
              audioBase64: msg.audio_base64,
              isStreaming: false,
            };
          }
          const lastUserIdx = updated.findLastIndex((m) => m.role === "user");
          if (lastUserIdx !== -1 && msg.user_text) {
            updated[lastUserIdx] = {
              ...updated[lastUserIdx],
              text: msg.user_text,
            };
          }
          return updated;
        });
        return;
      }

      if (msg.type === "error") {
        setIsStreaming(false);
        setIsProcessing(false);
        streamingTokenBufferRef.current = "";
        setStreamingTokenBuffer("");
        setStatusText(`Error: ${msg.message}`);

        setMessages((prev) => {
          const updated = [...prev];
          const lastAssistantIdx = updated.findLastIndex(
            (m) => m.role === "assistant"
          );
          if (lastAssistantIdx !== -1) {
            updated[lastAssistantIdx] = {
              ...updated[lastAssistantIdx],
              text: `Error: ${msg.message}`,
              isStreaming: false,
            };
          }
          return updated;
        });
      }
    },
    [playHostAudio]
  );

  // Initialize WebSocket connection
  const initWebSocket = useCallback(() => {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    try {
      const wsUrl = getWebSocketUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatusText("All Pipelines Active • Ready for Voice Input");
        if (wsReconnectTimeoutRef.current) {
          clearTimeout(wsReconnectTimeoutRef.current);
          wsReconnectTimeoutRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const data: WSInboundMessage = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (err) {
          console.error("WebSocket message JSON parse error:", err);
        }
      };

      ws.onclose = () => {
        if (!wsReconnectTimeoutRef.current) {
          wsReconnectTimeoutRef.current = setTimeout(initWebSocket, 3000);
        }
      };

      ws.onerror = (err) => {
        console.warn("WebSocket error:", err);
      };
    } catch (err) {
      console.error("Could not construct WebSocket:", err);
    }
  }, [handleWebSocketMessage]);

  // Initial Mount & Heartbeat
  useEffect(() => {
    const initTimer = setTimeout(() => {
      checkHealth();
      fetchModels();
      initWebSocket();
    }, 0);

    const healthInterval = setInterval(() => {
      checkHealth();
      fetchModels();
    }, 20000);
    const pingInterval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 25000);

    return () => {
      clearTimeout(initTimer);
      clearInterval(healthInterval);
      clearInterval(pingInterval);
      if (wsReconnectTimeoutRef.current) {
        clearTimeout(wsReconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [checkHealth, fetchModels, initWebSocket]);

  // Audio Recording Handlers
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStreamRef.current = stream;

      // Unlock AudioContext
      audioQueueRef.current?.getAudioContext();

      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const micAudioCtx = new AudioContextClass();
      micAudioCtxRef.current = micAudioCtx;

      const analyser = micAudioCtx.createAnalyser();
      analyser.fftSize = 1024;
      const source = micAudioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      setAnalyserNode(analyser);

      recordedChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = handleRecordingComplete;

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);
      setStatusText("Listening to your microphone...");

      const startTime = Date.now();
      timerIntervalRef.current = setInterval(() => {
        setRecordingSeconds(Math.floor((Date.now() - startTime) / 1000));
      }, 500);
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Please grant microphone permission in your browser.");
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    setIsRecording(false);
    setAnalyserNode(null);

    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }

    if (micAudioCtxRef.current) {
      micAudioCtxRef.current.close().catch(() => {});
      micAudioCtxRef.current = null;
    }
  };

  const handleToggleRecord = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Recording Complete: Transcribe & Synthesize
  const handleRecordingComplete = async () => {
    const rawChunks = recordedChunksRef.current;
    if (rawChunks.length === 0) return;

    const mimeType = mediaRecorderRef.current?.mimeType || "audio/webm";
    const rawBlob = new Blob(rawChunks, { type: mimeType });

    setIsProcessing(true);
    setStatusText("Transcribing speech with OpenVINO Whisper Base INT8...");
    streamingTokenBufferRef.current = "";
    setStreamingTokenBuffer("");

    // Create user placeholder and assistant placeholder
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: "user",
        text: "Processing speech...",
        timestamp: Date.now(),
      },
      {
        id: assistantMsgId,
        role: "assistant",
        text: "",
        isStreaming: true,
        timestamp: Date.now(),
      },
    ]);

    audioQueueRef.current?.reset();

    // Convert raw Blob to 16-bit PCM WAV
    let audioBlobToSend = rawBlob;
    try {
      const arrayBuf = await rawBlob.arrayBuffer();
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const decodeCtx = new AudioContextClass();
      const decodedBuffer = await decodeCtx.decodeAudioData(arrayBuf);
      audioBlobToSend = audioBufferToWavBlob(decodedBuffer);
      decodeCtx.close();
    } catch (convErr) {
      console.debug("Direct audio blob conversion used:", convErr);
    }

    // Convert to Base64
    const arrayBuf = await audioBlobToSend.arrayBuffer();
    let binary = "";
    const bytes = new Uint8Array(arrayBuf);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64Audio = window.btoa(binary);

    // Transmit via WebSocket if open
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "audio",
          data: base64Audio,
          speaker: selectedSpeaker,
          play_audio: playHostAudio,
          model: selectedModel,
          tts_device: selectedTTSDevice,
        })
      );
    } else {
      // Fallback via HTTP POST
      handleHttpAudioFallback(audioBlobToSend, assistantMsgId);
    }
  };

  // HTTP Audio Fallback
  const handleHttpAudioFallback = async (
    blob: Blob,
    assistantMsgId: string
  ) => {
    const formData = new FormData();
    formData.append("file", blob, "input.wav");
    formData.append("speaker", selectedSpeaker);
    formData.append("play_audio", String(playHostAudio));
    formData.append("tts_device", selectedTTSDevice);

    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/interact`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Interaction failed");
      }

      const data = await res.json();

      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === assistantMsgId) {
            return {
              ...msg,
              text: data.assistant_text,
              audioBase64: data.audio_base64,
              isStreaming: false,
            };
          }
          if (msg.role === "user" && msg.text === "Processing speech...") {
            return { ...msg, text: data.user_text };
          }
          return msg;
        })
      );

      if (data.metrics) {
        setMetrics(data.metrics);
      }

      if (data.audio_base64 && !playHostAudio && audioPlayerRef.current) {
        audioPlayerRef.current.src = `data:audio/wav;base64,${data.audio_base64}`;
        audioPlayerRef.current.play().catch(() => {});
      }

      setStatusText("All Pipelines Active • Ready for Voice Input");
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Error processing voice";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, text: `Error: ${errMsg}`, isStreaming: false }
            : msg
        )
      );
      setStatusText("Failed to process request.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Text Submission Handler
  const handleTextSubmit = async () => {
    const trimmed = inputText.trim();
    if (!trimmed || isProcessing || isRecording) return;

    setInputText("");
    setIsProcessing(true);
    setStatusText("Streaming reply and synthesizing speech...");
    streamingTokenBufferRef.current = "";
    setStreamingTokenBuffer("");

    audioQueueRef.current?.reset();

    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: "user",
        text: trimmed,
        timestamp: Date.now(),
      },
      {
        id: assistantMsgId,
        role: "assistant",
        text: "",
        isStreaming: true,
        timestamp: Date.now(),
      },
    ]);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "text",
          prompt: trimmed,
          speaker: selectedSpeaker,
          play_audio: playHostAudio,
          model: selectedModel,
          tts_device: selectedTTSDevice,
        })
      );
    } else {
      // Fallback via HTTP SSE tokens + TTS
      handleHttpTextFallback(trimmed, assistantMsgId);
    }
  };

  // Direct TTS Submission Handler
  const handleDirectTTSSubmit = async () => {
    const trimmed = inputText.trim();
    if (!trimmed || isProcessing || isRecording) return;

    setInputText("");
    setIsProcessing(true);
    setStatusText("Synthesizing speech with Kokoro-82M...");
    streamingTokenBufferRef.current = "";
    setStreamingTokenBuffer("");

    audioQueueRef.current?.reset();

    const userMsgId = `tts-user-${Date.now()}`;
    const ttsMsgId = `tts-out-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: "user",
        text: trimmed,
        timestamp: Date.now(),
        isTTSOnly: true,
      },
      {
        id: ttsMsgId,
        role: "assistant",
        text: trimmed,
        isStreaming: true,
        timestamp: Date.now(),
        isTTSOnly: true,
        speaker: selectedSpeaker,
      },
    ]);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "tts",
          text: trimmed,
          speaker: selectedSpeaker,
          play_audio: playHostAudio,
          tts_device: selectedTTSDevice,
        })
      );
    } else {
      // Fallback via HTTP /api/tts/generate
      handleHttpTTSFallback(trimmed, ttsMsgId);
    }
  };

  // HTTP Direct TTS Fallback
  const handleHttpTTSFallback = async (
    textToSynth: string,
    ttsMsgId: string
  ) => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/tts/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: textToSynth,
          speaker: selectedSpeaker,
          device: selectedTTSDevice,
          play_audio: playHostAudio,
        }),
      });

      if (!res.ok) {
        throw new Error(`TTS synthesis failed (Status ${res.status})`);
      }

      const data = await res.json();
      if (data.metrics) {
        setMetrics(data.metrics);
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === ttsMsgId
            ? {
                ...msg,
                text: data.assistant_text || textToSynth,
                audioBase64: data.audio_base64,
                isStreaming: false,
              }
            : msg
        )
      );

      if (!playHostAudio && data.audio_base64 && audioPlayerRef.current) {
        audioPlayerRef.current.src = `data:audio/wav;base64,${data.audio_base64}`;
        audioPlayerRef.current.play().catch(() => {});
      }

      setStatusText("All Pipelines Active • Ready for Voice Input");
    } catch (err: any) {
      console.error("HTTP TTS fallback error:", err);
      setStatusText(`TTS Error: ${err?.message || err}`);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === ttsMsgId
            ? {
                ...msg,
                text: `Error synthesizing speech: ${err?.message || err}`,
                isStreaming: false,
              }
            : msg
        )
      );
    } finally {
      setIsProcessing(false);
    }
  };

  // HTTP Text Fallback
  const handleHttpTextFallback = async (
    promptText: string,
    assistantMsgId: string
  ) => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/chat/tokens`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: promptText, model: selectedModel }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Token stream failed (Status ${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let fullText = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let sep: number;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);

          for (const line of frame.split("\n")) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith("data:")) continue;
            const dataStr = trimmedLine.slice(5).trim();
            if (dataStr === "[DONE]") continue;

            try {
              const evt = JSON.parse(dataStr);
              if (evt.token) {
                fullText += evt.token;
                setStreamingTokenBuffer(fullText);
              }
            } catch {
              // Ignore non-json lines
            }
          }
        }
      }

      setStreamingTokenBuffer("");
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, text: fullText, isStreaming: false }
            : msg
        )
      );

      // Synthesize audio
      const ttsRes = await fetch(`${baseUrl}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: fullText,
          speaker: selectedSpeaker,
          device: selectedTTSDevice,
        }),
      });

      if (ttsRes.ok) {
        const audioBlob = await ttsRes.blob();
        const readerBlob = new FileReader();
        readerBlob.onloadend = () => {
          const base64Data = (readerBlob.result as string)?.split(",")[1];
          if (base64Data) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  ? { ...msg, audioBase64: base64Data }
                  : msg
              )
            );
            if (!playHostAudio && audioPlayerRef.current) {
              audioPlayerRef.current.src = `data:audio/wav;base64,${base64Data}`;
              audioPlayerRef.current.play().catch(() => {});
            }
          }
        };
        readerBlob.readAsDataURL(audioBlob);
      }

      setStatusText("All Pipelines Active • Ready for Voice Input");
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Error processing text";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMsgId
            ? { ...msg, text: `Error: ${errMsg}`, isStreaming: false }
            : msg
        )
      );
      setStatusText("Error processing text.");
    } finally {
      setIsProcessing(false);
      setStreamingTokenBuffer("");
    }
  };

  // Replay Full Audio
  const handleReplayAudio = (audioBase64: string) => {
    audioQueueRef.current?.reset();
    if (audioPlayerRef.current) {
      audioPlayerRef.current.src = `data:audio/wav;base64,${audioBase64}`;
      audioPlayerRef.current.play().catch(() => {});
    }
  };

  return (
    <div className="relative w-screen h-screen max-h-screen overflow-hidden p-3 md:p-4.5 flex gap-4 bg-white">

      {/* Sidebar Component */}
      <Sidebar
        health={health}
        availableModels={availableModels}
        selectedModel={selectedModel}
        onSelectModel={handleSelectModel}
        isLoadingModels={isLoadingModels}
        onRefreshModels={fetchModels}
        onEjectModel={handleEjectModel}
        isEjectingModel={isEjectingModel}
        selectedSpeaker={selectedSpeaker}
        onSelectSpeaker={setSelectedSpeaker}
        selectedTTSDevice={selectedTTSDevice}
        onSelectTTSDevice={handleSelectTTSDevice}
        isSwitchingTTSDevice={isSwitchingTTSDevice}
        availableTTSDevices={availableTTSDevices}
        ttsEffectiveDevice={ttsEffectiveDevice}
        playHostAudio={playHostAudio}
        onTogglePlayHostAudio={setPlayHostAudio}
        isCheckingHealth={isCheckingHealth}
        onRefreshHealth={checkHealth}
        isOpenMobile={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full min-h-0 min-w-0 gap-3.5 z-10">
        {/* Topbar Component */}
        <Topbar
          statusText={statusText}
          metrics={metrics}
          onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
          ttsDevice={selectedTTSDevice}
          onSelectTTSDevice={handleSelectTTSDevice}
        />

        {/* Chat Area Component */}
        <ChatArea
          messages={messages}
          streamingTokenBuffer={streamingTokenBuffer}
          isStreaming={isStreaming}
          onReplayAudio={handleReplayAudio}
        />

        {/* Visualizer & Interaction Dock Component */}
        <Dock
          isRecording={isRecording}
          recordingSeconds={recordingSeconds}
          analyserNode={analyserNode}
          onToggleRecord={handleToggleRecord}
          inputText={inputText}
          onChangeInputText={setInputText}
          onSubmitText={handleTextSubmit}
          onSubmitDirectTTS={handleDirectTTSSubmit}
          interactionMode={interactionMode}
          onToggleInteractionMode={setInteractionMode}
          isProcessing={isProcessing}
        />
      </main>

      {/* Hidden Native Audio Element for Replay */}
      <audio ref={audioPlayerRef} className="hidden" />
    </div>
  );
}
