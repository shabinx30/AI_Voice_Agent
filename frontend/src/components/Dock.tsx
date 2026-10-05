"use client";

import React, { useRef, useEffect } from "react";
import { Mic, Square, Send, Volume2, Bot, Brain, Play, Pause } from "lucide-react";

interface DockProps {
  isRecording: boolean;
  recordingSeconds: number;
  analyserNode: AnalyserNode | null;
  onToggleRecord: () => void;
  inputText: string;
  onChangeInputText: (val: string) => void;
  onSubmitText: () => void;
  onSubmitDirectTTS: () => void;
  interactionMode: "assistant" | "tts";
  onToggleInteractionMode: (mode: "assistant" | "tts") => void;
  isProcessing: boolean;
  isStreaming?: boolean;
  isThinking?: boolean;
  isAudioPlaying?: boolean;
  isAudioPaused?: boolean;
  onToggleAudioPlayPause?: () => void;
  onCancelGeneration?: () => void;
  thinkMode?: boolean;
  onToggleThinkMode?: () => void;
  supportsThinking?: boolean;
  reasoningEffort?: "low" | "medium" | "high";
  onChangeReasoningEffort?: (effort: "low" | "medium" | "high") => void;
  activeModel?: string;
}

export function Dock({
  isRecording,
  recordingSeconds,
  analyserNode,
  onToggleRecord,
  inputText,
  onChangeInputText,
  onSubmitText,
  onSubmitDirectTTS,
  interactionMode,
  onToggleInteractionMode,
  isProcessing,
  isStreaming = false,
  isThinking = false,
  isAudioPlaying = false,
  isAudioPaused = false,
  onToggleAudioPlayPause,
  onCancelGeneration,
  thinkMode,
  onToggleThinkMode,
  supportsThinking,
  reasoningEffort,
  onChangeReasoningEffort,
  activeModel,
}: DockProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);

  // Format recording timer: MM:SS
  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const rem = secs % 60;
    return `${mins.toString().padStart(2, "0")}:${rem
      .toString()
      .padStart(2, "0")}`;
  };

  // Canvas visualizer animation loop - runs ONLY when microphone is recording
  useEffect(() => {
    if (!isRecording) {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let localAnimId: number;

    const render = () => {
      // Ensure canvas resolution matches element size for crisp rendering
      const rect = canvas.getBoundingClientRect();
      if (
        rect.width > 0 &&
        (canvas.width !== Math.floor(rect.width) ||
          canvas.height !== Math.floor(rect.height))
      ) {
        canvas.width = Math.floor(rect.width);
        canvas.height = Math.floor(rect.height);
      }

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      if (analyserNode) {
        // Live audio waveform from microphone - preserved red
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserNode.getByteTimeDomainData(dataArray);

        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#dc2626";

        ctx.beginPath();
        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * height) / 2;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
      } else {
        // Microphone active but analyser initializing: clean flat line
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#dc2626";
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
      }

      localAnimId = requestAnimationFrame(render);
    };

    localAnimId = requestAnimationFrame(render);
    animFrameRef.current = localAnimId;

    return () => {
      if (localAnimId) cancelAnimationFrame(localAnimId);
    };
  }, [isRecording, analyserNode]);

  const isGenerating = Boolean(isProcessing || isStreaming);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape" && isGenerating) {
      e.preventDefault();
      onCancelGeneration?.();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isGenerating) return;
      if (interactionMode === "tts") {
        onSubmitDirectTTS();
      } else {
        onSubmitText();
      }
    }
  };

  return (
    <footer className="glass-panel shrink-0 p-4 md:p-5 flex flex-col gap-3 shadow-xs bg-white border border-neutral-200">
      {/* Audio Waveform & Timer Wrapper - Rendered only when using the microphone */}
      {isRecording && (
        <div className="flex items-center justify-between w-full h-11 bg-neutral-50 rounded-xl px-4 border border-neutral-200 animate-wave-appear">
          <canvas
            ref={canvasRef}
            width={650}
            height={44}
            className="flex-1 h-full w-full max-w-full"
          />

          <div className="font-mono text-xs text-rose-700 font-semibold ml-3 flex items-center gap-1.5 shrink-0 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
            <span className="w-2 h-2 rounded-full bg-rose-600 animate-pulse" />
            {formatTime(recordingSeconds)}
          </div>
        </div>
      )}


      {/* Mode Selector & Controls Row */}
      <div className="flex items-center justify-between px-1 text-xs gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center p-0.5 rounded-lg bg-neutral-100 border border-neutral-200">
            <button
              type="button"
              onClick={() => onToggleInteractionMode("assistant")}
              className={`px-3 py-1 rounded-md font-medium flex items-center gap-1.5 transition-all cursor-pointer ${
                interactionMode === "assistant"
                  ? "bg-black text-white shadow-xs"
                  : "text-neutral-600 hover:text-black"
              }`}
            >
              <Bot className="w-3.5 h-3.5" />
              <span>AI Assistant</span>
            </button>
            <button
              type="button"
              onClick={() => onToggleInteractionMode("tts")}
              className={`px-3 py-1 rounded-md font-medium flex items-center gap-1.5 transition-all cursor-pointer ${
                interactionMode === "tts"
                  ? "bg-black text-white shadow-xs"
                  : "text-neutral-600 hover:text-black"
              }`}
            >
              <Volume2 className="w-3.5 h-3.5" />
              <span>Direct TTS Tool</span>
            </button>
          </div>

          {/* Think Mode Toggle & Reasoning Effort */}
          {interactionMode === "assistant" && onToggleThinkMode && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                id="dock-think-mode-toggle"
                onClick={supportsThinking ? onToggleThinkMode : undefined}
                disabled={!supportsThinking}
                aria-disabled={!supportsThinking}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 border transition-all select-none ${
                  !supportsThinking
                    ? "opacity-50 bg-neutral-100 text-neutral-400 border-neutral-200 cursor-not-allowed shadow-none"
                    : thinkMode
                    ? "bg-violet-600 text-white border-violet-700 shadow-2xs hover:bg-violet-700 cursor-pointer"
                    : "bg-neutral-100 hover:bg-neutral-200 text-neutral-700 hover:text-black border-neutral-300 shadow-2xs cursor-pointer"
                }`}
                title={
                  supportsThinking
                    ? `Think Mode: ${
                        thinkMode
                          ? "ON (" + (reasoningEffort || "medium") + " effort)"
                          : "OFF"
                      }. Click to toggle reasoning tokens.`
                    : `Think Mode is not supported by ${activeModel || "the active model"}. Select a reasoning model (e.g. DeepSeek-R1, Qwen3, QwQ) to enable.`
                }
              >
                <Brain
                  className={`w-3.5 h-3.5 ${
                    !supportsThinking
                      ? "text-neutral-400"
                      : thinkMode
                      ? "text-violet-200 animate-pulse"
                      : "text-neutral-500"
                  }`}
                />
                <span>
                  {supportsThinking
                    ? `Think ${thinkMode ? "ON" : "OFF"}`
                    : "Think"}
                </span>
                {supportsThinking ? (
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      thinkMode ? "bg-emerald-300" : "bg-neutral-400"
                    }`}
                    title="Reasoning model supported"
                  />
                ) : (
                  <span className="text-[9px] font-mono uppercase bg-neutral-200/80 text-neutral-400 px-1 py-0.2 rounded font-normal">
                    N/A
                  </span>
                )}
              </button>

              {/* Reasoning Effort Selector (shown ONLY when Think Mode is active and supported) */}
              {supportsThinking && thinkMode && onChangeReasoningEffort && (
                <div className="flex items-center bg-neutral-100 border border-neutral-200 rounded-lg p-0.5 text-[10px] animate-in fade-in duration-150">
                  {(["low", "medium", "high"] as const).map((eff) => (
                    <button
                      key={eff}
                      type="button"
                      onClick={() => onChangeReasoningEffort(eff)}
                      className={`px-2 py-0.5 rounded font-semibold uppercase transition-all cursor-pointer ${
                        (reasoningEffort || "medium") === eff
                          ? "bg-black text-white shadow-2xs"
                          : "text-neutral-500 hover:text-black hover:bg-white/60"
                      }`}
                    >
                      {eff}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="text-[11px] text-neutral-500 font-mono hidden sm:inline-block">
          {interactionMode === "tts"
            ? "Kokoro-82M speech synthesis (bypasses LLM)"
            : supportsThinking && thinkMode
            ? "Reasoning tokens enabled"
            : supportsThinking
            ? "Reasoning model ready"
            : "Conversational voice & text agent"}
        </div>
      </div>

      {/* Control Buttons & Input Bar */}
      <div className="flex items-center gap-3">
        {/* Mic / Stop Button */}
        <button
          onClick={onToggleRecord}
          disabled={isProcessing}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
          title={isRecording ? "Click to stop recording" : "Click to record voice"}
          className={`relative w-13 h-13 rounded-full flex items-center justify-center shrink-0 text-white transition-all duration-200 shadow-md cursor-pointer disabled:opacity-50 ${
            isRecording
              ? "bg-red-600 hover:bg-red-700 mic-recording-pulse shadow-red-500/30"
              : "bg-black hover:bg-neutral-800 hover:scale-105 shadow-black/20"
          }`}
        >
          {isRecording ? (
            <Square className="w-5 h-5 fill-current" />
          ) : (
            <Mic className="w-5 h-5" />
          )}
        </button>

        {/* Text Input Wrap */}
        <div className="flex-1 relative flex items-center">
          <input
            type="text"
            value={inputText}
            onChange={(e) => onChangeInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isRecording
                ? "Listening to voice input..."
                : isAudioPaused
                ? "Audio playback paused • Click Resume or press Space"
                : isGenerating
                ? isThinking
                  ? "Reasoning in progress... Press Stop or Esc to cancel"
                  : "Generating response... Press Stop or Esc to cancel"
                : interactionMode === "tts"
                ? "Type text to synthesize and speak directly with Kokoro TTS..."
                : supportsThinking && thinkMode
                ? "Type a message (Think mode active)..."
                : "Type a message or click mic to talk..."
            }
            disabled={isRecording}
            className={`w-full h-12 bg-white border rounded-full pl-5 pr-22 text-sm text-black placeholder:text-neutral-400 outline-none transition-all focus:ring-2 disabled:opacity-50 ${
              supportsThinking && thinkMode
                ? "border-violet-300 focus:border-violet-600 focus:ring-violet-500/10"
                : "border-neutral-300 focus:border-black focus:ring-black/10"
            }`}
          />

          <div className="absolute right-1.5 flex items-center gap-1">
            {/* Quick Direct TTS Button (available in assistant mode too when not generating) */}
            {interactionMode === "assistant" && !isGenerating && (
              <button
                type="button"
                onClick={onSubmitDirectTTS}
                disabled={!inputText.trim() || isGenerating || isRecording}
                title="Synthesize and speak directly with TTS (bypasses LLM)"
                className="w-9 h-9 rounded-full bg-neutral-100 hover:bg-neutral-200 border border-neutral-300 disabled:opacity-30 flex items-center justify-center text-black transition-all cursor-pointer shadow-xs"
              >
                <Volume2 className="w-4 h-4 text-black" />
              </button>
            )}

            {/* Primary Action Button */}
            {interactionMode === "tts" ? (
              <button
                type="button"
                onClick={onSubmitDirectTTS}
                disabled={!inputText.trim() || isGenerating || isRecording}
                title="Synthesize Speech (TTS)"
                className="px-3.5 h-9 rounded-full bg-black hover:bg-neutral-800 disabled:opacity-30 disabled:hover:bg-black flex items-center gap-1.5 text-white text-xs font-semibold transition-all cursor-pointer shadow-xs"
              >
                <Volume2 className="w-3.5 h-3.5" />
                <span>Speak</span>
              </button>
            ) : isGenerating ? (
              <div className="flex items-center gap-1.5">
                {onToggleAudioPlayPause && (isAudioPlaying || isAudioPaused) && (
                  <button
                    type="button"
                    id="dock-toggle-audio-btn"
                    onClick={onToggleAudioPlayPause}
                    title={isAudioPaused ? "Resume audio playback (Space)" : "Pause audio playback (Space)"}
                    aria-label={isAudioPaused ? "Resume audio" : "Pause audio"}
                    className={`h-9 px-3 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer shadow-xs ${
                      isAudioPaused
                        ? "bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300"
                        : "bg-neutral-900 hover:bg-black text-white"
                    }`}
                  >
                    {isAudioPaused ? (
                      <>
                        <Play className="w-3.5 h-3.5 fill-current" />
                        <span>Resume</span>
                      </>
                    ) : (
                      <>
                        <Pause className="w-3.5 h-3.5 fill-current" />
                        <span>Pause</span>
                      </>
                    )}
                  </button>
                )}
                <button
                  type="button"
                  id="dock-cancel-generation-btn"
                  onClick={onCancelGeneration}
                  title="Cancel text generation (Esc)"
                  aria-label="Cancel text generation"
                  className="h-9 px-3.5 rounded-full bg-red-600 hover:bg-red-700 active:scale-95 text-white text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer shadow-sm animate-in fade-in"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Stop</span>
                </button>
              </div>
            ) : (isAudioPlaying || isAudioPaused) && onToggleAudioPlayPause ? (
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  id="dock-toggle-audio-btn"
                  onClick={onToggleAudioPlayPause}
                  title={isAudioPaused ? "Resume audio playback (Space)" : "Pause audio playback (Space)"}
                  aria-label={isAudioPaused ? "Resume audio" : "Pause audio"}
                  className={`h-9 px-3 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer shadow-xs ${
                    isAudioPaused
                      ? "bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300 animate-pulse"
                      : "bg-neutral-900 hover:bg-black text-white"
                  }`}
                >
                  {isAudioPaused ? (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>Resume</span>
                    </>
                  ) : (
                    <>
                      <Pause className="w-3.5 h-3.5 fill-current" />
                      <span>Pause</span>
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={onSubmitText}
                  disabled={!inputText.trim() || isGenerating || isRecording}
                  title="Send Message to Assistant"
                  className="w-9 h-9 rounded-full bg-black hover:bg-neutral-800 disabled:opacity-30 disabled:hover:bg-black flex items-center justify-center text-white transition-all cursor-pointer shadow-xs"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={onSubmitText}
                disabled={!inputText.trim() || isGenerating || isRecording}
                title="Send Message to Assistant"
                className="w-9 h-9 rounded-full bg-black hover:bg-neutral-800 disabled:opacity-30 disabled:hover:bg-black flex items-center justify-center text-white transition-all cursor-pointer shadow-xs"
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </footer>
  );
}
