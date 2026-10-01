"use client";

import React, { useRef, useEffect } from "react";
import { Mic, Square, Send } from "lucide-react";

interface DockProps {
  isRecording: boolean;
  recordingSeconds: number;
  analyserNode: AnalyserNode | null;
  onToggleRecord: () => void;
  inputText: string;
  onChangeInputText: (val: string) => void;
  onSubmitText: () => void;
  isProcessing: boolean;
}

export function Dock({
  isRecording,
  recordingSeconds,
  analyserNode,
  onToggleRecord,
  inputText,
  onChangeInputText,
  onSubmitText,
  isProcessing,
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

  // Canvas visualizer animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let localAnimId: number;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      if (isRecording && analyserNode) {
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
        // Idle gentle harmonic sine wave - monochrome
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
        ctx.beginPath();

        const sliceWidth = width / 60;
        let x = 0;
        const time = Date.now() * 0.003;

        for (let i = 0; i < 60; i++) {
          const y = height / 2 + Math.sin(i * 0.2 + time) * 4;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmitText();
    }
  };

  return (
    <footer className="glass-panel shrink-0 p-4 md:p-5 flex flex-col gap-3 shadow-xs bg-white border border-neutral-200">
      {/* Audio Waveform & Timer Wrapper */}
      <div className="flex items-center justify-between w-full h-11 bg-neutral-50 rounded-xl px-4 border border-neutral-200">
        <canvas
          ref={canvasRef}
          width={650}
          height={44}
          className="flex-1 h-full w-full max-w-full"
        />

        {isRecording && (
          <div className="font-mono text-xs text-rose-700 font-semibold ml-3 flex items-center gap-1.5 shrink-0 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
            <span className="w-2 h-2 rounded-full bg-rose-600 animate-pulse" />
            {formatTime(recordingSeconds)}
          </div>
        )}
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
                : "Type a message or click mic to talk..."
            }
            disabled={isRecording}
            className="w-full h-12 bg-white border border-neutral-300 focus:border-black rounded-full pl-5 pr-13 text-sm text-black placeholder:text-neutral-400 outline-none transition-all focus:ring-2 focus:ring-black/10 disabled:opacity-50"
          />

          <button
            onClick={onSubmitText}
            disabled={!inputText.trim() || isProcessing || isRecording}
            title="Send Message"
            className="absolute right-1.5 w-9 h-9 rounded-full bg-black hover:bg-neutral-800 disabled:opacity-30 disabled:hover:bg-black flex items-center justify-center text-white transition-all cursor-pointer shadow-xs"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </footer>
  );
}
