"use client";

import React from "react";
import { Menu, Zap, Timer, Brain } from "lucide-react";
import { LatencyMetrics } from "@/lib/types";

interface TopbarProps {
  statusText: string;
  metrics: LatencyMetrics;
  onOpenMobileSidebar: () => void;
  ttsDevice?: string;
  onSelectTTSDevice?: (device: string) => void;
  thinkMode?: boolean;
  onToggleThinkMode?: () => void;
  supportsThinking?: boolean;
  reasoningEffort?: string;
}

export function Topbar({
  statusText,
  metrics,
  onOpenMobileSidebar,
  ttsDevice,
  onSelectTTSDevice,
  thinkMode,
  onToggleThinkMode,
  supportsThinking,
  reasoningEffort,
}: TopbarProps) {
  return (
    <header className="glass-panel shrink-0 flex items-center justify-between px-5 py-3.5 gap-4 bg-white border border-neutral-200 shadow-xs">
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onOpenMobileSidebar}
          className="md:hidden text-black hover:text-neutral-700 p-1 rounded-md"
          aria-label="Open Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2.5 min-w-0">
          <span className="pulse-beacon shrink-0" />
          <span
            id="system-status-text"
            className="text-xs md:text-sm font-medium text-black truncate tracking-wide"
          >
            {statusText}
          </span>
        </div>
      </div>

      {/* Latency Metrics Pills */}
      <div className="flex items-center gap-1.5 md:gap-2 shrink-0 font-mono text-[11px] overflow-x-auto">
        <div
          className="px-2 py-1 rounded-md bg-neutral-100 border border-neutral-200 text-neutral-500"
          title="Speech to Text Latency"
        >
          <span className="text-neutral-500 mr-1">STT:</span>
          <span className="text-black font-semibold">
            {metrics.stt_ms !== undefined ? `${metrics.stt_ms}ms` : "--"}
          </span>
        </div>

        <div
          className="px-2 py-1 rounded-md bg-neutral-100 border border-neutral-200 text-neutral-500"
          title="LLM Generation Latency"
        >
          <span className="text-neutral-500 mr-1">LLM:</span>
          <span className="text-black font-semibold">
            {metrics.llm_ms !== undefined ? `${metrics.llm_ms}ms` : "--"}
          </span>
        </div>

        <div
          className="px-2 py-1 rounded-md bg-neutral-100 border border-neutral-200 text-neutral-500"
          title="Text to Speech Latency"
        >
          <span className="text-neutral-500 mr-1">TTS:</span>
          <span className="text-black font-semibold">
            {metrics.tts_ms !== undefined ? `${metrics.tts_ms}ms` : "--"}
          </span>
        </div>

        {/* TTS Unit Quick Selector */}
        {ttsDevice && (
          <button
            type="button"
            onClick={() => {
              const dev = ttsDevice.toLowerCase();
              const next = dev === "cpu" ? "npu" : dev === "npu" ? "npu_only" : "cpu";
              onSelectTTSDevice?.(next);
            }}
            className="px-2 py-1 rounded-md bg-neutral-100 hover:bg-neutral-200 border border-neutral-300 text-black font-semibold flex items-center gap-1 shadow-2xs transition-colors cursor-pointer"
            title={`Kokoro TTS Processing Unit: ${ttsDevice.toUpperCase()}. Click to cycle (CPU -> NPU Hybrid -> NPU Only).`}
          >
            <span className="text-neutral-500 mr-0.5">UNIT:</span>
            <span className="text-black font-bold uppercase">
              {ttsDevice.toLowerCase() === "npu_only" ? "NPU (FULL)" : ttsDevice}
            </span>
          </button>
        )}

        {/* LLM Think Mode Toggle Button */}
        {onToggleThinkMode && (
          <button
            type="button"
            id="think-mode-topbar-btn"
            onClick={onToggleThinkMode}
            className={`px-2 py-1 rounded-md border font-semibold flex items-center gap-1.5 shadow-2xs transition-all cursor-pointer ${
              thinkMode
                ? "bg-violet-600 text-white border-violet-700 shadow-violet-200"
                : "bg-neutral-100 hover:bg-neutral-200 border-neutral-300 text-neutral-700 hover:text-black"
            }`}
            title={
              supportsThinking
                ? `Think Mode: ${thinkMode ? "ON (" + (reasoningEffort || "medium") + " effort)" : "OFF"}. Click to toggle reasoning tokens.`
                : "Current model does not natively support reasoning. Click to toggle think mode."
            }
          >
            <Brain className={`w-3.5 h-3.5 ${thinkMode ? "text-violet-200 animate-pulse" : "text-neutral-500"}`} />
            <span className="text-[10px] tracking-wider font-bold">
              THINK: {thinkMode ? "ON" : "OFF"}
            </span>
            {supportsThinking && (
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  thinkMode ? "bg-emerald-300" : "bg-neutral-400"
                }`}
                title="Reasoning model detected"
              />
            )}
          </button>
        )}

        {/* TTFA Latency Badge */}
        <div
          className="px-2 py-1 rounded-md bg-neutral-100 border border-neutral-300 text-black font-semibold flex items-center gap-1 shadow-2xs"
          title="Time To First Audio (Voice Latency)"
        >
          <Zap className="w-3 h-3 text-black" />
          <span className="opacity-80">TTFA:</span>
          <span>
            {metrics.ttfa_ms !== undefined ? `${metrics.ttfa_ms}ms` : "--"}
          </span>
        </div>

        {/* Total Latency Badge */}
        <div
          className="px-2 py-1 rounded-md bg-black text-white font-semibold hidden sm:flex items-center gap-1 shadow-2xs"
          title="Total End-to-End Latency"
        >
          <Timer className="w-3 h-3 text-white" />
          <span className="opacity-80">Total:</span>
          <span>
            {metrics.total_ms !== undefined ? `${metrics.total_ms}ms` : "--"}
          </span>
        </div>
      </div>
    </header>
  );
}
