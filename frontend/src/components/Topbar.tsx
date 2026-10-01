"use client";

import React from "react";
import { Menu, Zap, Timer } from "lucide-react";
import { LatencyMetrics } from "@/lib/types";

interface TopbarProps {
  statusText: string;
  metrics: LatencyMetrics;
  onOpenMobileSidebar: () => void;
}

export function Topbar({
  statusText,
  metrics,
  onOpenMobileSidebar,
}: TopbarProps) {
  return (
    <header className="glass-panel shrink-0 flex items-center justify-between px-5 py-3.5 gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onOpenMobileSidebar}
          className="md:hidden text-slate-300 hover:text-white p-1 rounded-md"
          aria-label="Open Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2.5 min-w-0">
          <span className="pulse-beacon shrink-0" />
          <span
            id="system-status-text"
            className="text-xs md:text-sm font-medium text-slate-300 truncate tracking-wide"
          >
            {statusText}
          </span>
        </div>
      </div>

      {/* Latency Metrics Pills */}
      <div className="flex items-center gap-1.5 md:gap-2 shrink-0 font-mono text-[11px] overflow-x-auto">
        <div
          className="px-2 py-1 rounded-md bg-white/4 border border-white/8 text-slate-400"
          title="Speech to Text Latency"
        >
          <span className="text-slate-500 mr-1">STT:</span>
          <span className="text-slate-200">
            {metrics.stt_ms !== undefined ? `${metrics.stt_ms}ms` : "--"}
          </span>
        </div>

        <div
          className="px-2 py-1 rounded-md bg-white/4 border border-white/8 text-slate-400"
          title="LLM Generation Latency"
        >
          <span className="text-slate-500 mr-1">LLM:</span>
          <span className="text-slate-200">
            {metrics.llm_ms !== undefined ? `${metrics.llm_ms}ms` : "--"}
          </span>
        </div>

        <div
          className="px-2 py-1 rounded-md bg-white/4 border border-white/8 text-slate-400"
          title="Text to Speech Latency"
        >
          <span className="text-slate-500 mr-1">TTS:</span>
          <span className="text-slate-200">
            {metrics.tts_ms !== undefined ? `${metrics.tts_ms}ms` : "--"}
          </span>
        </div>

        {/* TTFA Highlight Badge */}
        <div
          className="px-2 py-1 rounded-md bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-semibold flex items-center gap-1 shadow-sm shadow-cyan-500/10"
          title="Time To First Audio (Voice Latency)"
        >
          <Zap className="w-3 h-3 text-cyan-400" />
          <span className="opacity-80">TTFA:</span>
          <span>
            {metrics.ttfa_ms !== undefined ? `${metrics.ttfa_ms}ms` : "--"}
          </span>
        </div>

        {/* Total Latency Badge */}
        <div
          className="px-2 py-1 rounded-md bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 font-semibold hidden sm:flex items-center gap-1 shadow-sm shadow-indigo-500/10"
          title="Total End-to-End Latency"
        >
          <Timer className="w-3 h-3 text-indigo-400" />
          <span className="opacity-80">Total:</span>
          <span>
            {metrics.total_ms !== undefined ? `${metrics.total_ms}ms` : "--"}
          </span>
        </div>
      </div>
    </header>
  );
}
