"use client";

import React from "react";
import { Mic, RefreshCw, Cpu, Sparkles, Bot, Eject } from "lucide-react";
import { HealthStatus, LMStudioModelInfo } from "@/lib/types";

interface SidebarProps {
  health: HealthStatus | null;
  availableModels: LMStudioModelInfo[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
  isLoadingModels: boolean;
  onRefreshModels: () => void;
  onEjectModel?: (modelId?: string) => void;
  isEjectingModel?: boolean;
  selectedSpeaker: string;
  onSelectSpeaker: (speaker: string) => void;
  playHostAudio: boolean;
  onTogglePlayHostAudio: (val: boolean) => void;
  isCheckingHealth: boolean;
  onRefreshHealth: () => void;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

export function Sidebar({
  health,
  availableModels,
  selectedModel,
  onSelectModel,
  isLoadingModels,
  onRefreshModels,
  onEjectModel,
  isEjectingModel,
  selectedSpeaker,
  onSelectSpeaker,
  playHostAudio,
  onTogglePlayHostAudio,
  isCheckingHealth,
  onRefreshHealth,
  isOpenMobile,
  onCloseMobile,
}: SidebarProps) {
  // Helper to format speaker names nicely
  const formatSpeakerName = (spk: string) => {
    const parts = spk.split("_");
    if (parts.length === 2) {
      const nat = parts[0].startsWith("a") ? "US" : "UK";
      const gen = parts[0].endsWith("f") ? "Female" : "Male";
      const name = parts[1].charAt(0).toUpperCase() + parts[1].slice(1);
      return `${name} (${nat} ${gen})`;
    }
    return spk.charAt(0).toUpperCase() + spk.slice(1);
  };

  const defaultSpeakers = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_aoede",
    "af_kore",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "am_echo",
    "am_eric",
    "am_liam",
    "am_onyx",
    "am_puck",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_daniel",
  ];

  const speakersList =
    health?.tts_speakers && health.tts_speakers.length > 0
      ? health.tts_speakers
      : defaultSpeakers;

  const activeModelInfo = availableModels?.find((m) => m.id === selectedModel);

  return (
    <aside
      className={`glass-panel flex flex-col h-full w-77.5 shrink-0 p-5 gap-5 overflow-y-auto transition-all duration-300 z-30 ${
        isOpenMobile
          ? "fixed inset-y-4 left-4 max-w-[calc(100vw-32px)] shadow-2xl"
          : "hidden md:flex"
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between pb-1 border-b border-white/5">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-linear-to-br from-indigo-500 via-indigo-600 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30">
            <Mic className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              NexusVoice
            </h1>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
              Personal Assistant
            </span>
          </div>
        </div>

        {isOpenMobile && (
          <button
            onClick={onCloseMobile}
            className="md:hidden text-slate-400 hover:text-white p-1 rounded-md"
            aria-label="Close Sidebar"
          >
            ✕
          </button>
        )}
      </div>

      {/* Hardware & Pipelines Section */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            Pipelines & Hardware
          </h2>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
            LOCAL
          </span>
        </div>

        {/* STT Card */}
        <div className="p-3 rounded-xl bg-white/3 border border-white/6 hover:bg-white/6 hover:border-white/12 transition-all duration-200">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <span className="status-indicator online" />
              <span className="text-xs font-semibold text-slate-200">
                OpenVINO STT
              </span>
            </div>
            <span className="text-[10px] font-mono text-indigo-300 bg-indigo-500/10 px-1 rounded">
              INT8
            </span>
          </div>
          <div className="text-[12px] font-mono text-slate-400 truncate">
            {health?.stt_model
              ? health.stt_model.split("/").pop()
              : "whisper-base-int8-ov"}
          </div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
            <span>Device:</span>
            <strong className="text-slate-300 font-mono">
              {health?.stt_device ? `${health.stt_device} (Arc GPU)` : "NPU+GPU"}
            </strong>
          </div>
        </div>

        {/* LLM Card */}
        <div className="p-3 rounded-xl bg-white/3 border border-white/6 hover:bg-white/6 hover:border-white/12 transition-all duration-200">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <span
                className={`status-indicator ${
                  health?.lm_studio_connected ? "online" : "offline"
                }`}
              />
              <span className="text-xs font-semibold text-slate-200">
                LM Studio LLM
              </span>
            </div>
            <span className="text-[10px] font-mono text-cyan-300 bg-cyan-500/10 px-1 rounded">
              Local LLM
            </span>
          </div>
          <div className="text-[12px] font-mono text-slate-300 truncate font-medium">
            {health?.lm_studio_connected
              ? activeModelInfo?.name || selectedModel || health.lm_studio_model || "qwen2.5-0.5b-instruct"
              : "Offline (Check LM Studio)"}
          </div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
            <span>Status:</span>
            <strong className="text-slate-300 font-mono text-[10px]">
              {activeModelInfo?.loaded ? "In Memory (Active)" : (health?.lm_studio_connected ? "Online • Ready" : "Disconnected")}
            </strong>
          </div>
        </div>

        {/* TTS Card */}
        <div className="p-3 rounded-xl bg-white/3 border border-white/6 hover:bg-white/6 hover:border-white/12 transition-all duration-200">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <span className="status-indicator online" />
              <span className="text-xs font-semibold text-slate-200">
                Kokoro-82M TTS
              </span>
            </div>
            <span className="text-[10px] font-mono text-purple-300 bg-purple-500/10 px-1 rounded">
              PyTorch / OV
            </span>
          </div>
          <div className="text-[12px] font-mono text-slate-400 truncate">
            {health?.tts_model ? health.tts_model.split("/").pop() : "Kokoro-82M"}
          </div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-between">
            <span>Sample Rate:</span>
            <strong className="text-slate-300 font-mono">24,000 Hz</strong>
          </div>
        </div>
      </div>

      {/* Model Configuration Section */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Bot className="w-3.5 h-3.5 text-cyan-400" />
            Model Configuration
          </h2>
          <button
            onClick={onRefreshModels}
            disabled={isLoadingModels}
            title="Scan for available models in LM Studio"
            className="p-1 rounded-md text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-colors disabled:opacity-50"
          >
            <RefreshCw
              className={`w-3 h-3 ${isLoadingModels ? "animate-spin text-cyan-400" : ""}`}
            />
          </button>
        </div>

        {/* Model Select */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <label
              htmlFor="model-select"
              className="text-xs text-slate-300 font-medium"
            >
              Active LM Studio Model
            </label>
            <div className="flex items-center gap-1.5">
              {activeModelInfo?.loaded && (
                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                  Loaded
                </span>
              )}
              {onEjectModel && (
                <button
                  type="button"
                  onClick={() => onEjectModel(selectedModel)}
                  disabled={isEjectingModel || !health?.lm_studio_connected}
                  title="Eject model from memory to free VRAM"
                  className="flex items-center gap-1 text-[10px] font-medium text-rose-400 hover:text-rose-300 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 px-1.5 py-0.5 rounded transition-colors disabled:opacity-40"
                >
                  <Eject className={`w-2.5 h-2.5 ${isEjectingModel ? "animate-pulse" : ""}`} />
                  <span>{isEjectingModel ? "Ejecting..." : "Eject"}</span>
                </button>
              )}
            </div>
          </div>
          <div className="relative">
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => onSelectModel(e.target.value)}
              disabled={isLoadingModels || !health?.lm_studio_connected}
              className="w-full appearance-none bg-slate-900/80 border border-white/10 hover:border-indigo-500/40 text-slate-200 text-xs rounded-lg px-3 py-2.5 outline-none focus:border-indigo-500 transition-colors cursor-pointer pr-8 font-sans disabled:opacity-50"
            >
              {availableModels && availableModels.length > 0 ? (
                availableModels.map((m) => (
                  <option key={m.id} value={m.id} className="bg-slate-900 text-white">
                    {m.name || m.id} {m.params ? `(${m.params})` : ""} {m.loaded ? "• Loaded" : ""}
                  </option>
                ))
              ) : (
                <option value={selectedModel || "qwen2.5-0.5b-instruct"} className="bg-slate-900 text-white">
                  {selectedModel || health?.lm_studio_model || "qwen2.5-0.5b-instruct"}
                </option>
              )}
            </select>
            <div className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400">
              ▼
            </div>
          </div>
          {activeModelInfo && (
            <div className="flex items-center justify-between text-[10px] text-slate-400 px-0.5">
              <span>{activeModelInfo.params ? `Params: ${activeModelInfo.params}` : "Local Model"}</span>
              <span>{activeModelInfo.size_formatted ? `Size: ${activeModelInfo.size_formatted}` : ""}</span>
            </div>
          )}
        </div>
      </div>

      {/* Voice Configuration Section */}
      <div className="flex flex-col gap-3">
        <h2 className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          Voice Configuration
        </h2>

        {/* Speaker Select */}
        <div className="flex flex-col gap-1.5">
          <label
            htmlFor="speaker-select"
            className="text-xs text-slate-300 font-medium"
          >
            Voice Persona
          </label>
          <div className="relative">
            <select
              id="speaker-select"
              value={selectedSpeaker}
              onChange={(e) => onSelectSpeaker(e.target.value)}
              className="w-full appearance-none bg-slate-900/80 border border-white/10 hover:border-indigo-500/40 text-slate-200 text-xs rounded-lg px-3 py-2.5 outline-none focus:border-indigo-500 transition-colors cursor-pointer pr-8 font-sans"
            >
              {speakersList.map((spk) => (
                <option key={spk} value={spk} className="bg-slate-900 text-white">
                  {formatSpeakerName(spk)}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400">
              ▼
            </div>
          </div>
        </div>

        {/* Host Speaker Audio Toggle */}
        <div
          className="flex items-center justify-between p-2.5 rounded-lg bg-white/2 border border-white/6 hover:bg-white/4 transition-colors"
          title="Enable to route audio to server host speakers instead of browser"
        >
          <div className="flex flex-col">
            <label
              htmlFor="play-host-audio"
              className="text-xs font-medium text-slate-300 cursor-pointer"
            >
              Host Speaker Output
            </label>
            <span className="text-[10px] text-slate-400">
              Play on host machine hardware
            </span>
          </div>

          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              id="play-host-audio"
              checked={playHostAudio}
              onChange={(e) => onTogglePlayHostAudio(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
          </label>
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="mt-auto pt-2">
        <button
          onClick={onRefreshHealth}
          disabled={isCheckingHealth}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg bg-white/4 border border-white/8 hover:bg-white/8 hover:border-white/15 text-slate-300 text-xs font-semibold transition-all duration-200 disabled:opacity-50"
        >
          <RefreshCw
            className={`w-3.5 h-3.5 text-indigo-400 ${
              isCheckingHealth ? "animate-spin" : ""
            }`}
          />
          <span>{isCheckingHealth ? "Checking..." : "Check Connectivity"}</span>
        </button>
      </div>
    </aside>
  );
}
