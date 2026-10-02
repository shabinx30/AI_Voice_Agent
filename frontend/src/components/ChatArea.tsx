"use client";

import React, { useEffect, useRef } from "react";
import { Bot, User, Play, Volume2, Download } from "lucide-react";
import { ChatMessage } from "@/lib/types";

interface ChatAreaProps {
  messages: ChatMessage[];
  streamingTokenBuffer: string;
  isStreaming: boolean;
  onReplayAudio: (audioBase64: string) => void;
}

export function ChatArea({
  messages,
  streamingTokenBuffer,
  isStreaming,
  onReplayAudio,
}: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages or stream updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingTokenBuffer]);

  const handleDownloadWav = (base64Audio: string, filename = "speech.wav") => {
    const link = document.createElement("a");
    link.href = `data:audio/wav;base64,${base64Audio}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section
      ref={scrollRef}
      id="chat-messages"
      className="glass-panel flex-1 min-h-0 overflow-y-auto p-4 md:p-6 flex flex-col gap-4 scroll-smooth bg-white border border-neutral-200 shadow-xs"
    >
      {/* Empty State */}
      {messages.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-neutral-400">
          <div className="w-12 h-12 rounded-2xl bg-neutral-100 flex items-center justify-center text-neutral-500 mb-3 border border-neutral-200">
            <Volume2 className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-semibold text-black mb-1">
            NexusVoice Assistant & TTS Engine
          </h3>
          <p className="text-xs max-w-sm text-neutral-500">
            Speak through your microphone, chat with the AI assistant, or switch to Direct TTS to synthesize speech with Kokoro-82M.
          </p>
        </div>
      )}

      {/* Render Chat Messages */}
      {messages.map((msg, index) => {
        const isUser = msg.role === "user";
        const isLastAssistant =
          !isUser && index === messages.length - 1 && isStreaming;
        const isTTS = Boolean(msg.isTTSOnly);

        return (
          <div
            key={msg.id}
            className={`flex items-start gap-3 max-w-[85%] ${
              isUser ? "self-end flex-row-reverse" : "self-start"
            } animate-in fade-in slide-in-from-bottom-2 duration-200`}
          >
            {/* Avatar */}
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 shadow-xs ${
                isUser
                  ? "bg-neutral-200 text-black border border-neutral-300"
                  : isTTS
                  ? "bg-neutral-900 text-white"
                  : "bg-black text-white"
              }`}
            >
              {isUser ? (
                <User className="w-4 h-4" />
              ) : isTTS ? (
                <Volume2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
            </div>

            {/* Bubble */}
            <div
              className={`rounded-2xl p-4 transition-all duration-200 ${
                isUser
                  ? "bg-neutral-100 border border-neutral-300 text-black shadow-xs"
                  : isTTS
                  ? "bg-neutral-50 border border-neutral-300 text-black shadow-xs"
                  : "bg-white border border-neutral-200 text-black shadow-xs"
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <div className="text-[11px] font-bold uppercase tracking-wider text-neutral-500 flex items-center gap-1.5">
                  <span>{isUser ? "You" : isTTS ? "Direct TTS" : "NexusVoice"}</span>
                  {isTTS && !isUser && (
                    <span className="px-1.5 py-0.2 rounded text-[10px] bg-neutral-200 text-neutral-700 font-mono lowercase">
                      kokoro-82m
                    </span>
                  )}
                </div>
                {msg.speaker && !isUser && (
                  <span className="text-[10px] text-neutral-400 font-mono">
                    {msg.speaker}
                  </span>
                )}
              </div>

              <div className="text-sm leading-relaxed whitespace-pre-wrap wrap-break-word text-black">
                {msg.text ? (
                  msg.text
                ) : isLastAssistant ? (
                  streamingTokenBuffer ? (
                    <span>
                      {streamingTokenBuffer}
                      <span className="stream-caret" />
                    </span>
                  ) : (
                    <span className="text-neutral-500 italic flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-black animate-ping inline-block" />
                      {isTTS ? "Synthesizing speech..." : "Thinking..."}
                    </span>
                  )
                ) : (
                  <span className="text-neutral-500 italic">...</span>
                )}
              </div>

              {/* Audio Controls for Completed Speech Output */}
              {!isUser && msg.audioBase64 && !isLastAssistant && (
                <div className="mt-3 pt-2.5 border-t border-neutral-200 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => onReplayAudio(msg.audioBase64!)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neutral-100 hover:bg-neutral-200 border border-neutral-300 text-black text-xs font-medium transition-colors cursor-pointer"
                    >
                      <Play className="w-3.5 h-3.5 fill-current text-black" />
                      <span>Play Audio</span>
                    </button>

                    <button
                      onClick={() =>
                        handleDownloadWav(
                          msg.audioBase64!,
                          `tts_${msg.speaker || "speech"}_${msg.id}.wav`
                        )
                      }
                      title="Download speech audio file (WAV)"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-neutral-100 hover:bg-neutral-200 border border-neutral-300 text-neutral-700 hover:text-black text-xs font-medium transition-colors cursor-pointer"
                    >
                      <Download className="w-3.5 h-3.5 text-neutral-700" />
                      <span>WAV</span>
                    </button>
                  </div>

                  <span className="text-[11px] text-neutral-500 flex items-center gap-1 font-mono">
                    <Volume2 className="w-3 h-3 text-neutral-400" />
                    24kHz
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </section>
  );
}
