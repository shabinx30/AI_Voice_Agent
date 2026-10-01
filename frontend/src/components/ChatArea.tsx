"use client";

import React, { useEffect, useRef } from "react";
import { Bot, User, Play, Volume2, Sparkles } from "lucide-react";
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

  return (
    <section
      ref={scrollRef}
      id="chat-messages"
      className="glass-panel flex-1 min-h-0 overflow-y-auto p-4 md:p-6 flex flex-col gap-4 scroll-smooth bg-white border border-neutral-200 shadow-xs"
    >
      {/* Welcome Message Card */}
      <div className="flex items-start gap-3.5 max-w-[85%] self-start animate-in fade-in slide-in-from-bottom-2 duration-300">
        <div className="w-9 h-9 rounded-xl bg-black flex items-center justify-center text-white shrink-0 shadow-xs">
          <Bot className="w-5 h-5" />
        </div>
        <div className="rounded-2xl p-4 bg-neutral-50 border border-neutral-200 shadow-xs text-black">
          <div className="text-[11px] font-bold uppercase tracking-wider text-neutral-500 mb-1.5 flex items-center gap-1.5">
            <Sparkles className="w-3 h-3 text-black" />
            NexusVoice Assistant
          </div>
          <div className="text-sm leading-relaxed text-black">
            Hello! I am your personal voice assistant. Speak by clicking the
            microphone button or type below. Your voice is transcribed locally
            via <strong className="text-black font-semibold">OpenVINO Whisper Base INT8</strong>,
            processed by your <strong className="text-black font-semibold">LM Studio LLM</strong>,
            and spoken back with <strong className="text-black font-semibold">Kokoro-82M TTS</strong>!
          </div>
        </div>
      </div>

      {/* Render Chat Messages */}
      {messages.map((msg, index) => {
        const isUser = msg.role === "user";
        const isLastAssistant =
          !isUser && index === messages.length - 1 && isStreaming;

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
                  : "bg-black text-white"
              }`}
            >
              {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            {/* Bubble */}
            <div
              className={`rounded-2xl p-4 transition-all duration-200 ${
                isUser
                  ? "bg-neutral-100 border border-neutral-300 text-black shadow-xs"
                  : "bg-white border border-neutral-200 text-black shadow-xs"
              }`}
            >
              <div className="text-[11px] font-bold uppercase tracking-wider text-neutral-500 mb-1">
                {isUser ? "You" : "NexusVoice"}
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
                      Thinking...
                    </span>
                  )
                ) : (
                  <span className="text-neutral-500 italic">...</span>
                )}
              </div>

              {/* Audio Controls for Completed Assistant Reply */}
              {!isUser && msg.audioBase64 && !isLastAssistant && (
                <div className="mt-3 pt-2.5 border-t border-neutral-200 flex items-center gap-2">
                  <button
                    onClick={() => onReplayAudio(msg.audioBase64!)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neutral-100 hover:bg-neutral-200 border border-neutral-300 text-black text-xs font-medium transition-colors cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5 fill-current text-black" />
                    <span>Replay Full Audio</span>
                  </button>
                  <span className="text-[11px] text-neutral-500 flex items-center gap-1">
                    <Volume2 className="w-3 h-3 text-neutral-500" />
                    Kokoro 24kHz
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
