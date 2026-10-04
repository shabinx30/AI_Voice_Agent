"use client";

import React, { useEffect, useRef, useState } from "react";
import { Bot, User, Play, Pause, Volume2, Download, Brain, ChevronDown, ChevronRight, Square } from "lucide-react";
import { ChatMessage } from "@/lib/types";
import { MarkdownContent } from "@/components/MarkdownContent";

interface ChatAreaProps {
  messages: ChatMessage[];
  streamingTokenBuffer: string;
  streamingThoughtBuffer?: string;
  isStreaming: boolean;
  isThinking?: boolean;
  isAudioPlaying?: boolean;
  isAudioPaused?: boolean;
  onToggleAudioPlayPause?: () => void;
  onReplayAudio: (audioBase64: string, messageId?: string) => void;
  activeReplayId?: string | null;
  onCancelGeneration?: () => void;
}

function ThinkingAccordion({
  thought,
  isLive,
  defaultOpen,
}: {
  thought: string;
  isLive?: boolean;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState<boolean>(defaultOpen ?? Boolean(isLive));

  // Automatically open when reasoning is actively streaming
  useEffect(() => {
    if (isLive) {
      setIsOpen(true);
    }
  }, [isLive]);

  const wordCount = thought.trim() ? thought.trim().split(/\s+/).length : 0;

  return (
    <div className="mb-3 rounded-xl border border-violet-200 bg-violet-50/40 overflow-hidden shadow-2xs transition-all">
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full flex items-center justify-between px-3.5 py-2 text-left bg-violet-100/60 hover:bg-violet-100/90 transition-colors cursor-pointer select-none"
      >
        <div className="flex items-center gap-2 min-w-0">
          <Brain className={`w-3.5 h-3.5 text-violet-600 shrink-0 ${isLive ? "animate-pulse" : ""}`} />
          <span className="text-xs font-semibold text-violet-950 tracking-tight">
            Thinking Process
          </span>
          {isLive ? (
            <span className="inline-flex items-center gap-1 text-[10px] font-mono text-violet-700 bg-violet-200/80 px-2 py-0.5 rounded-full font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-violet-600 animate-ping inline-block" />
              Reasoning...
            </span>
          ) : (
            <span className="text-[10px] font-mono text-neutral-500 bg-white/80 px-1.5 py-0.5 rounded border border-violet-200/60">
              {wordCount} words
            </span>
          )}
        </div>
        <div className="text-violet-600 shrink-0 ml-2">
          {isOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </div>
      </button>

      {isOpen && (
        <div className="p-3.5 text-xs font-mono text-neutral-700 bg-white/90 border-t border-violet-200/70 leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto select-text">
          {thought}
          {isLive && <span className="stream-caret ml-0.5" />}
        </div>
      )}
    </div>
  );
}

export function ChatArea({
  messages,
  streamingTokenBuffer,
  streamingThoughtBuffer = "",
  isStreaming,
  isThinking = false,
  isAudioPlaying = false,
  isAudioPaused = false,
  onToggleAudioPlayPause,
  onReplayAudio,
  activeReplayId,
  onCancelGeneration,
}: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages or stream updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingTokenBuffer, streamingThoughtBuffer]);

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

        // Determine thought content for this message
        const messageThought =
          msg.thought ||
          (isLastAssistant && streamingThoughtBuffer ? streamingThoughtBuffer : "");
        const isMessageCurrentlyThinking =
          isLastAssistant && Boolean(isThinking || (streamingThoughtBuffer && !streamingTokenBuffer));

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
                  {msg.thought && !isUser && (
                    <span className="px-1.5 py-0.2 rounded text-[10px] bg-violet-100 text-violet-700 font-mono flex items-center gap-1">
                      <Brain className="w-2.5 h-2.5" />
                      thought
                    </span>
                  )}
                  {msg.isCancelled && !isUser && (
                    <span className="px-1.5 py-0.2 rounded text-[10px] bg-red-50 text-red-600 border border-red-200 font-mono">
                      stopped
                    </span>
                  )}
                </div>
                {msg.speaker && !isUser && (
                  <span className="text-[10px] text-neutral-400 font-mono">
                    {msg.speaker}
                  </span>
                )}
              </div>

              {/* Thinking Accordion (Shown if message has reasoning thoughts or is actively reasoning) */}
              {!isUser && !isTTS && messageThought && (
                <ThinkingAccordion
                  thought={messageThought}
                  isLive={isMessageCurrentlyThinking}
                  defaultOpen={isMessageCurrentlyThinking}
                />
              )}

              <div className="text-sm leading-relaxed wrap-break-word text-black">
                {msg.text ? (
                  isUser ? (
                    <div className="whitespace-pre-wrap">{msg.text}</div>
                  ) : (
                    <MarkdownContent content={msg.text} />
                  )
                ) : msg.isCancelled ? (
                  <span className="text-neutral-500 italic text-xs">
                    (Generation stopped by user)
                  </span>
                ) : isLastAssistant ? (
                  streamingTokenBuffer ? (
                    <MarkdownContent
                      content={streamingTokenBuffer}
                      isStreaming={true}
                    />
                  ) : isMessageCurrentlyThinking ? (
                    <span className="text-violet-600/80 italic text-xs flex items-center gap-1.5 font-medium">
                      <Brain className="w-3.5 h-3.5 text-violet-500 animate-pulse" />
                      Synthesizing reasoning tokens before formulating answer...
                    </span>
                  ) : (
                    <span className="text-neutral-500 italic flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-black animate-ping inline-block" />
                      {isTTS ? "Synthesizing speech..." : "Generating response..."}
                    </span>
                  )
                ) : (
                  <span className="text-neutral-500 italic">...</span>
                )}
              </div>

              {/* Generating / Active Audio Playback Controls */}
              {isLastAssistant && (isStreaming || isAudioPlaying || isAudioPaused) && (
                <div className="mt-3 pt-2.5 border-t border-neutral-100 flex flex-wrap items-center justify-between gap-2 animate-in fade-in duration-200">
                  <div className="flex items-center gap-2">
                    {onToggleAudioPlayPause && (isAudioPlaying || isAudioPaused) && (
                      <button
                        type="button"
                        id="chat-toggle-audio-btn"
                        onClick={onToggleAudioPlayPause}
                        title={isAudioPaused ? "Resume audio playback (Space)" : "Pause audio playback (Space)"}
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer shadow-2xs ${
                          isAudioPaused
                            ? "bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300"
                            : "bg-black hover:bg-neutral-800 text-white border border-black"
                        }`}
                      >
                        {isAudioPaused ? (
                          <>
                            <Play className="w-3.5 h-3.5 fill-current" />
                            <span>Resume Audio</span>
                          </>
                        ) : (
                          <>
                            <Pause className="w-3.5 h-3.5 fill-current" />
                            <span>Pause Audio</span>
                          </>
                        )}
                      </button>
                    )}

                    {/* Animated sound wave bars or paused badge */}
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-neutral-100 border border-neutral-200 text-[11px] font-mono select-none">
                      {isAudioPaused ? (
                        <span className="text-amber-800 flex items-center gap-1.5 font-semibold">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                          Audio Paused
                        </span>
                      ) : isAudioPlaying ? (
                        <div className="flex items-center gap-1.5 text-neutral-800 font-medium">
                          <div className="flex items-center gap-0.5 h-3">
                            <span className="w-0.5 h-3 bg-black rounded-full animate-soundwave-1 inline-block" />
                            <span className="w-0.5 h-3 bg-black rounded-full animate-soundwave-2 inline-block" />
                            <span className="w-0.5 h-3 bg-black rounded-full animate-soundwave-3 inline-block" />
                            <span className="w-0.5 h-3 bg-black rounded-full animate-soundwave-4 inline-block" />
                          </div>
                          <span>Speaking speech...</span>
                        </div>
                      ) : (
                        <span className="text-neutral-500 flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-neutral-400 animate-ping inline-block" />
                          Generating speech...
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Stop Generation Button if LLM is still streaming */}
                  {isStreaming && onCancelGeneration && (
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        id="chat-cancel-generation-btn"
                        onClick={onCancelGeneration}
                        title="Stop generation (Esc)"
                        className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-neutral-100 hover:bg-red-50 hover:text-red-700 hover:border-red-200 border border-neutral-200 text-neutral-600 text-xs font-medium transition-all cursor-pointer shadow-2xs group"
                      >
                        <Square className="w-3 h-3 fill-current text-neutral-400 group-hover:text-red-600 transition-colors" />
                        <span>Stop</span>
                      </button>
                      <span className="text-[10px] text-neutral-400 font-mono hidden sm:inline">
                        Esc
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Audio Controls for Completed Speech Output */}
              {!isUser && msg.audioBase64 && (!isLastAssistant || (!isStreaming && !isAudioPlaying && !isAudioPaused)) && (
                <div className="mt-3 pt-2.5 border-t border-neutral-200 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => onReplayAudio(msg.audioBase64!, msg.id)}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer ${
                        activeReplayId === msg.id && isAudioPaused
                          ? "bg-amber-100 hover:bg-amber-200 border-amber-300 text-amber-950 font-semibold"
                          : activeReplayId === msg.id && isAudioPlaying
                          ? "bg-black hover:bg-neutral-800 border-black text-white font-semibold"
                          : "bg-neutral-100 hover:bg-neutral-200 border-neutral-300 text-black"
                      }`}
                    >
                      {activeReplayId === msg.id && isAudioPlaying && !isAudioPaused ? (
                        <>
                          <Pause className="w-3.5 h-3.5 fill-current" />
                          <span>Pause Audio</span>
                        </>
                      ) : activeReplayId === msg.id && isAudioPaused ? (
                        <>
                          <Play className="w-3.5 h-3.5 fill-current" />
                          <span>Resume Audio</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5 fill-current" />
                          <span>Play Audio</span>
                        </>
                      )}
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
