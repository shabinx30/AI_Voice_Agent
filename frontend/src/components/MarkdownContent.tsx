"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { Copy, Check } from "lucide-react";

interface MarkdownContentProps {
  content: string;
  className?: string;
  isStreaming?: boolean;
}

function CodeBlock({
  language,
  value,
}: {
  language: string;
  value: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code to clipboard", err);
    }
  };

  return (
    <div className="relative my-3 rounded-xl border border-neutral-800 bg-neutral-950 overflow-hidden shadow-xs text-left">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-neutral-900 border-b border-neutral-800 text-[11px] font-mono text-neutral-400 select-none">
        <span className="uppercase tracking-wider text-[10px] font-semibold text-neutral-300">
          {language || "code"}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-0.5 rounded hover:bg-neutral-800 hover:text-white transition-colors cursor-pointer text-xs"
          title="Copy code to clipboard"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400 font-sans text-[11px] font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span className="font-sans text-[11px]">Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 text-xs font-mono text-neutral-200 overflow-x-auto leading-relaxed">
        <code>{value}</code>
      </pre>
    </div>
  );
}

export function MarkdownContent({
  content,
  className = "",
  isStreaming = false,
}: MarkdownContentProps) {
  return (
    <div
      className={`markdown-body text-sm leading-relaxed text-black wrap-break-word ${
        isStreaming ? "markdown-streaming" : ""
      } ${className}`}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 className="text-xl font-bold tracking-tight text-neutral-950 mt-4 mb-2 first:mt-0 pb-1 border-b border-neutral-100">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-bold tracking-tight text-neutral-950 mt-3.5 mb-2 first:mt-0 pb-0.5 border-b border-neutral-100">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold tracking-tight text-neutral-950 mt-3 mb-1.5 first:mt-0">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-sm font-semibold text-neutral-950 mt-2 mb-1 first:mt-0">
              {children}
            </h4>
          ),
          h5: ({ children }) => (
            <h5 className="text-xs font-semibold uppercase tracking-wider text-neutral-600 mt-2 mb-1 first:mt-0">
              {children}
            </h5>
          ),
          h6: ({ children }) => (
            <h6 className="text-xs font-medium text-neutral-500 mt-2 mb-1 first:mt-0">
              {children}
            </h6>
          ),
          // Paragraphs
          p: ({ children }) => (
            <p className="mb-2.5 last:mb-0 leading-relaxed text-neutral-900">
              {children}
            </p>
          ),
          // Strong & Emphasis & Strikethrough
          strong: ({ children }) => (
            <strong className="font-bold text-neutral-950">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-neutral-800">
              {children}
            </em>
          ),
          del: ({ children }) => (
            <del className="line-through text-neutral-400">
              {children}
            </del>
          ),
          // Lists
          ul: ({ children }) => (
            <ul className="list-disc list-outside pl-5 mb-2.5 space-y-1 text-neutral-900">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside pl-5 mb-2.5 space-y-1.5 text-neutral-900 font-normal">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="pl-1 leading-relaxed text-neutral-900">
              {children}
            </li>
          ),
          // Task List Checkboxes
          input: ({ type, checked, ...props }) => {
            if (type === "checkbox") {
              return (
                <input
                  type="checkbox"
                  checked={checked}
                  readOnly
                  className="mr-2 rounded border-neutral-300 text-black focus:ring-0 cursor-default align-middle"
                  {...props}
                />
              );
            }
            return <input type={type} {...props} />;
          },
          // Blockquote
          blockquote: ({ children }) => (
            <blockquote className="border-l-3 border-neutral-400 pl-3.5 py-1.5 my-3 text-neutral-700 italic bg-neutral-50/80 rounded-r-lg">
              {children}
            </blockquote>
          ),
          // Pre & Code
          pre: ({ children }) => {
            return <>{children}</>;
          },
          code: ({ className: codeClassName, children, ...props }) => {
            const match = /language-(\w+)/.exec(codeClassName || "");
            const codeString = String(children).replace(/\n$/, "");
            const isMultiLine = codeString.includes("\n");

            if (match || isMultiLine) {
              return (
                <CodeBlock
                  language={match ? match[1] : ""}
                  value={codeString}
                />
              );
            }

            return (
              <code
                className="px-1.5 py-0.5 mx-0.5 rounded bg-neutral-100 border border-neutral-200 text-neutral-900 font-mono text-[12.5px] font-medium"
                {...props}
              >
                {children}
              </code>
            );
          },
          // Table
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-xl border border-neutral-200 shadow-2xs">
              <table className="min-w-full divide-y divide-neutral-200 text-xs text-left">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-neutral-50 font-semibold">{children}</thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-neutral-100 bg-white">{children}</tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-neutral-50/60 transition-colors">{children}</tr>
          ),
          th: ({ children }) => (
            <th className="px-3.5 py-2.5 font-semibold text-neutral-900 border-b border-neutral-200">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3.5 py-2 text-neutral-800">
              {children}
            </td>
          ),
          // Horizontal Rule
          hr: () => <hr className="my-3.5 border-neutral-200" />,
          // Links
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline decoration-blue-300 underline-offset-2 hover:text-blue-800 hover:decoration-blue-600 transition-colors font-medium inline-flex items-center gap-1"
            >
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
      {isStreaming && (
        <span className="stream-caret ml-0.5 inline-block align-baseline" />
      )}
    </div>
  );
}
