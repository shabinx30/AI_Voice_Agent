import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NexusVoice • AI Personal Assistant",
  description:
    "Next-generation voice assistant powered by OpenVINO Whisper Base INT8 STT, LM Studio LLM inference, and Kokoro-82M TTS speech generation.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-white text-black selection:bg-neutral-200 selection:text-black">
        {children}
      </body>
    </html>
  );
}
