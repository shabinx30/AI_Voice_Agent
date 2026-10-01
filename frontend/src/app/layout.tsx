import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const plusJakarta = Plus_Jakarta_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

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
    <html
      lang="en"
      className={`${plusJakarta.variable} ${jetbrainsMono.variable} dark`}
    >
      <body className="font-sans antialiased bg-[#07090e] text-[#f8fafc] selection:bg-indigo-500/30 selection:text-white">
        {children}
      </body>
    </html>
  );
}
