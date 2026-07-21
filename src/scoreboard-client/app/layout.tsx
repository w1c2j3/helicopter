import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RWKV Skills · Helicopter",
  description: "RWKV evaluation dashboard and scheduler control panel"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
