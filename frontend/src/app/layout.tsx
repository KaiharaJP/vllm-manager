import type { Metadata } from "next";
import "@/app/globals.css";

export const metadata: Metadata = {
  title: "vLLM Manager",
  description: "vLLM サーバー管理ダッシュボード",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body className="min-h-screen bg-bg-primary">
        {children}
      </body>
    </html>
  );
}
