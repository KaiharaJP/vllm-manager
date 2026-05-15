/** @type {import('next').NextConfig} */
const backendProxyTarget =
  process.env.BACKEND_INTERNAL_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

const nextConfig = {
  output: "standalone",
  env: {
    // 未設定時は同一オリジン経由（rewrites）を使う
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "",
    NEXT_PUBLIC_LITELLM_URL: process.env.NEXT_PUBLIC_LITELLM_URL || "http://localhost:4000",
  },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${backendProxyTarget}/api/:path*` },
      { source: "/ws/:path*", destination: `${backendProxyTarget}/ws/:path*` },
      { source: "/v1/:path*", destination: `${backendProxyTarget}/v1/:path*` },
      { source: "/docs", destination: `${backendProxyTarget}/docs` },
      { source: "/openapi.json", destination: `${backendProxyTarget}/openapi.json` },
    ];
  },
};

module.exports = nextConfig;
