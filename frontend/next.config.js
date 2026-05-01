/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    NEXT_PUBLIC_LITELLM_URL: process.env.NEXT_PUBLIC_LITELLM_URL || "http://localhost:4000",
  },
};

export default nextConfig;
