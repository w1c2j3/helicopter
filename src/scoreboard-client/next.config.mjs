import os from "node:os";

const apiBase = process.env.SCOREBOARD_API_BASE_URL || "http://127.0.0.1:7860";
const configuredDevOrigins = (process.env.SCOREBOARD_ALLOWED_DEV_ORIGINS || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const localDevOrigins = Object.values(os.networkInterfaces())
  .flat()
  .filter((address) => address?.family === "IPv4" && !address.internal)
  .map((address) => address.address);

/** @type {import('next').NextConfig} */
const nextConfig = {
  allowedDevOrigins: [...new Set([
    "localhost",
    "127.0.0.1",
    ...localDevOrigins,
    ...configuredDevOrigins,
  ])],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiBase}/api/:path*`
      }
    ];
  }
};

export default nextConfig;
