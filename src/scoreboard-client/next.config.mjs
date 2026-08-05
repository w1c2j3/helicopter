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
  distDir: process.env.SCOREBOARD_NEXT_DIST_DIR || ".next",
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
  },
  async headers() {
    const contentSecurityPolicy = [
      "default-src 'self'",
      "base-uri 'self'",
      "form-action 'self'",
      "frame-ancestors 'none'",
      "object-src 'none'",
      "script-src 'self' 'unsafe-inline'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "connect-src 'self'",
      "upgrade-insecure-requests",
    ].join("; ");
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Content-Security-Policy", value: contentSecurityPolicy },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
          { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
        ],
      },
    ];
  }
};

export default nextConfig;
