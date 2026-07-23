const apiBase = process.env.SCOREBOARD_API_BASE_URL || "http://127.0.0.1:7860";

/** @type {import('next').NextConfig} */
const nextConfig = {
  allowedDevOrigins: ["172.21.122.93"],
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
