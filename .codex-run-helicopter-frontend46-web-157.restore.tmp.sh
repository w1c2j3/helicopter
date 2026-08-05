#!/usr/bin/env bash
set -euo pipefail

export SCOREBOARD_API_BASE_URL=http://127.0.0.1:7862
export NEXT_TELEMETRY_DISABLED=1
export PATH=/home/rwkv/.local/share/pi-node/node-v22.22.3-linux-x64/bin:/usr/local/bin:/usr/bin:/bin

cd /home/rwkv/chase/helicopter/scoreboard-client
exec npm run dev -- --hostname 0.0.0.0 --port 3010
