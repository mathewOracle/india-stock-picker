#!/bin/sh
# Start India Stock Picker. Usage: ./serve.sh [port]   (default 61287)
# Proxy defaults to the corporate sysproxy; override HTTPS_PROXY to change, or set it empty off-network.
cd "$(dirname "$0")" || exit 1
PORT="${1:-61287}"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is busy. Pick another: ./serve.sh 61288" >&2
  exit 1
fi

export HTTPS_PROXY="${HTTPS_PROXY-http://sysproxy.wal-mart.com:8080}"
export HTTP_PROXY="$HTTPS_PROXY"
echo "India Stock Picker -> http://127.0.0.1:$PORT"
exec .venv/bin/uvicorn app.main:app --port "$PORT" --log-level info
