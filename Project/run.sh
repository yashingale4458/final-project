#!/bin/sh
set -eu
 
cd "$(dirname "$0")"
 
API_PID=""
 
cleanup() {
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
}
 
trap cleanup EXIT INT TERM
 
echo "Starting FastAPI..."
 
python3 -m uvicorn api.routes:service_application \
    --host 0.0.0.0 \
    --port 8081 \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --root-path /proxy/8081 &
 
API_PID=$!
 
echo "Waiting for FastAPI..."
until curl -fsS http://localhost:8081/openapi.json >/dev/null 2>&1; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "FastAPI stopped unexpectedly." >&2
        exit 1
    fi
    sleep 1
done
 
echo "FastAPI is ready."
echo "Starting Streamlit..."
 
python3 -m streamlit run run_dashboard.py \
    --server.address 0.0.0.0 \
    --server.port 8501
 