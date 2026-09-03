#!/bin/bash
set -e

echo "============================================================"
echo "STARTING PLANT DISEASE DETECTION APPLICATION"
echo "============================================================"

export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="${PORT:-10000}"
export GRADIO_ANALYTICS_ENABLED="False"

echo "Host: ${GRADIO_SERVER_NAME}"
echo "Port: ${GRADIO_SERVER_PORT}"

exec python app.py
