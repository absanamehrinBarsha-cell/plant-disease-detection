#!/bin/bash
set -e

echo "============================================================"
echo "STARTING PLANT DISEASE DETECTION APPLICATION"
echo "============================================================"

PORT="${PORT:-10000}"

echo "Using PORT: ${PORT}"

exec python app.py
