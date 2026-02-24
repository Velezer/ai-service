#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  cp .env.example .env
fi

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
