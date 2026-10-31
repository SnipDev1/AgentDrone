#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================================="
echo " AgentDrone - Запуск серверного приложения"
echo " Директория: $PROJECT_DIR"
echo "================================================================="

if [ ! -d ".venv" ]; then
    echo "[1/3] Создание виртуального окружения .venv..."
    python3 -m venv .venv
fi

echo "[2/3] Активация окружения..."
source .venv/bin/activate

echo "[3/3] Запуск Streamlit на 127.0.0.1:8501..."
echo "Приложение доступно извне через Caddy reverse-proxy:"
echo "-> http://89.189.177.216/"
echo "================================================================="

# Streamlit запуск с отключением CORS/XSRF для корректной работы через reverse-proxy
exec streamlit run src/ui/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
