#!/bin/bash
# ==============================================================================
# AgentDrone - Настройка разделения маршрутов /sber и /dev на сервере
# Хост: 89.189.177.216
# /sber -> http://127.0.0.1:8501 (Ветка со Сбер-дизайном)
# /dev  -> http://127.0.0.1:8502 (Ветка разработчиков)
# ==============================================================================
set -e

echo "[1/6] Подготовка репозитория разработчиков (/home/snip/AgentDrone-dev)..."
if [ ! -d "/home/snip/AgentDrone-dev" ]; then
    git clone https://github.com/SnipDev1/AgentDrone.git /home/snip/AgentDrone-dev
fi

cd /home/snip/AgentDrone-dev
git fetch origin
# Переключаем на ветку разработчиков (andrey)
if git show-ref --verify --quiet refs/remotes/origin/andrey; then
    git checkout andrey || git checkout -b andrey origin/andrey
fi

# Создаем venv для dev если не создан
if [ ! -d "/home/snip/AgentDrone-dev/.venv" ]; then
    echo "Создание .venv для AgentDrone-dev..."
    python3 -m venv /home/snip/AgentDrone-dev/.venv
    /home/snip/AgentDrone-dev/.venv/bin/pip install --upgrade pip
    /home/snip/AgentDrone-dev/.venv/bin/pip install -r /home/snip/AgentDrone-dev/requirements.txt
fi

echo "[2/6] Установка systemd сервисов..."
sudo cp /home/snip/AgentDrone/deploy/agentdrone-sber.service /etc/systemd/system/
sudo cp /home/snip/AgentDrone/deploy/agentdrone-dev.service /etc/systemd/system/

echo "[3/6] Настройка Caddy reverse proxy..."
if [ -f "/etc/caddy/Caddyfile" ]; then
    sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak_$(date +%s)
fi
sudo cp /home/snip/AgentDrone/deploy/Caddyfile /etc/caddy/Caddyfile

echo "[4/6] Перезапуск демонов..."
sudo systemctl daemon-reload
# Отключаем старый единый сервис если был активен
sudo systemctl stop agentdrone 2>/dev/null || true
sudo systemctl disable agentdrone 2>/dev/null || true

# Включаем и запускаем sber и dev
sudo systemctl enable agentdrone-sber agentdrone-dev
sudo systemctl restart agentdrone-sber
sudo systemctl restart agentdrone-dev
sudo systemctl reload caddy || sudo systemctl restart caddy

echo "[5/6] Проверка статуса сервисов..."
sudo systemctl status agentdrone-sber --no-pager | head -n 10
sudo systemctl status agentdrone-dev --no-pager | head -n 10

echo "[6/6] Готово! Проверьте адреса:"
echo " -> Ветка Сбера (Роль №3): http://89.189.177.216/sber"
echo " -> Ветка разработчиков:   http://89.189.177.216/dev"
