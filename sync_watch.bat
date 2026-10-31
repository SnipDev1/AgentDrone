@echo off
title AgentDrone - Live Auto-Sync to Server
cd /d "%~dp0"
call sync_to_server.bat --watch
pause
