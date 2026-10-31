@echo off
chcp 65001 >nul
setlocal

:: ============================================================================
:: AgentDrone - Sync to Remote Server
:: Server: 89.189.177.216 | Port: 28882 | User: snip
:: ============================================================================

set REMOTE_HOST=89.189.177.216
set REMOTE_PORT=28882
set REMOTE_USER=snip
set REMOTE_DIR=/home/snip/AgentDrone
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set SSH_IDENTITY=
if exist "%SCRIPT_DIR%keys\id_ed25519_team" set SSH_IDENTITY=-i "%SCRIPT_DIR%keys\id_ed25519_team"
if exist "%SCRIPT_DIR%keys\id_ed25519_frontend" set SSH_IDENTITY=-i "%SCRIPT_DIR%keys\id_ed25519_frontend"
if exist "%SCRIPT_DIR%keys\id_ed25519_backend" set SSH_IDENTITY=-i "%SCRIPT_DIR%keys\id_ed25519_backend"
if exist "%SCRIPT_DIR%keys\id_ed25519_cv" set SSH_IDENTITY=-i "%SCRIPT_DIR%keys\id_ed25519_cv"

if "%1"=="--watch" goto :watch_mode
if "%1"=="-w" goto :watch_mode
if "%1"=="--test" goto :run_tests
if "%1"=="--run" goto :run_tests
if "%1"=="--help" goto :show_help

:do_sync
echo [%time:~0,8%] Syncing project files to %REMOTE_HOST%:%REMOTE_PORT%...

set TEMP_ARCHIVE=%TEMP%\agentdrone_sync_%RANDOM%.tar.gz

tar --exclude=.git --exclude=dataset --exclude=data/samples --exclude=downloads --exclude=keys --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.venv --exclude=venv --exclude=*.tar.gz --exclude=*.part -czf "%TEMP_ARCHIVE%" .

if not exist "%TEMP_ARCHIVE%" (
    echo [ERROR] Failed to create sync archive.
    exit /b 1
)

scp -P %REMOTE_PORT% %SSH_IDENTITY% "%TEMP_ARCHIVE%" %REMOTE_USER%@%REMOTE_HOST%:/tmp/agentdrone_sync.tar.gz >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to upload archive to server. Check network / SSH connection.
    if exist "%TEMP_ARCHIVE%" del "%TEMP_ARCHIVE%" >nul 2>&1
    exit /b 1
)

ssh -p %REMOTE_PORT% %SSH_IDENTITY% %REMOTE_USER%@%REMOTE_HOST% "mkdir -p %REMOTE_DIR% && tar -xzf /tmp/agentdrone_sync.tar.gz -C %REMOTE_DIR% && rm -f /tmp/agentdrone_sync.tar.gz" >nul 2>&1
if exist "%TEMP_ARCHIVE%" del "%TEMP_ARCHIVE%" >nul 2>&1

if errorlevel 1 (
    echo [ERROR] Failed to unpack files on server.
    exit /b 1
)

echo [%time:~0,8%] SUCCESS: Project synchronized to %REMOTE_DIR%

if "%1"=="--once" exit /b 0
if not "%1"=="" goto :eof

pause
exit /b 0

:watch_mode
echo ============================================================================
echo   AgentDrone AUTO-SYNC WATCH MODE
echo   Target: %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PORT% -^> %REMOTE_DIR%
echo   Watching for file changes. Press Ctrl+C to stop.
echo ============================================================================

call :do_sync --once

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%deploy\sync_watcher.ps1"
exit /b 0

:run_tests
call :do_sync --once
echo.
echo ============================================================================
echo   Running pytest on remote server (%REMOTE_HOST%)
echo ============================================================================
ssh -p %REMOTE_PORT% %SSH_IDENTITY% %REMOTE_USER%@%REMOTE_HOST% "cd %REMOTE_DIR% && if [ -d .venv ]; then . .venv/bin/activate; fi && PYTHONPATH=. pytest -v"
exit /b 0

:show_help
echo Usage:
echo   sync_to_server.bat          - Run single sync to server
echo   sync_to_server.bat --watch  - Start real-time auto-sync on file change
echo   sync_to_server.bat --test   - Sync and run pytest tests on server
echo   sync_to_server.bat --help   - Show this help
exit /b 0
