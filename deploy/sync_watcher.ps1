# AgentDrone - Live File Watcher and Auto-Sync to Server
param(
    [string]$Server = "89.189.177.216",
    [int]$Port = 28882,
    [string]$User = "snip",
    [string]$RemoteDir = "/home/snip/AgentDrone"
)

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Write-Host "[Watcher] Отслеживание изменений в: $ProjectRoot" -ForegroundColor Cyan
Write-Host "[Watcher] Сервер назначения: ${User}@${Server}:${Port}${RemoteDir}" -ForegroundColor Cyan

# Выбираем ключ SSH при наличии
$SshKey = ""
$KeyCandidates = @(
    "$ProjectRoot\keys\id_ed25519_team",
    "$ProjectRoot\keys\id_ed25519_frontend",
    "$ProjectRoot\keys\id_ed25519_backend",
    "$ProjectRoot\keys\id_ed25519_cv"
)
foreach ($k in $KeyCandidates) {
    if (Test-Path $k) {
        $SshKey = "-i `"$k`""
        break
    }
}

$SshPort = "-p $Port"
$ScpPort = "-P $Port"

function Invoke-Sync {
    $timeStr = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[$timeStr] Обнаружены изменения, синхронизация..." -ForegroundColor Yellow -NoNewline

    $tempArchive = [System.IO.Path]::GetTempFileName() + ".tar.gz"
    try {
        # Создаем архив
        $tarArgs = @(
            "--exclude=.git",
            "--exclude=dataset",
            "--exclude=data",
            "--exclude=downloads",
            "--exclude=keys",
            "--exclude=__pycache__",
            "--exclude=.pytest_cache",
            "--exclude=.ruff_cache",
            "--exclude=.venv",
            "--exclude=venv",
            "--exclude=*.tar.gz",
            "--exclude=*.part",
            "-czf", $tempArchive, "."
        )
        Push-Location $ProjectRoot
        & tar.exe $tarArgs 2>$null
        Pop-Location

        if (-not (Test-Path $tempArchive)) {
            Write-Host " ОШИБКА: не удалось создать архив" -ForegroundColor Red
            return
        }

        # Отправляем на сервер
        $scpDest = "${User}@${Server}:/tmp/agentdrone_sync.tar.gz"
        if ($SshKey) {
            & scp.exe -P $Port $KeyCandidates[0] $tempArchive $scpDest 2>$null
        } else {
            & scp.exe -P $Port $tempArchive $scpDest 2>$null
        }

        # Распаковываем
        $remoteCmd = "mkdir -p $RemoteDir && tar -xzf /tmp/agentdrone_sync.tar.gz -C $RemoteDir && rm -f /tmp/agentdrone_sync.tar.gz"
        if ($SshKey) {
            & ssh.exe -p $Port $KeyCandidates[0] "${User}@${Server}" $remoteCmd 2>$null
        } else {
            & ssh.exe -p $Port "${User}@${Server}" $remoteCmd 2>$null
        }

        Write-Host " УСПЕШНО" -ForegroundColor Green
    }
    catch {
        Write-Host " СБОЙ: $_" -ForegroundColor Red
    }
    finally {
        if (Test-Path $tempArchive) {
            Remove-Item -Force $tempArchive -ErrorAction SilentlyContinue
        }
    }
}

# Инициализируем FileSystemWatcher
$Watcher = New-Object System.IO.FileSystemWatcher
$Watcher.Path = $ProjectRoot
$Watcher.IncludeSubdirectories = $true
$Watcher.EnableRaisingEvents = $true

$IgnoredDirs = @('\.git\', '\dataset\', '\data\', '\downloads\', '\keys\', '\__pycache__\', '\.pytest_cache\', '\.ruff_cache\', '\.venv\', '\venv\')
$ValidExtensions = @('.py', '.md', '.json', '.txt', '.html', '.css', '.yaml', '.yml', '.toml', '.sh', '.bat')

$LastSyncTime = [DateTime]::MinValue
$DebounceMs = 1200

$Action = {
    param($source, $event)
    $path = $event.FullPath

    # Проверка игнорируемых директорий
    foreach ($ign in $IgnoredDirs) {
        if ($path.Contains($ign)) { return }
    }

    # Проверка расширения файла
    $ext = [System.IO.Path]::GetExtension($path).ToLower()
    if ($ValidExtensions -notcontains $ext) { return }

    $now = [DateTime]::UtcNow
    $diff = ($now - $script:LastSyncTime).TotalMilliseconds
    if ($diff -gt $DebounceMs) {
        $script:LastSyncTime = $now
        Invoke-Sync
    }
}

Register-ObjectEvent $Watcher "Changed" -Action $Action | Out-Null
Register-ObjectEvent $Watcher "Created" -Action $Action | Out-Null
Register-ObjectEvent $Watcher "Deleted" -Action $Action | Out-Null

Write-Host "[Watcher] Наблюдатель активен. Измените любой .py или .md файл, чтобы увидеть синхронизацию." -ForegroundColor Green
Write-Host "[Watcher] Нажмите Ctrl+C для остановки." -ForegroundColor Gray

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    $Watcher.EnableRaisingEvents = $false
    $Watcher.Dispose()
    Write-Host "`n[Watcher] Остановлен." -ForegroundColor Yellow
}
