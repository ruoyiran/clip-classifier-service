param(
    [string]$ServiceName = "ClipClassifierService",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Nssm = Get-Command nssm.exe -ErrorAction SilentlyContinue
$Logs = Join-Path $Root "logs"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run setup from README.md first."
}

if (-not $Nssm) {
    throw "nssm.exe is required to install the Windows service. Install NSSM and retry."
}

New-Item -ItemType Directory -Force $Logs | Out-Null

& $Nssm.Source install $ServiceName $Python "-m uvicorn clip_classifier_service.main:app --host 0.0.0.0 --port $Port --workers 1"
& $Nssm.Source set $ServiceName AppDirectory $Root
& $Nssm.Source set $ServiceName AppStdout (Join-Path $Logs "service.out.log")
& $Nssm.Source set $ServiceName AppStderr (Join-Path $Logs "service.err.log")
& $Nssm.Source set $ServiceName Start SERVICE_AUTO_START

Write-Output "Installed $ServiceName on port $Port. Start with: Start-Service $ServiceName"
