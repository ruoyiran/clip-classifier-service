param(
    [string]$ServiceName = "OpenAIClipClassifierService"
)

$ErrorActionPreference = "Stop"

$service = Get-Service -Name $ServiceName -ErrorAction Stop
if ($service.Status -ne "Stopped") {
    Stop-Service -Name $ServiceName -Force
}

Get-Service -Name $ServiceName
