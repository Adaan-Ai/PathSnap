param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$ModelPath = "models\pathsnap_rf.joblib"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot -ErrorAction Stop).Path
$resolvedModelPath = Join-Path $repoRoot $ModelPath
if (-not (Test-Path -LiteralPath $resolvedModelPath -PathType Leaf)) {
    throw "Model artifact not found: $resolvedModelPath"
}
if (-not (Get-ChildItem -LiteralPath $resolvedDataRoot -Directory -ErrorAction Stop | Select-Object -First 1)) {
    throw "Data root must contain one or more trip directories: $resolvedDataRoot"
}

$env:PATHSNAP_DATA_ROOT = $resolvedDataRoot
$env:PATHSNAP_MODEL = $resolvedModelPath
$env:PATHSNAP_HOST = "0.0.0.0"
$env:PATHSNAP_PORT = "8000"
$env:PYTHONPATH = Join-Path $repoRoot "src"
Set-Location -LiteralPath $repoRoot
Write-Host "Starting PathSnap for trip inputs at $resolvedDataRoot"
Write-Host "On this laptop: http://localhost:8000/"
Write-Host "On another laptop on the same private network: http://<this-laptop-ip>:8000/"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $pythonExe -PathType Leaf) {
    & $pythonExe -m pathsnap.service
} else {
    python -m pathsnap.service
}
