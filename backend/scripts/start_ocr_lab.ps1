param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$Python = "py",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing OCR Lab..."
& $Python -m venv $VenvPath
$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "Installing requirements (if needed)..."
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r requirements.txt

$url = "http://$BindHost`:$Port/ocr"
Write-Host "Opening OCR Lab at $url"
Start-Process $url | Out-Null

Write-Host "Starting API server..."
& $venvPython -m uvicorn app.main:app --host $BindHost --port $Port --reload
