param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$Python = "py",
    [string]$VenvPath = ".venv",
    [int]$OcrTimeoutSeconds = 60,
    [int]$OcrMaxImageSide = 1400,
    [string]$OcrEngines = "rapidocr",
    [int]$OcrRateLimitPerMin = 120
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

$env:OCR_REQUEST_TIMEOUT_SECONDS = "$OcrTimeoutSeconds"
$env:OCR_MAX_IMAGE_SIDE = "$OcrMaxImageSide"
$env:OCR_ENGINES = "$OcrEngines"
$env:OCR_RATE_LIMIT_PER_MIN = "$OcrRateLimitPerMin"
Write-Host "OCR_REQUEST_TIMEOUT_SECONDS=$($env:OCR_REQUEST_TIMEOUT_SECONDS)"
Write-Host "OCR_MAX_IMAGE_SIDE=$($env:OCR_MAX_IMAGE_SIDE)"
Write-Host "OCR_ENGINES=$($env:OCR_ENGINES)"
Write-Host "OCR_RATE_LIMIT_PER_MIN=$($env:OCR_RATE_LIMIT_PER_MIN)"

$url = "http://$BindHost`:$Port/ocr"
Write-Host "Opening OCR Lab at $url"
Start-Process $url | Out-Null

Write-Host "Starting API server..."
& $venvPython -m uvicorn app.main:app --host $BindHost --port $Port --reload
