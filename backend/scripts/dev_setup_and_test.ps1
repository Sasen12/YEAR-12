param(
    [string]$Python = "py",
    [string]$VenvPath = ".venv"
)

Write-Host "Creating virtual environment at $VenvPath (if missing)"
& $Python -m venv $VenvPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create virtual environment using '$Python'."
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "Installing requirements"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Host "Running migrations (creates app.db)"
& $venvPython run_migrations.py
if ($LASTEXITCODE -ne 0) {
    throw "Migration run failed."
}

Write-Host "Running tests"
& $venvPython -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed."
}

Write-Host "Done. Setup and tests passed."
