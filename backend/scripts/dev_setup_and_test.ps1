param(
    [string]$Python = "py -3.11",
    [string]$VenvPath = ".venv"
)

Write-Host "Creating virtual environment at $VenvPath (if missing)"
& $Python -m venv $VenvPath
Write-Host "Activating virtual environment"
& .\$VenvPath\Scripts\Activate.ps1
Write-Host "Installing requirements"
pip install -r requirements.txt
Write-Host "Running migrations (creates app.db)"
python run_migrations.py
Write-Host "Running tests"
python -m pytest -q

Write-Host "Done. If tests failed, inspect output above."
