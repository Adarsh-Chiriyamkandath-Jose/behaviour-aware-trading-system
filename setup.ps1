# Windows setup: creates .venv, installs dependencies, and activates it.
# Usage (from project root):  .\setup.ps1

python setup_env.py
if ($LASTEXITCODE -eq 0) {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "Environment ready and activated." -ForegroundColor Green
}
