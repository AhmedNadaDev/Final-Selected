# NovaCine — One-click setup script (Windows PowerShell)
$ErrorActionPreference = "Stop"

function Banner($msg) {
    Write-Host ""
    Write-Host "▶ $msg" -ForegroundColor Cyan
}

Banner "NovaCine Setup (Windows)"
Write-Host "Text-to-Video Generation System"
Write-Host "================================"

# ── Python check ────────────────────────────────────────────
Banner "Checking Python"
python --version

# ── Backend ─────────────────────────────────────────────────
Banner "Setting up backend"
Push-Location backend
if (-not (Test-Path .venv)) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# CLIP is optional; install best-effort
pip install "git+https://github.com/openai/CLIP.git" 2>$null
deactivate
Pop-Location

# ── Frontend ────────────────────────────────────────────────
Banner "Setting up frontend"
Push-Location frontend
npm install
Pop-Location

# ── Output dirs ─────────────────────────────────────────────
Banner "Creating output directories"
New-Item -ItemType Directory -Force -Path backend\outputs    | Out-Null
New-Item -ItemType Directory -Force -Path evaluation\results | Out-Null

Banner "Setup complete!"
Write-Host ""
Write-Host "Start the backend:" -ForegroundColor Green
Write-Host "  cd backend; .\.venv\Scripts\Activate.ps1; uvicorn main:app --reload --port 8000"
Write-Host ""
Write-Host "Start the frontend:" -ForegroundColor Green
Write-Host "  cd frontend; npm run dev"
Write-Host ""
Write-Host "Or use Docker (NVIDIA GPU recommended):" -ForegroundColor Green
Write-Host "  docker-compose up --build"
Write-Host ""
Write-Host "Open http://localhost:5173 in your browser"
