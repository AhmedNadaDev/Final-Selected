#!/usr/bin/env bash
# NovaCine — One-click setup script
set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
RESET="\033[0m"

banner() { echo -e "\n${CYAN}${BOLD}▶ $1${RESET}"; }

banner "NovaCine Setup"
echo "Text-to-Video Generation System"
echo "================================"

# Python version check
banner "Checking Python version"
python3 --version

# Backend setup
banner "Setting up backend"
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Backend dependencies installed${RESET}"
deactivate
cd ..

# Frontend setup
banner "Setting up frontend"
cd frontend
npm install --silent
echo -e "${GREEN}✓ Frontend dependencies installed${RESET}"
cd ..

# Create directories
banner "Creating output directories"
mkdir -p backend/outputs
mkdir -p evaluation/results
echo -e "${GREEN}✓ Directories created${RESET}"

banner "Setup complete!"
echo ""
echo "To start the backend:"
echo "  cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000"
echo ""
echo "To start the frontend:"
echo "  cd frontend && npm run dev"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"
echo ""
echo "Open http://localhost:5173 in your browser"
