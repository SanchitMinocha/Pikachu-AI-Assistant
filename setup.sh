#!/bin/bash
# SanchitAI – One-command setup script
# Usage: bash setup.sh

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════╗"
echo "║          SanchitAI – Setup Script            ║"
echo "║   Personal AI Assistant by Sanchit Minocha   ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ---- Step 1: Python virtual environment ----
echo -e "${CYAN}[1/6] Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}  ✓ Virtual environment already exists${NC}"
fi

source venv/bin/activate

# ---- Step 2: Install Python dependencies ----
echo -e "${CYAN}[2/6] Installing Python dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# ---- Step 3: Set up .env ----
echo -e "${CYAN}[3/6] Setting up environment configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}  ⚠ Created .env from .env.example${NC}"
    echo -e "${YELLOW}    Please edit .env to set your API keys / Ollama config${NC}"
else
    echo -e "${GREEN}  ✓ .env already exists${NC}"
fi

# ---- Step 4: Check Ollama ----
echo -e "${CYAN}[4/6] Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}  ✓ Ollama is installed${NC}"
    if ollama list 2>/dev/null | grep -q "llama3.2"; then
        echo -e "${GREEN}  ✓ llama3.2 model is available${NC}"
    else
        echo -e "${YELLOW}  ⚠ Pulling llama3.2:3b model (this may take a while)...${NC}"
        ollama pull llama3.2:3b || echo -e "${YELLOW}    Pull failed — make sure Ollama is running: ollama serve${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ Ollama not found. Install from: https://ollama.com${NC}"
    echo -e "${YELLOW}    Or set HF_API_TOKEN / GROQ_API_KEY in .env for cloud backends${NC}"
fi

# ---- Step 5: Collect fresh data & build index ----
echo -e "${CYAN}[5/6] Collecting latest data and building RAG index...${NC}"
python scripts/collect_web_data.py
python scripts/build_index.py
echo -e "${GREEN}  ✓ RAG index built${NC}"

# ---- Step 6: Create logs directory ----
mkdir -p logs
echo -e "${GREEN}  ✓ Logs directory ready${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗"
echo "║           Setup Complete!                    ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To start the development server:"
echo -e "  ${CYAN}source venv/bin/activate && python app.py${NC}"
echo ""
echo -e "To test the API:"
echo -e "  ${CYAN}curl -X POST http://localhost:5050/api/chat \\"
echo -e "    -H 'Content-Type: application/json' \\"
echo -e "    -d '{\"message\": \"Who is Sanchit Minocha?\"}' | python -m json.tool${NC}"
echo ""
echo -e "To deploy on Apache (see apache/sanchit-ai.conf for details):"
echo -e "  ${CYAN}sudo cp apache/sanchit-ai.conf /etc/apache2/sites-available/"
echo -e "  sudo a2ensite sanchit-ai && sudo systemctl reload apache2${NC}"
echo ""
echo -e "To rebuild index after editing personal_data.json:"
echo -e "  ${CYAN}python scripts/build_index.py${NC}"
echo ""
