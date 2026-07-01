#!/usr/bin/env bash
#
# setup_server.sh — one-shot server setup for traffic-signal-control
#
# Works on : Debian & Ubuntu (apt). No PPA required.
# Installs  : git, tmux, a Python venv, SUMO (headless), and all Python deps.
#
# SUMO install strategy (automatic):
#   1. Try the distro's own 'sumo' apt package (Debian/Ubuntu both ship one).
#   2. If unavailable, fall back to the pip 'eclipse-sumo' wheel (bundles the
#      SUMO binaries — no sudo / no PPA needed).
#
# Usage (on a fresh server):
#   sudo apt-get update && sudo apt-get install -y git   # bootstrap git to clone
#   git clone <your-repo-url> && cd traffic-signal-control
#   chmod +x setup_server.sh
#   ./setup_server.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "==> Project directory: $PROJECT_DIR"

# ── 0. Sanity: apt-based system ────────────────────────────────────────────
if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: apt-get not found. This script targets Debian/Ubuntu."
  exit 1
fi

# ── 1. Base system packages (git, tmux, python venv) ───────────────────────
echo "==> Installing base packages (git, tmux, python venv)..."
sudo apt-get update -y
sudo apt-get install -y git tmux python3 python3-venv python3-pip

# ── 2. Try system SUMO from the distro repos (no PPA) ───────────────────────
SUMO_SOURCE=""
SUMO_HOME_DETECTED=""
echo "==> Attempting to install SUMO from distro repositories..."
if sudo apt-get install -y sumo sumo-tools; then
  SUMO_SOURCE="apt"
  if [ -d /usr/share/sumo ]; then
    SUMO_HOME_DETECTED=/usr/share/sumo
  else
    SUMO_HOME_DETECTED="$(dirname "$(dirname "$(readlink -f "$(command -v sumo)")")")/share/sumo"
  fi
  echo "==> Installed system SUMO (apt). SUMO_HOME=$SUMO_HOME_DETECTED"
else
  echo "==> System SUMO package not available; will install via pip (eclipse-sumo)."
fi

# ── 3. Python virtualenv ───────────────────────────────────────────────────
echo "==> Creating virtualenv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# ── 4. PyTorch (CPU build) ─────────────────────────────────────────────────
echo "==> Installing PyTorch (CPU build — PPO MlpPolicy runs on CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ── 5. SUMO via pip if apt didn't provide it ───────────────────────────────
SUMO_BIN_DIR=""
if [ "$SUMO_SOURCE" != "apt" ]; then
  echo "==> Installing SUMO via pip (eclipse-sumo + traci + sumolib)..."
  pip install eclipse-sumo traci sumolib
  SUMO_SOURCE="pip"
  SUMO_HOME_DETECTED="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')"
  SUMO_BIN_DIR="$SUMO_HOME_DETECTED/bin"
  export PATH="$SUMO_BIN_DIR:$PATH"
  echo "==> pip SUMO installed. SUMO_HOME=$SUMO_HOME_DETECTED"
fi

export SUMO_HOME="$SUMO_HOME_DETECTED"

# ── 6. Project Python deps (traci/sumolib pinned in requirements too) ──────
echo "==> Installing Python requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

# ── 7. Persist SUMO_HOME (and PATH if pip-SUMO) into the venv activate ──────
if ! grep -q "SUMO_HOME" "$VENV_DIR/bin/activate"; then
  echo "export SUMO_HOME=\"$SUMO_HOME\"" >> "$VENV_DIR/bin/activate"
  echo "==> Added SUMO_HOME export to venv activate script."
fi
if [ -n "$SUMO_BIN_DIR" ] && ! grep -q "$SUMO_BIN_DIR" "$VENV_DIR/bin/activate"; then
  echo "export PATH=\"$SUMO_BIN_DIR:\$PATH\"" >> "$VENV_DIR/bin/activate"
  echo "==> Added SUMO bin dir to venv PATH."
fi

# ── 8. Verify the full stack ───────────────────────────────────────────────
echo "==> Verifying SUMO binary..."
sumo --version | head -1

echo "==> Verifying Python imports..."
python - <<'PY'
import gymnasium, numpy, traci, sumolib, stable_baselines3, torch
print("  gymnasium          ", gymnasium.__version__)
print("  numpy              ", numpy.__version__)
print("  stable_baselines3  ", stable_baselines3.__version__)
print("  torch              ", torch.__version__, "(cuda:", torch.cuda.is_available(), ")")
print("  OK: all imports resolved")
PY

echo ""
echo "============================================================"
echo " Setup complete.   (SUMO source: $SUMO_SOURCE)"
echo ""
echo " Train inside tmux:"
echo "   tmux new -s train"
echo "   source .venv/bin/activate"
echo "   python -m src.train"
echo "   (detach: Ctrl-b then d   |   reattach: tmux attach -t train)"
echo ""
echo " Monitor with TensorBoard (separate tmux window/session):"
echo "   source .venv/bin/activate"
echo "   tensorboard --logdir logs --host 0.0.0.0 --port 6006"
echo ""
echo " Evaluate after training:"
echo "   python -m src.evaluate"
echo "============================================================"
