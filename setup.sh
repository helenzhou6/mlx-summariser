apt update
apt install nvtop htop tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
source .venv/bin/activate
tmux