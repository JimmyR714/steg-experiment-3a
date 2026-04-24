#!/bin/bash
set -e

echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

echo "=== Creating virtual environment ==="
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

echo "=== Installing vLLM ==="
pip install --upgrade pip
pip install vllm

echo "=== Downloading and starting Qwen2.5-14B-AWQ ==="
echo "Model will be downloaded on first run (~8 GB)"

echo ""
echo "=== Setup complete ==="
echo "To activate the environment:  source ~/vllm-env/bin/activate"
echo "To start the server:"
echo "  vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --dtype auto --max-model-len 4096"
echo "To use the logprobs script:"
echo "  python3 logprobs.py --prompt 'Your text here'"
