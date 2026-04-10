"""Wrapper to launch vLLM server with overlay packages.

Injects newer packages (e.g. transformers>=5.5 for Gemma 4) before
starting the server. Used on LUMI where the container has older versions.

Usage (same args as vllm.entrypoints.openai.api_server):
    python lumi/vllm_serve.py --model /path/to/model --port 8000 ...
"""
import os
import sys

# Inject overlay packages if they exist
overlay = "/scratch/project_465002609/julian/containers/gemma4_packages"
if os.path.isdir(overlay):
    sys.path.insert(0, overlay)
    print(f"[vllm_serve] Injected overlay: {overlay}", flush=True)

# Re-invoke as module so vLLM's own arg parsing works
sys.argv[0] = "vllm"
from vllm.entrypoints.openai.api_server import main
main()
