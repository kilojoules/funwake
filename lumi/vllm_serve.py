"""Wrapper to launch vLLM server with overlay packages.

Injects newer packages (e.g. transformers>=5.5 for Gemma 4) before
starting the server. Used on LUMI where the container has older versions.

Usage (same args as `python -m vllm.entrypoints.openai.api_server`):
    python lumi/vllm_serve.py --model /path/to/model --port 8000 ...
"""
import os
import sys
import runpy

# Inject overlay packages if they exist
overlay = "/scratch/project_465002609/julian/containers/gemma4_packages"
if os.path.isdir(overlay):
    sys.path.insert(0, overlay)
    print(f"[vllm_serve] Injected overlay: {overlay}", flush=True)

# Run vLLM as if invoked with `python -m vllm.entrypoints.openai.api_server`
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
