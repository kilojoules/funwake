"""Wrapper to launch vLLM server with overlay packages if needed.

Usage (same args as `python -m vllm.entrypoints.openai.api_server`):
    python lumi/vllm_serve.py --model /path/to/model --port 8000 ...
"""
import os
import sys

# Inject overlay packages if they exist (e.g. newer transformers for new models)
overlay = "/scratch/project_465002609/julian/containers/gemma4_packages"
if os.path.isdir(overlay):
    sys.path.insert(0, overlay)
    print(f"[vllm_serve] Injected overlay: {overlay}", flush=True)

if __name__ == "__main__":
    # Use runpy to properly run as __main__
    import runpy
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
