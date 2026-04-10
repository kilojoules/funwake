"""Wrapper to launch vLLM server with overlay packages.

Injects newer packages (e.g. transformers>=5.5 for Gemma 4) before
starting the server. Used on LUMI where the container has older versions.

Usage (same args as `python -m vllm.entrypoints.openai.api_server`):
    python lumi/vllm_serve.py --model /path/to/model --port 8000 ...
"""
import multiprocessing
import os
import sys

# Inject overlay packages if they exist
overlay = "/scratch/project_465002609/julian/containers/gemma4_packages"
if os.path.isdir(overlay):
    sys.path.insert(0, overlay)
    print(f"[vllm_serve] Injected overlay: {overlay}", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Execute the vLLM module as __main__ so multiprocessing spawn works
    import vllm.entrypoints.openai.api_server as server_mod
    server_path = server_mod.__file__
    sys.argv[0] = server_path
    with open(server_path) as f:
        code = f.read()
    exec(compile(code, server_path, "exec"), {"__name__": "__main__", "__file__": server_path})
