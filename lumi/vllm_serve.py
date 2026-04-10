"""Wrapper to launch vLLM server with overlay packages.

Injects newer packages (e.g. transformers>=5.5 for Gemma 4) before
starting the server. Used on LUMI where the container has older versions.
"""
import os
import sys

# Inject overlay packages if they exist
overlay = "/scratch/project_465002609/julian/containers/gemma4_packages"
if os.path.isdir(overlay):
    sys.path.insert(0, overlay)
    print(f"[vllm_serve] Injected overlay: {overlay}", flush=True)

# Now import and run vLLM
from vllm.entrypoints.openai.api_server import run_server
import asyncio
asyncio.run(run_server())
