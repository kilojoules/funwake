"""iter_118 (Gemini, gemini_cli_5hr) ported to the scale-aware signature.

Seeded ancestor: ancestor=iter_118,
port_transform=lr0->(50/240)*D (=0.2083*D; exactly 50 at D=240), alpha0 passed
through unchanged. At D=240 bit-identical to runs/gemini_cli_5hr/iter_118.py (G3).
"""
from _port import make_port

schedule_fn, _archived = make_port("runs/gemini_cli_5hr/iter_118.py")
