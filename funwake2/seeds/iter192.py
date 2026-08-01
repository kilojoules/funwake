"""iter_192 (Claude, schedule_only_5hr) ported to the scale-aware signature.

Seeded ancestor: ancestor=iter_192,
port_transform=lr0->(50/240)*D (=0.2083*D; exactly 50 at D=240), alpha0 passed
through unchanged. Behavior of the archived schedule is otherwise untouched, so
at D=240 this is bit-identical to runs/schedule_only_5hr/iter_192.py (G3).
"""
from _port import make_port

schedule_fn, _archived = make_port("runs/schedule_only_5hr/iter_192.py")
