"""Tests for the sandbox code safety checker.

Verifies that legitimate optimizer code passes and dangerous code is blocked.
Run with: python -m pytest tests/test_sandbox.py -v
"""
import pytest
from sandbox import check_code_safety


# ─── Legitimate code that MUST pass ──────────────────────────────────

class TestLegitimateCode:

    def test_allows_jax_imports(self):
        code = "import jax\nimport jax.numpy as jnp\nimport jax.random"
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_numpy_import(self):
        code = "import numpy as np"
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_math_functools(self):
        code = "import math\nimport functools"
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_pixwake_imports(self):
        code = (
            "from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve\n"
            "from pixwake.optim.sgd import boundary_penalty, spacing_penalty\n"
            "from pixwake.deficit import BastankhahGaussianDeficit\n"
        )
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_jax_lax(self):
        code = "import jax.lax\nfrom jax import lax"
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_typical_optimizer(self):
        """Full realistic optimizer using topfarm_sgd_solve."""
        code = '''
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    key = jax.random.PRNGKey(0)
    init_x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
    key, _ = jax.random.split(key)
    init_y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))

    settings = SGDSettings(learning_rate=50.0, max_iter=4000)
    opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, settings)
    return opt_x, opt_y
'''
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_custom_adam_loop(self):
        """Custom optimizer using jax.grad directly."""
        code = '''
import jax
import jax.numpy as jnp

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    grad_fn = jax.grad(lambda params: objective(params[:n_target], params[n_target:]))
    params = jnp.zeros(2 * n_target)
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    for i in range(1000):
        g = grad_fn(params)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        params = params - 0.01 * m / (jnp.sqrt(v) + 1e-8)
    return params[:n_target], params[n_target:]
'''
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_schedule_fn(self):
        """Schedule-only mode function."""
        code = '''
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr = lr0 * (1 - t)**2
    alpha = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    return lr, alpha, 0.9, 0.999
'''
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_allows_empty_code(self):
        safe, reason = check_code_safety("")
        assert safe, reason

    def test_allows_from_jax_import(self):
        code = "from jax.numpy import array, zeros"
        safe, reason = check_code_safety(code)
        assert safe, reason


# ─── Dangerous imports that MUST fail ────────────────────────────────

class TestBlockedImports:

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "socket", "shutil",
        "http", "urllib", "ctypes", "pathlib",
        "sys", "importlib", "signal", "multiprocessing",
        "threading", "pickle", "marshal", "code",
    ])
    def test_blocks_dangerous_import(self, module):
        code = f"import {module}"
        safe, reason = check_code_safety(code)
        assert not safe
        assert module in reason

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "socket", "shutil",
        "http.client", "urllib.request",
    ])
    def test_blocks_from_import(self, module):
        code = f"from {module} import *"
        safe, reason = check_code_safety(code)
        assert not safe

    def test_blocks_os_path(self):
        safe, _ = check_code_safety("import os.path")
        assert not safe

    def test_blocks_http_server(self):
        safe, _ = check_code_safety("from http.server import HTTPServer")
        assert not safe


# ─── Dangerous builtins that MUST fail ───────────────────────────────

class TestBlockedBuiltins:

    def test_blocks_exec(self):
        safe, reason = check_code_safety("exec('import os')")
        assert not safe
        assert "exec" in reason

    def test_blocks_eval(self):
        safe, reason = check_code_safety("x = eval('1+1')")
        assert not safe
        assert "eval" in reason

    def test_blocks_dunder_import(self):
        safe, reason = check_code_safety("__import__('os')")
        assert not safe
        assert "__import__" in reason

    def test_blocks_open(self):
        safe, reason = check_code_safety("f = open('/etc/passwd')")
        assert not safe
        assert "open" in reason

    def test_blocks_compile(self):
        safe, reason = check_code_safety("compile('code', '', 'exec')")
        assert not safe
        assert "compile" in reason

    def test_blocks_breakpoint(self):
        safe, reason = check_code_safety("breakpoint()")
        assert not safe

    def test_blocks_globals(self):
        safe, reason = check_code_safety("g = globals()")
        assert not safe


# ─── Obfuscation attempts ───────────────────────────────────────────

class TestObfuscationAttempts:

    def test_blocks_aliased_os_import(self):
        safe, _ = check_code_safety("import os as notdangerous")
        assert not safe

    def test_blocks_from_os_import_system(self):
        safe, _ = check_code_safety("from os import system")
        assert not safe

    def test_blocks_dunder_builtins_access(self):
        safe, reason = check_code_safety("x = __builtins__")
        assert not safe
        assert "__builtins__" in reason

    def test_blocks_subclasses_access(self):
        safe, _ = check_code_safety("x = ().__class__.__subclasses__()")
        assert not safe

    def test_blocks_globals_attribute(self):
        code = "def f(): return f.__globals__"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_code_attribute(self):
        code = "def f(): pass\nc = f.__code__"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_nested_import_in_function(self):
        code = "def f():\n    import subprocess\n    subprocess.run(['ls'])"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_exec_in_string(self):
        """exec() call even with innocuous-looking args."""
        code = "exec('x = 1')"
        safe, _ = check_code_safety(code)
        assert not safe


# ─── Information leakage prevention ──────────────────────────────────

class TestInfoLeakage:

    def test_blocks_os_environ_via_import(self):
        """Can't import os to access environ."""
        code = "import os\nrowp = os.environ.get('FUNWAKE_PROBLEM')"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_file_read_attempt(self):
        """Can't use open() to read held-out data."""
        code = "data = open('results/problem_rowp.json').read()"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_pathlib_read(self):
        """Can't use pathlib to read files."""
        code = "import pathlib\ndata = pathlib.Path('results/baselines.json').read_text()"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_subprocess_cat(self):
        """Can't use subprocess to cat files."""
        code = "import subprocess\nsubprocess.run(['cat', '/etc/passwd'])"
        safe, _ = check_code_safety(code)
        assert not safe

    def test_blocks_socket_exfil(self):
        """Can't open sockets for data exfiltration."""
        code = "import socket\ns = socket.socket()"
        safe, _ = check_code_safety(code)
        assert not safe


# ─── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:

    def test_syntax_error_rejected(self):
        safe, reason = check_code_safety("def f(:\n    pass")
        assert not safe
        assert "SyntaxError" in reason

    def test_jax_open_method_not_blocked(self):
        """open as an attribute (e.g. jnp.open) should not be blocked —
        only bare open() calls are blocked."""
        code = "import jax.numpy as jnp\nx = jnp.ones(3)"
        safe, reason = check_code_safety(code)
        assert safe, reason

    def test_multiple_imports_mixed(self):
        """One bad import among good ones still fails."""
        code = "import jax\nimport numpy\nimport os"
        safe, _ = check_code_safety(code)
        assert not safe
