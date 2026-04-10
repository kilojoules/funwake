"""Static code safety checker for LLM-generated optimizer scripts.

Uses AST analysis to enforce an import allowlist and block dangerous
builtins. Designed to be fast (<1s) so it doesn't slow the iteration loop.

Usage:
    from sandbox import check_code_safety
    safe, reason = check_code_safety(code_string)
"""
import ast

# Modules the optimizer legitimately needs — everything else is blocked.
ALLOWED_MODULE_PREFIXES = (
    "jax",
    "numpy",
    "math",
    "functools",
    "pixwake",
)

# Builtins that could be used to escape the sandbox.
DANGEROUS_BUILTINS = frozenset({
    "exec", "eval", "__import__", "compile",
    "globals", "locals", "vars", "breakpoint",
    "getattr", "setattr", "delattr",
})

# Dunder attributes used in sandbox escape exploits.
DANGEROUS_DUNDERS = frozenset({
    "__builtins__", "__subclasses__", "__bases__",
    "__globals__", "__code__", "__import__",
})


def check_code_safety(code: str) -> tuple[bool, str]:
    """Check whether optimizer code is safe to execute.

    Returns (True, "") if the code passes all checks,
    or (False, reason) describing the first violation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # --- Import checks (allowlist) ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed(alias.name):
                    return False, f"Blocked import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module and not _module_allowed(node.module):
                return False, f"Blocked import: from {node.module}"

        # --- Dangerous builtin calls ---
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "open":
                return False, "Blocked: open() — optimizer should not do file I/O"
            if name in DANGEROUS_BUILTINS:
                return False, f"Blocked builtin: {name}()"

        # --- Dangerous dunder attribute access ---
        elif isinstance(node, ast.Attribute):
            if node.attr in DANGEROUS_DUNDERS:
                return False, f"Blocked attribute: .{node.attr}"

        # --- Dangerous dunder name references (e.g. bare __builtins__) ---
        elif isinstance(node, ast.Name):
            if node.id in DANGEROUS_DUNDERS:
                return False, f"Blocked name: {node.id}"

    return True, ""


def _module_allowed(module_name: str) -> bool:
    """Check if a module is in the allowlist."""
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in ALLOWED_MODULE_PREFIXES
    )


def _call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node, if it's a simple name."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""
