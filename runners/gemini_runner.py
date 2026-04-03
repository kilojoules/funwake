"""Gemini backend — tool-calling loop via the Gemini API.

This is a refactored version of the original agent_cli.py that delegates
tool execution to the standalone scripts in tools/.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from .base import BaseRunner, RunConfig, AttemptResult

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ─── Tool definitions for Gemini function calling ────────────────────

TOOL_DECLARATIONS = [
    {
        "name": "read_file",
        "description": "Read a file from the playground or results directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to read (e.g. playground/pixwake/src/pixwake/optim/sgd.py)"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write an optimizer script to the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Filename for the optimizer (e.g. my_optimizer.py)"},
                "content": {"type": "string", "description": "Python source code"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_tests",
        "description": "Run unit tests on an optimizer script. Returns pass/fail with details.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to optimizer script"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "run_optimizer",
        "description": "Score an optimizer on the training farm. Returns AEP in GWh.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to optimizer script"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "test_generalization",
        "description": "Test optimizer on held-out farm. Returns PASS/FAIL only — no AEP.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to optimizer script"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "get_status",
        "description": "Get current best AEP vs baseline and attempt summary.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
]


class GeminiRunner(BaseRunner):
    """Gemini function-calling agent loop."""

    def __init__(self, config: RunConfig, model: str = "gemini-2.5-flash"):
        super().__init__(config)
        if genai is None:
            raise ImportError("google-genai package required for Gemini backend")

        api_key = self._load_api_key()
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.conversation_history = []

    def _load_api_key(self) -> str:
        key_path = os.path.expanduser("~/.gem")
        if os.path.exists(key_path):
            return Path(key_path).read_text().strip()
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("No Gemini API key found (~/.gem or GEMINI_API_KEY)")
        return key

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool call, delegating to standalone scripts."""
        env = {**os.environ, "PYTHONPATH": self.config.pythonpath,
               "JAX_ENABLE_X64": "True"}
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")

        if name == "read_file":
            path = args["path"]
            # Restrict to allowed directories
            allowed_prefixes = ["playground/", "results/"]
            if not any(path.startswith(p) for p in allowed_prefixes):
                return json.dumps({"error": f"Cannot read {path} — only playground/ and results/ allowed"})
            try:
                return Path(path).read_text()[:10000]
            except FileNotFoundError:
                return json.dumps({"error": f"File not found: {path}"})

        elif name == "write_file":
            filename = os.path.basename(args["path"])
            attempt_num = len(self.attempts) + 1
            # Save as iter_NNN.py and also as the requested name
            workspace_path = os.path.join(self.config.output_dir,
                                          f"iter_{attempt_num:03d}.py")
            generated_path = os.path.join("playground", f"_generated_opt.py")
            Path(workspace_path).write_text(args["content"])
            Path(generated_path).write_text(args["content"])
            return json.dumps({"saved": workspace_path, "generated": generated_path})

        elif name == "run_tests":
            result = subprocess.run(
                ["python", os.path.join(tools_dir, "run_tests.py"),
                 args["path"], self.config.train_problem],
                capture_output=True, text=True, timeout=120, env=env
            )
            return result.stdout

        elif name == "run_optimizer":
            result = subprocess.run(
                ["python", os.path.join(tools_dir, "run_optimizer.py"),
                 args["path"],
                 "--problem", self.config.train_problem,
                 "--timeout", str(self.config.timeout_per_run)],
                capture_output=True, text=True, timeout=self.config.timeout_per_run + 10,
                env=env
            )
            # Log the attempt
            try:
                data = json.loads(result.stdout)
                attempt = AttemptResult(
                    attempt=len(self.attempts) + 1,
                    timestamp=time.time(),
                )
                if "error" in data:
                    attempt.error = data["error"]
                else:
                    attempt.train_aep = data.get("aep_gwh")
                    attempt.train_feasible = data.get("feasible")
                    attempt.train_time = data.get("time_s")
                    attempt.train_baseline = data.get("baseline")
                    attempt.strategy = "sgd_solve"
                self.log_attempt(attempt)
            except (json.JSONDecodeError, KeyError):
                pass
            return result.stdout

        elif name == "test_generalization":
            result = subprocess.run(
                ["python", os.path.join(tools_dir, "test_generalization.py"),
                 args["path"],
                 "--problem", self.config.rowp_problem,
                 "--timeout", str(self.config.timeout_per_run + 30)],
                capture_output=True, text=True,
                timeout=self.config.timeout_per_run + 40, env=env
            )
            # Update the last attempt with ROWP results
            try:
                data = json.loads(result.stdout)
                if self.attempts and "error" not in self.attempts[-1]:
                    self.attempts[-1]["rowp_feasible"] = data.get("feasible")
                    self.attempts[-1]["rowp_time"] = data.get("time_s")
                    # Save log
                    with open(self.log_path, "w") as f:
                        json.dump(self.attempts, f, indent=2)
            except (json.JSONDecodeError, KeyError):
                pass
            return result.stdout

        elif name == "get_status":
            result = subprocess.run(
                ["python", os.path.join(tools_dir, "get_status.py"),
                 "--log", self.log_path,
                 "--baselines", self.config.baselines,
                 "--train-farm", self.config.train_farm],
                capture_output=True, text=True, timeout=10, env=env
            )
            return result.stdout

        return json.dumps({"error": f"Unknown tool: {name}"})

    def run(self):
        """Main Gemini tool-calling loop."""
        self.start_time = time.time()

        # Initialize conversation
        system_prompt = self.build_system_prompt()
        self.conversation_history = []

        # Hot-start: include seed optimizer in first message
        first_msg = "Begin optimizing. Explore the codebase, then write and test optimizers."
        if self.config.hot_start and os.path.exists(self.config.hot_start):
            seed = Path(self.config.hot_start).read_text()
            first_msg += f"\n\nHere's a seed optimizer to start from:\n```python\n{seed}\n```"

        self.conversation_history.append({"role": "user", "parts": [{"text": first_msg}]})

        turn = 0
        while self.time_remaining() > 0:
            turn += 1

            # Inject memory context periodically
            if turn % 10 == 0:
                memory = self.build_memory_context()
                self.conversation_history.append({
                    "role": "user",
                    "parts": [{"text": f"[Status update]\n{memory}\n\nTime remaining: {self.time_remaining()/60:.0f} min. Continue optimizing."}]
                })

            # Call Gemini
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.conversation_history,
                    config={
                        "system_instruction": system_prompt,
                        "tools": [{"function_declarations": TOOL_DECLARATIONS}],
                    }
                )
            except Exception as e:
                print(f"[turn {turn}] Gemini API error: {e}")
                time.sleep(5)
                continue

            # Process response
            candidate = response.candidates[0]
            assistant_parts = []

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    print(f"[turn {turn}] Tool: {fc.name}({json.dumps(dict(fc.args))[:80]}...)")
                    result = self._execute_tool(fc.name, dict(fc.args))
                    assistant_parts.append(part)

                    # Add tool result
                    self.conversation_history.append({
                        "role": "model", "parts": assistant_parts
                    })
                    self.conversation_history.append({
                        "role": "user",
                        "parts": [{"function_response": {"name": fc.name, "response": {"result": result}}}]
                    })
                    assistant_parts = []
                elif hasattr(part, "text") and part.text:
                    print(f"[turn {turn}] LLM: {part.text[:200]}...")
                    assistant_parts.append(part)

            if assistant_parts:
                self.conversation_history.append({
                    "role": "model", "parts": assistant_parts
                })

            # Context pruning after 40 turns
            if turn % 40 == 0 and len(self.conversation_history) > 20:
                self._compact_history()

        print(f"\nDone. Best AEP: {self.best_aep:.1f} GWh over {len(self.attempts)} attempts.")

    def _compact_history(self):
        """Compress old conversation turns to prevent quality degradation."""
        # Keep system context, first 2 messages, and last 10 messages
        if len(self.conversation_history) <= 12:
            return
        summary = self.build_memory_context()
        kept = self.conversation_history[:2] + [{
            "role": "user",
            "parts": [{"text": f"[Conversation compacted. Summary of progress:]\n{summary}"}]
        }] + self.conversation_history[-10:]
        self.conversation_history = kept
