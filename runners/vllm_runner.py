"""vLLM backend — talks to a local OpenAI-compatible server.

For use with self-hosted models (e.g. Llama 405B on LUMI).
Uses structured-output prompting with JSON parsing rather than
native function calling, which can be flaky on open-weight models.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import requests

from .base import BaseRunner, RunConfig, AttemptResult


class VLLMRunner(BaseRunner):
    """Agent loop using a self-hosted vLLM OpenAI-compatible server."""

    def __init__(self, config: RunConfig,
                 model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
                 base_url: str = "http://localhost:8000",
                 max_tokens: int = 16384,
                 temperature: float = 0.7):
        super().__init__(config)
        self.model = model
        # Normalize base URL: strip trailing /v1 if present (we add it ourselves)
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Verify server is reachable
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=10)
            models = resp.json().get("data", [])
            print(f"[vLLM] Connected to {base_url}, models: {[m['id'] for m in models]}")
        except Exception as e:
            print(f"[vLLM] Warning: server not reachable at {base_url}: {e}")

    def _chat(self, messages: list[dict]) -> str:
        """Send a chat completion request, return assistant text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload, timeout=600
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        # Some models (deepseek-r1) put output in "reasoning" with empty "content"
        content = msg.get("content") or ""
        if not content.strip() and msg.get("reasoning"):
            content = msg["reasoning"]
        return content

    def _parse_action(self, text: str) -> tuple[str, dict]:
        """Parse the LLM's structured action from its response.

        Expected format:
        ACTION: tool_name
        ARGS: {"key": "value", ...}
        or
        CODE:
        ```python
        ...
        ```
        """
        text = text.strip()

        # Look for ACTION: / ARGS: pattern
        if "ACTION:" in text:
            lines = text.split("\n")
            action = None
            args_str = None
            for i, line in enumerate(lines):
                if line.strip().startswith("ACTION:"):
                    action = line.split("ACTION:")[1].strip()
                elif line.strip().startswith("ARGS:"):
                    args_str = line.split("ARGS:")[1].strip()
                    # Collect multiline JSON
                    if not args_str.endswith("}"):
                        for j in range(i+1, len(lines)):
                            args_str += lines[j]
                            if "}" in lines[j]:
                                break
            if action:
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                return action, args

        # Look for code block (implicit write_file + run action)
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
            return "submit_code", {"code": code}

        return "none", {}

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool, return result string."""
        env = {**os.environ, "PYTHONPATH": self.config.pythonpath,
               "JAX_ENABLE_X64": "True"}
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        project_root = os.path.join(os.path.dirname(__file__), "..")

        if name == "read_file":
            path = args.get("path", "")
            allowed = ["playground/", "results/"]
            if not any(path.startswith(p) for p in allowed):
                return f"Error: cannot read {path}"
            full = os.path.join(project_root, path)
            try:
                text = Path(full).read_text()
                return text[:10000]
            except FileNotFoundError:
                return f"Error: {path} not found"

        elif name == "submit_code":
            code = args.get("code", "")
            attempt_num = len(self.attempts) + 1

            # Save to workspace
            iter_path = os.path.join(self.config.output_dir,
                                     f"iter_{attempt_num:03d}.py")
            gen_path = os.path.join(project_root, "playground",
                                    "_generated_opt.py")
            Path(iter_path).write_text(code)
            Path(gen_path).write_text(code)

            # Run tests first
            test_result = subprocess.run(
                ["python", os.path.join(tools_dir, "run_tests.py"),
                 iter_path, "--quick"],
                capture_output=True, text=True, timeout=60, env=env,
                cwd=project_root
            )
            try:
                test_data = json.loads(test_result.stdout)
                if not test_data.get("passed"):
                    return f"Tests FAILED:\n{test_data.get('output', '')}"
            except json.JSONDecodeError:
                pass

            # Score on training farm
            score_result = subprocess.run(
                ["python", os.path.join(tools_dir, "run_optimizer.py"),
                 iter_path,
                 "--problem", os.path.join(project_root, self.config.train_problem),
                 "--timeout", str(self.config.timeout_per_run)],
                capture_output=True, text=True,
                timeout=self.config.timeout_per_run + 30, env=env,
                cwd=project_root
            )

            try:
                data = json.loads(score_result.stdout)
            except json.JSONDecodeError:
                data = {"error": score_result.stdout[:500]}

            # Log attempt
            result = AttemptResult(
                attempt=attempt_num,
                timestamp=time.time(),
            )
            if "error" in data:
                result.error = data["error"][:500]
            else:
                result.train_aep = data.get("aep_gwh")
                result.train_feasible = data.get("feasible")
                result.train_time = data.get("time_s")
                result.train_baseline = data.get("baseline")
                result.strategy = "sgd_solve" if "topfarm_sgd_solve" in code else "custom"

                # Silent ROWP scoring
                rowp_problem = os.path.join(project_root, self.config.rowp_problem)
                if os.path.exists(rowp_problem):
                    try:
                        rowp_result = subprocess.run(
                            ["python", os.path.join(tools_dir, "run_optimizer.py"),
                             iter_path, "--problem", rowp_problem,
                             "--timeout", str(self.config.timeout_per_run + 60)],
                            capture_output=True, text=True,
                            timeout=self.config.timeout_per_run + 90, env=env,
                            cwd=project_root
                        )
                        rowp_data = json.loads(rowp_result.stdout)
                        if "aep_gwh" in rowp_data:
                            result.rowp_aep = rowp_data["aep_gwh"]
                            result.rowp_feasible = rowp_data.get("feasible")
                            result.rowp_time = rowp_data.get("time_s")
                    except Exception:
                        pass

            self.log_attempt(result)

            if result.error:
                return f"ERROR: {result.error}"
            return (f"AEP: {result.train_aep:.2f} GWh "
                    f"(baseline: {result.train_baseline:.2f}, "
                    f"gap: {result.train_aep - result.train_baseline:+.2f})\n"
                    f"Feasible: {result.train_feasible}\n"
                    f"Time: {result.train_time:.1f}s\n"
                    f"Strategy: {result.strategy}\n"
                    f"Best so far: {self.best_aep:.2f} GWh")

        elif name == "run_tests":
            path = args.get("path", "")
            result = subprocess.run(
                ["python", os.path.join(tools_dir, "run_tests.py"),
                 path, self.config.train_problem],
                capture_output=True, text=True, timeout=120, env=env,
                cwd=os.path.join(os.path.dirname(__file__), "..")
            )
            return result.stdout

        elif name == "get_status":
            return json.dumps({
                "attempts": len(self.attempts),
                "best_aep": round(self.best_aep, 2),
                "baseline": round(self._get_baseline_aep(), 2),
                "gap": round(self.best_aep - self._get_baseline_aep(), 2),
                "time_remaining_min": round(self.time_remaining() / 60, 1),
            })

        return f"Unknown action: {name}"

    def run(self):
        """Main loop: structured-output conversation with the vLLM server."""
        self.start_time = time.time()

        system_prompt = self.build_system_prompt()
        system_prompt += """

## How to respond

For each turn, respond with ONE of:

1. Read a file:
   ACTION: read_file
   ARGS: {"path": "playground/pixwake/src/pixwake/optim/sgd.py"}

2. Submit an optimizer (will be tested and scored automatically):
   CODE:
   ```python
   def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
       ...
       return opt_x, opt_y
   ```

3. Check status:
   ACTION: get_status

Always respond with exactly ONE action per turn.
"""
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Hot-start
        first_msg = "Begin optimizing. Read the problem and pixwake source, then write and submit optimizers."
        if self.config.hot_start and os.path.exists(self.config.hot_start):
            seed = Path(self.config.hot_start).read_text()
            first_msg += f"\n\nSeed optimizer:\n```python\n{seed}\n```"
        messages.append({"role": "user", "content": first_msg})

        turn = 0
        while self.time_remaining() > 30:
            turn += 1
            print(f"\n[turn {turn}] {self.time_remaining()/60:.0f} min remaining, "
                  f"{len(self.attempts)} attempts, best={self.best_aep:.1f}")

            try:
                response = self._chat(messages)
            except Exception as e:
                print(f"[turn {turn}] API error: {e}")
                time.sleep(5)
                continue

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            print(f"[turn {turn}] Response: {response[:200]}...")

            # Parse and execute action
            action, args = self._parse_action(response)
            if action == "none":
                messages.append({"role": "user", "content":
                    f"Please respond with an ACTION or CODE block. "
                    f"Time remaining: {self.time_remaining()/60:.0f} min."})
                continue

            result = self._execute_tool(action, args)
            messages.append({"role": "user", "content": f"Result:\n{result}"})
            print(f"[turn {turn}] {action} → {result[:200]}...")

            # Phase-2 nudge
            if self.in_phase2() and turn % 10 == 0:
                messages.append({"role": "user", "content":
                    "Consider trying a custom optimizer with jax.grad instead of topfarm_sgd_solve."})

            # Context pruning
            if len(messages) > 40:
                memory = self.build_memory_context()
                messages = messages[:2] + [
                    {"role": "user", "content": f"[Context compressed]\n{memory}"},
                ] + messages[-20:]

        print(f"\nDone. Best AEP: {self.best_aep:.1f} GWh over {len(self.attempts)} attempts.")
