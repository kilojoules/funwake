#!/usr/bin/env python
"""FunWake agent CLI — supports Gemini API and Claude Code backends.

Usage:
  # Gemini (original behavior)
  python agent_cli.py --provider gemini --model gemini-2.5-flash \\
      --time-budget 3600 --hot-start results/seed_optimizer.py

  # Claude Code
  python agent_cli.py --provider claude-code \\
      --time-budget 3600 --hot-start results/seed_optimizer.py

  # Claude Code, single long session (more autonomous)
  python agent_cli.py --provider claude-code --cc-iterations 1 \\
      --cc-max-turns 100 --time-budget 3600
"""
import argparse
import sys

from runners import RunConfig, GeminiRunner, ClaudeCodeRunner, VLLMRunner


def main():
    p = argparse.ArgumentParser(
        description="LLM agent for wind farm layout optimization"
    )

    # Shared arguments
    p.add_argument("--provider", required=True,
                   choices=["gemini", "claude-code", "vllm"],
                   help="LLM backend to use")
    p.add_argument("--wind-csv", required=True,
                   help="Path to wind resource CSV")
    p.add_argument("--time-budget", type=int, default=3600,
                   help="Total time budget in seconds (default: 3600)")
    p.add_argument("--hot-start", default=None,
                   help="Path to seed optimizer script")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: results_agent_{provider})")
    p.add_argument("--timeout-per-run", type=int, default=60,
                   help="Timeout per optimizer evaluation (default: 60s)")

    # Gemini/vLLM-specific
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Model name (default: gemini-2.5-flash)")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="vLLM server URL (default: http://localhost:8000)")

    # Claude Code-specific
    p.add_argument("--cc-max-turns", type=int, default=30,
                   help="Max turns per Claude Code invocation (default: 30)")
    p.add_argument("--cc-iterations", type=int, default=0,
                   help="Number of Claude Code invocations. 0=auto (default: 0)")
    p.add_argument("--schedule-only", action="store_true",
                   help="Constrain LLM to write only schedule_fn(), not optimize()")

    args = p.parse_args()

    # Build config
    output_dir = args.output_dir or f"results_agent_{args.provider.replace('-', '_')}"
    config = RunConfig(
        wind_csv=args.wind_csv,
        time_budget=args.time_budget,
        output_dir=output_dir,
        hot_start=args.hot_start,
        timeout_per_run=args.timeout_per_run,
    )

    # Create runner
    if args.provider == "gemini":
        runner = GeminiRunner(config, model=args.model)
    elif args.provider == "claude-code":
        runner = ClaudeCodeRunner(
            config,
            max_turns_per_iter=args.cc_max_turns,
            iterations=args.cc_iterations,
            schedule_only=args.schedule_only,
        )
    elif args.provider == "vllm":
        runner = VLLMRunner(
            config,
            model=args.model,
            base_url=args.base_url,
        )
    else:
        print(f"Unknown provider: {args.provider}", file=sys.stderr)
        sys.exit(1)

    # Run
    print(f"Starting {args.provider} agent")
    print(f"  Time budget: {args.time_budget/60:.0f} min")
    print(f"  Output: {output_dir}/")
    if args.hot_start:
        print(f"  Hot-start: {args.hot_start}")
    print()

    runner.run()


if __name__ == "__main__":
    main()
