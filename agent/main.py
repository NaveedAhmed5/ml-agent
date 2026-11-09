"""Command Line Interface for MLOps Autonomous AI Agent.

Supports:
1. Single-Query Mode:
   python -m agent.main --query "Find the best model and check if retraining is necessary"
   python -m agent.main --query "..." --json
2. Interactive REPL Mode:
   python -m agent.main
"""

from __future__ import annotations

import argparse
import json

from agent.logger import get_logger
from agent.orchestrator import MLOpsOrchestrator
from agent.schemas import ToolExecutionStatus

logger = get_logger("agent_cli")


def print_banner() -> None:
    """Print the interactive greeting banner."""
    print("=" * 80)
    print("  \033[1;36m🤖 MLOps Autonomous AI Agent — Operations Control Center\033[0m")
    print("=" * 80)
    print("  Enterprise workflow orchestration across \033[1mDVC\033[0m, \033[1mMLflow\033[0m, \033[1mAirflow\033[0m, and \033[1mKubernetes\033[0m.")
    print("  Special Commands:")
    print("    \033[33mhelp\033[0m   - Display sample queries for all 5 evaluation scenarios")
    print("    \033[33mtools\033[0m  - List all 21 registered MLOps tools by operational domain")
    print("    \033[33mclear\033[0m  - Clear terminal screen")
    print("    \033[33mexit\033[0m   - Exit interactive session")
    print("=" * 80)


def print_help_scenarios() -> None:
    """Display sample queries for all 5 evaluation scenarios."""
    print("\n\033[1;32m=== Evaluation Scenarios & Sample Queries ===\033[0m\n")
    print("\033[1mScenario 1: DVC Data Management Agent\033[0m")
    print("  'List all DVC tracked datasets, verify current version and hash, and check for uncommitted changes.'\n")
    print("\033[1mScenario 2: MLflow Experiment Intelligence Agent\033[0m")
    print("  'Find the best performing model in MLflow, evaluate production readiness, and compare with candidate runs.'\n")
    print("\033[1mScenario 3: Apache Airflow Workflow Automation Agent\033[0m")
    print("  'List available Airflow DAGs, check status of mlops_model_training_pipeline, and diagnose any failed tasks.'\n")
    print("\033[1mScenario 4: Kubernetes Operations Agent\033[0m")
    print("  'Inspect Kubernetes pods and deployments in mlops-prod, diagnose unhealthy containers, and audit app version.'\n")
    print("\033[1mScenario 5: Mandatory End-to-End Multi-Tool Integration\033[0m")
    print("  'Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.'\n")


def print_tool_registry(orchestrator: MLOpsOrchestrator) -> None:
    """Print all registered tools grouped by domain."""
    tools = orchestrator.registry.list_all()
    by_domain: dict[str, list] = {}
    for t in tools:
        by_domain.setdefault(t.domain.upper(), []).append(t)

    print(f"\n\033[1;32m=== Registered MLOps Tools ({len(tools)} Total) ===\033[0m\n")
    for domain, d_tools in by_domain.items():
        print(f"\033[1;34m[{domain} DOMAIN] ({len(d_tools)} tools)\033[0m")
        for t in d_tools:
            print(f"  • \033[1m{t.name}\033[0m: {t.description}")
        print()


def display_trace_and_response(response, verbose: bool = False) -> None:
    """Pretty print the execution trace and the final grounded response."""
    print("\n" + "—" * 80)
    print(f"\033[1mGoal:\033[0m {response.goal}")
    print(f"\033[1mRequest ID:\033[0m {response.request_id} | \033[1mStatus:\033[0m {response.status.value}")
    print("—" * 80)

    # Tool Execution Trace
    if response.steps_executed:
        print("\n\033[1;33m📋 Operational Tool Execution Trace:\033[0m")
        for i, step in enumerate(response.steps_executed, 1):
            status_color = "\033[32m" if step.status == ToolExecutionStatus.SUCCESS else "\033[31m"
            print(
                f"  [{i}] \033[1m{step.tool_name}\033[0m — "
                f"{status_color}{step.status.value}\033[0m ({step.duration_ms:.1f}ms)"
            )
            if step.status != ToolExecutionStatus.SUCCESS and step.error:
                print(f"      \033[31mNotice: {step.error}\033[0m")
            elif verbose and step.data:
                print(f"      Data: {json.dumps(step.data, default=str)[:120]}...")
        print()

    # Grounded Report Output
    print("\033[1;36m🎯 Grounded AI Agent Response:\033[0m\n")
    print(response.summary)
    print("\n" + "=" * 80)


def run_interactive(orchestrator: MLOpsOrchestrator, verbose: bool = False) -> None:
    """Execute the interactive REPL."""
    print_banner()

    while True:
        try:
            user_input = input("\n\033[1;35mmlops-agent > \033[0m").strip()
            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                print("\nShutting down MLOps AI Agent. Session completed.\n")
                break
            elif cmd_lower == "help":
                print_help_scenarios()
                continue
            elif cmd_lower == "tools":
                print_tool_registry(orchestrator)
                continue
            elif cmd_lower == "clear":
                print("\033[2J\033[H", end="")
                print_banner()
                continue

            response = orchestrator.plan_and_execute(goal=user_input)
            display_trace_and_response(response, verbose=verbose)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break
        except Exception as exc:
            print(f"\n\033[31mError during execution: {exc}\033[0m")


def main() -> None:
    """Parse arguments and launch appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Autonomous Production-Ready MLOps AI Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single-query goal to execute non-interactively (e.g. --query 'Check model status and explain if retraining is needed')",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Alias for --query",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw structured JSON of AgentResponse (useful for pipelines and automation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model (e.g. gemini-2.5-flash, gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose execution data and payloads",
    )

    args = parser.parse_args()

    orchestrator = MLOpsOrchestrator(model_name=args.model)
    query = args.query or args.goal

    if query:
        response = orchestrator.plan_and_execute(goal=query)
        if args.json:
            print(response.model_dump_json(indent=2))
        else:
            display_trace_and_response(response, verbose=args.verbose)
    else:
        run_interactive(orchestrator, verbose=args.verbose)


if __name__ == "__main__":
    main()
