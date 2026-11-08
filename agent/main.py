"""CLI and entrypoint for MLOps AI Agent."""

import argparse
import sys
from agent.logger import get_logger
from agent.orchestrator import MLOpsOrchestrator

logger = get_logger("agent_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLOps AI Agent")
    parser.add_argument("--goal", type=str, required=True, help="MLOps workflow goal description")
    parser.add_argument("--model", type=str, default=None, help="LLM model identifier")

    args = parser.parse_args()
    orchestrator = MLOpsOrchestrator(model_name=args.model)
    response = orchestrator.plan_and_execute(goal=args.goal)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

