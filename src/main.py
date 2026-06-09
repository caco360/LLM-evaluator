import argparse
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

from config import load_prompt
from db import initDB, insert_case_results, insert_run
from eval import evaluate_prompt, run_summary, check_latest_accuracy_drop

WARNING_THRESHOLD = 0.03


def build_run_id(prompt_version: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"run_{timestamp}_{prompt_version}_{suffix}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM regression evaluation.")
    parser.add_argument(
        "--prompt",
        default="prompts/support_classifier_v5_conservative.yaml",
        help="Path to the prompt YAML file.",
    )
    parser.add_argument(
        "--dataset",
        default="src/data/golden_dataset.json",
        help="Path to the golden dataset JSON file.",
    )
    args = parser.parse_args()

    initDB()

    prompt = load_prompt(args.prompt)
    case_results = await evaluate_prompt(args.prompt, args.dataset)
    summary = run_summary(case_results, args.prompt, args.dataset, prompt)
    summary["run_id"] = build_run_id(summary["prompt_version"])

    insert_run(summary)
    insert_case_results(summary["run_id"], case_results)
    check_latest_accuracy_drop(WARNING_THRESHOLD)


    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
