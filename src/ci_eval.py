import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import load_prompt
from db import initDB, insert_case_results, insert_run
from eval import check_latest_accuracy_drop, evaluate_prompt, run_summary
from runs.report import generate_latest_run_report


REPORT_DIR = Path("src/runs/reports")


def build_run_id(prompt_version: str, label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"ci_{label}_{timestamp}_{prompt_version}_{suffix}"


async def run_and_store_eval(prompt_path: str, dataset_path: str, label: str) -> dict:
    prompt = load_prompt(prompt_path)
    case_results = await evaluate_prompt(prompt_path, dataset_path)
    summary = run_summary(case_results, prompt_path, dataset_path, prompt)
    summary["run_id"] = build_run_id(summary["prompt_version"], label)

    insert_run(summary)
    insert_case_results(summary["run_id"], case_results)

    if summary["success_count"] == 0:
        raise RuntimeError(
            f"{label} eval produced 0 successful cases. "
            "Check OPENAI_API_KEY, model output format, and case-level errors."
        )

    return summary


def write_pr_comment(
    candidate_summary: dict,
    warning_alerts: list[str],
    critical_alerts: list[str],
    report_path: Path,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comment_path = REPORT_DIR / "pr_comment.md"

    status = "CRITICAL" if critical_alerts else "WARNING" if warning_alerts else "PASS"
    lines = [
        "## Model Regression Eval",
        "",
        f"Status: **{status}**",
        "",
        f"Prompt: `{candidate_summary['prompt_version']}`",
        f"Model: `{candidate_summary['model']}`",
        f"Category accuracy: `{candidate_summary['overall_category_accuracy']:.2%}`",
        f"Summary score: `{candidate_summary['overall_summary_score']:.2f}`",
        f"Total tokens: `{candidate_summary['total_tokens']}`",
        f"Average latency: `{candidate_summary['avg_latency_ms']:.2f} ms`",
        "",
    ]

    if warning_alerts:
        lines.append("### Warnings")
        lines.extend(f"- {alert}" for alert in warning_alerts)
        lines.append("")

    if critical_alerts:
        lines.append("### Critical Regressions")
        lines.extend(f"- {alert}" for alert in critical_alerts)
        lines.append("")

    lines.extend(
        [
            "### Report",
            f"HTML report generated at `{report_path}`.",
            "",
        ]
    )

    comment_path.write_text("\n".join(lines), encoding="utf-8")
    return comment_path


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run CI model regression eval.")
    parser.add_argument(
        "--baseline-prompt",
        default="prompts/support_classifier_v1_balanced.yaml",
        help="Prompt YAML used as the baseline.",
    )
    parser.add_argument(
        "--candidate-prompt",
        required=True,
        help="Prompt YAML being evaluated in the PR.",
    )
    parser.add_argument(
        "--dataset",
        default="src/data/golden_dataset.json",
        help="Golden dataset JSON file.",
    )
    parser.add_argument("--warning-category-threshold", type=float, default=0.03)
    parser.add_argument("--warning-summary-threshold", type=float, default=0.03)
    parser.add_argument("--warning-tokens-threshold", type=float, default=1000.0)
    parser.add_argument("--warning-latency-threshold", type=float, default=500.0)
    parser.add_argument("--critical-category-threshold", type=float, default=0.08)
    parser.add_argument("--critical-summary-threshold", type=float, default=0.50)
    parser.add_argument("--critical-tokens-threshold", type=float, default=3000.0)
    parser.add_argument("--critical-latency-threshold", type=float, default=1500.0)
    args = parser.parse_args()

    initDB()

    baseline_prompt = load_prompt(args.baseline_prompt)
    baseline_summary = await run_and_store_eval(args.baseline_prompt, args.dataset, "baseline")
    candidate_summary = await run_and_store_eval(args.candidate_prompt, args.dataset, "candidate")

    warning_alerts = check_latest_accuracy_drop(
        args.warning_category_threshold,
        args.warning_summary_threshold,
        args.warning_tokens_threshold,
        args.warning_latency_threshold,
        baseline_prompt.version_id,
    )
    critical_alerts = check_latest_accuracy_drop(
        args.critical_category_threshold,
        args.critical_summary_threshold,
        args.critical_tokens_threshold,
        args.critical_latency_threshold,
        baseline_prompt.version_id,
    )

    report_path = generate_latest_run_report()
    comment_path = write_pr_comment(
        candidate_summary,
        warning_alerts,
        critical_alerts,
        report_path,
    )

    print(json.dumps({
        "baseline_run_id": baseline_summary["run_id"],
        "candidate_run_id": candidate_summary["run_id"],
        "warning_alerts": warning_alerts,
        "critical_alerts": critical_alerts,
        "report_path": str(report_path),
        "comment_path": str(comment_path),
    }, indent=2))

    return 1 if critical_alerts else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
