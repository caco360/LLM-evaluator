from feature import email_classifier, score_summary
from config import load_prompt, PromptConfig
import json 
import asyncio
import time
from datetime import datetime, timezone
from db import get_latest_run, get_latest_run_for_prompt_version, get_last_two_runs


def load_golden_dataset(golden_ds_path:str)->list[dict]:
    with open(golden_ds_path,"r",encoding="utf-8") as file:
        data = json.load(file)
    return data

async def run_one_case(test_case: dict, prompt, semaphore: asyncio.Semaphore) -> dict:
    result = {
        "case_id": test_case["id"],
        "input": test_case["input"],
        "expected_output": test_case["expected_output"],
    }

    start_time = time.perf_counter()
    try:
        async with semaphore:
            model_out = await email_classifier(prompt, test_case["input"])
        latency_ms = round((time.perf_counter() - start_time) * 1000,2)
        result["model_output"] = model_out["parsed_out"].model_dump()
        result["usage"]=model_out["usage"]
        result["latency_ms"]= latency_ms
        result["status"] = "success"
        result["category_match"]=(result["model_output"]["category"] == result["expected_output"]["category"])
        async with semaphore:
            summary_score = await score_summary(test_case["input"],test_case["expected_output"]["summary"], result["model_output"]["summary"])
        result["summary_score"] = summary_score.model_dump()["score"]


    except Exception as error:
        result["model_output"] = None
        result["status"] = "error"
        result["error"] = str(error)

    return result

async def evaluate_prompt (prompt_path:str, golden_ds_path : str) -> list[dict]:
    semaphore = asyncio.Semaphore(5)
    prompt = load_prompt(prompt_path)
    golden_ds = load_golden_dataset(golden_ds_path)
    tasks = []
    for test_case in golden_ds:
        task = run_one_case(test_case,prompt, semaphore)
        tasks.append(task)  

    return await asyncio.gather(*tasks)

def run_summary(run_result: list[dict], prompt_path: str, golden_ds_path: str, prompt: PromptConfig)->dict:
    summary ={}
    summary["prompt_version"] = prompt.version_id
    summary["prompt_file"] = prompt_path
    summary["model"] = prompt.model
    summary["dataset_file"] = golden_ds_path
    summary["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["total_cases"] = len(run_result)
    summary["success_count"] = sum(1 for r in run_result if r["status"] == "success")
    summary["error_count"] = sum(1 for r in run_result if r["status"] == "error")
    summary["overall_category_accuracy"] = round(sum(r["category_match"] for r in run_result if r["status"] == "success") / summary["success_count"], 4) if summary["success_count"] > 0 else None
    summary["overall_summary_score"] = round(sum(r["summary_score"] for r in run_result if r["status"] == "success") / summary["success_count"], 4) if summary["success_count"] > 0 else None
    summary["total_tokens"] = sum(r["usage"]["total_tokens"] for r in run_result if r["status"] == "success")
    summary["avg_latency_ms"] = round(sum(r["latency_ms"] for r in run_result if r["status"] == "success") / summary["success_count"], 2) if summary["success_count"] > 0 else None
    return summary

#compares latest run against a configurable baseline prompt version
def check_latest_accuracy_drop(
    category_threshold: float,
    summary_threshold: float,
    tokens_threshold: float,
    latency_threshold: float,
    baseline_prompt_version: str = "support_classifier_v1_balanced",
) -> list[str]:
    latest = get_latest_run()

    if latest is None:
        print("No runs to compare")
        return []

    baseline = get_latest_run_for_prompt_version(
        baseline_prompt_version,
        exclude_run_id=latest["run_id"],
    )

    if baseline is None:
        print(f"No baseline run found for prompt version {baseline_prompt_version}")
        return []

    previous_accuracy = {"overall_category_accuracy": baseline["overall_category_accuracy"], "overall_summary_score": baseline["overall_summary_score"], "total_tokens": baseline["total_tokens"], "avg_latency_ms": baseline["avg_latency_ms"]}
    latest_accuracy = {"overall_category_accuracy": latest["overall_category_accuracy"], "overall_summary_score": latest["overall_summary_score"], "total_tokens": latest["total_tokens"], "avg_latency_ms": latest["avg_latency_ms"]}

    drop = {key: previous_accuracy[key] - latest_accuracy[key] for key in previous_accuracy}
    alerts = []

    if drop["overall_category_accuracy"] >= category_threshold:
        alerts.append(f"Category accuracy dropped by {drop['overall_category_accuracy']:.2%}")
    else:
        print("Category accuracy drop is within threshold")

    if drop["overall_summary_score"] >= summary_threshold:
        alerts.append(f"Summary score dropped by {drop['overall_summary_score']:.2f}")
    else:
        print("Summary score drop is within threshold")

    if (-drop["total_tokens"]) >= tokens_threshold:
        alerts.append(f"Total token usage increased by {-drop['total_tokens']}")
    else:
        print("Total tokens increase is within threshold")

    if (-drop["avg_latency_ms"]) >= latency_threshold:
        alerts.append(f"Average latency increased by {-drop['avg_latency_ms']:.2f} ms")
    else:
        print("Average latency increase is within threshold")

    return alerts
