from feature import email_classifier, score_summary
from config import load_prompt
import json 
import asyncio
import time


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




