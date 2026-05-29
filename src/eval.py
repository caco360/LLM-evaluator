from feature import email_classifier
from config import load_prompt
import json 
import asyncio


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

    try:
        async with semaphore:
            model_out = await email_classifier(prompt, test_case["input"])

        result["model_output"] = model_out.model_dump()
        result["status"] = "success"
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




