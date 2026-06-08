from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from config import EmailClassificationOutput, PromptConfig, SummaryScore


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_client() -> AsyncOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=OPENAI_API_KEY)



def build_input(prompt:PromptConfig,user_email)->str:
    few_shot_text = "\n\n".join(
        [
            "Example:\n"
            f"Customer email: {example.input}\n"
            f"Expected JSON: {example.output.model_dump_json()}"
            for example in prompt.few_shot_examples
        ]
    )

    sections = [prompt.system_prompt]
    if few_shot_text:
        sections.append(few_shot_text)
    sections.append(f"Customer email:\n{user_email}")

    return "\n\n".join(sections)


async def email_classifier(prompt: PromptConfig, user_email:str)->dict:
    client = get_client()
    response = await client.responses.create(
        model = prompt.model,
        input = build_input(prompt,user_email)
    )
    usage = response.usage
    parsed_out =  EmailClassificationOutput.model_validate_json(response.output_text)
    return {
        "parsed_out": parsed_out,
        "usage":{
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
    }

async def score_summary (email_text : str, expected_summary: str, actual_summary: str)->SummaryScore:
    client = get_client()
    response = await client.responses.create(
        model = "gpt-4o-mini",
        input = f"""" You are an evaluation judge for a customer support email classifier.

        Your task is to score how accurately the actual summary captures the customer's issue compared to the expected summary and original email.

        Only evaluate the summary. Do not evaluate the category.

        Scoring scale:
        5 = Fully accurate. Captures the main issue with no meaningful omissions or hallucinations.
        4 = Mostly accurate. Captures the main issue but misses a minor detail.
        3 = Partially accurate. Captures some of the issue but misses or distorts an important detail.
        2 = Mostly inaccurate. Mentions the general topic but misses the actual customer issue.
        1 = Inaccurate. Unrelated, misleading, or contradicts the email.

        Return only valid JSON with this exact shape:
        {{
        "score": 1
        }}

        Rules:
        - The score must be an integer from 1 to 5.
        - Do not include explanations.
        - Do not include markdown.
        - Do not include any text outside the JSON.

        Original customer email:
        {email_text}

        Expected summary:
        {expected_summary}

        Actual summary:
        {actual_summary}"""
    )

    return SummaryScore.model_validate_json(response.output_text)

    
