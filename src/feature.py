from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from config import EmailClassificationOutput, PromptConfig


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


async def email_classifier(prompt: PromptConfig, user_email:str)->EmailClassificationOutput:
    client = get_client()
    response = await client.responses.create(
        model = prompt.model,
        input = build_input(prompt,user_email)
    )
    return EmailClassificationOutput.model_validate_json(response.output_text)
