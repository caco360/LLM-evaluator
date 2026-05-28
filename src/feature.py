from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from config import EmailClassificationOutput, PromptConfig, load_prompt


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=OPENAI_API_KEY)



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


def email_classifier(prompt_path:str, user_email:str)->EmailClassificationOutput:
    prompt = load_prompt(prompt_path)
    response = client.responses.create(
        model = prompt.model,
        input = build_input(prompt,user_email)
    )
    return EmailClassificationOutput.model_validate_json(response.output_text)