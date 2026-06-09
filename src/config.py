from typing import Literal
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

Category = Literal["billing", "technical", "general", "account", "irrelevant"]

class EmailClassificationOutput(BaseModel):
    category: Category
    summary: str
    
class FewShotExample(BaseModel):
    input: str
    output: EmailClassificationOutput

class PromptConfig(BaseModel):
    version_id: str
    created_at: str
    description: str
    model: str
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

class SummaryScore(BaseModel):
    score : int




#Convert the yaml prompt file to PromptConfig object
def load_prompt(prompt_path: str | Path) -> PromptConfig:
    path = Path(prompt_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return PromptConfig.model_validate(data)



