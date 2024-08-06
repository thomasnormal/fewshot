# Simple Few-Shot Learning with LLMs

A small [DSPy](https://github.com/stanfordnlp/dspy) clone built on [Instructor](https://python.useinstructor.com/)

## Key Features

- **Pydantic Models**: Robust data validation and serialization using Pydantic.
- **Optimizers**: Includes an Optuna-based few-shot optimizer for hyperparameter tuning.
- **Vision Models**: Easy to tune few-shot prompts, even with image examples.
- **Chat Model Templates**: Uses prompt prefilling to and custom templates to make the most of modern LLM APIs.
- **Asynchronous Processing**: Utilizes `asyncio` for efficient concurrent task handling.

## Usage

The framework supports various AI tasks. Here's a basic example for question answering:

```python
from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor, openai
from tqdm.asyncio import tqdm

from fewshot import Predictor, Example
from optimizers import OptunaFewShot

# DSPy inspired Pydantic classes for inputs.
class Question(BaseModel):
    """Answer questions with short factoid answers."""
    question: str

class Answer(BaseModel):
    reasoning: str = Field(description="reasoning for the answer")
    answer: str = Field(description="often between 1 and 5 words")


async def main():
    dataset = load_dataset("hotpot_qa", "fullwiki")
    trainset = [(Question(question=x["question"]), x["answer"]) for x in dataset["train"]]

    # Use any Instructor supported LLM
    client = instructor.from_openai(openai.AsyncOpenAI())
    pred = Predictor(client, "gpt-4o-mini", output_type=Answer, optimizer=OptunaFewShot(3))

    with tqdm(pred.as_completed(trainset), total=len(trainset)) as pbar:
        async for (input, expected), answer in pbar:
            score = int(answer.answer == expected)
            # PyTorch Inspired backwards function
            pred.backwards(input, answer, score)
            pbar.set_postfix(correctness=sum(ex.score for ex in pred.log) / len(pred.log))

    pred.inspect_history()
```

## Example of Few Shot tuning on images
