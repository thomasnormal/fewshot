# Simple Few-Shot Learning with LLMs

A small [DSPy](https://github.com/stanfordnlp/dspy) clone built on [Instructor](https://python.useinstructor.com/)

## Key Features

- **Pydantic Models**: Robust data validation and serialization using Pydantic.
- **Optimizers**: Includes an Optuna-based few-shot optimizer for hyperparameter tuning.
- **Vision Models**: Easy to tune few-shot prompts, even with image examples.
- **Chat Model Templates**: Uses prompt prefilling to and custom templates to make the most of modern LLM APIs.
- **Asynchronous Processing**: Utilizes `asyncio` for efficient concurrent task handling.

## Usage
```bash
git clone git@github.com:thomasnormal/fewshot.git
cd fewshot
pip install -e .
python examples/simple.py
```

The framework supports various AI tasks. Here's a basic example for question answering:

```python
import instructor
import openai
from datasets import load_dataset
from pydantic import Field, BaseModel
from tqdm.asyncio import tqdm

from fewshot import Predictor
from fewshot.optimizers import OptunaFewShot

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

    client = instructor.from_openai(openai.AsyncOpenAI())  # Use any Instructor supported LLM
    pred = Predictor(client, "gpt-4o-mini", output_type=Answer, optimizer=OptunaFewShot(3))

    async for t, (input, expected), answer in pred.as_completed(trainset):
        score = int(answer.answer == expected)
        t.backwards(score=score)  # Update the model, just like PyTorch

    pred.inspect_history()  # Inpsect the messages sent to the LLM
```

## Example of Few Shot tuning on images
![circles](https://raw.githubusercontent.com/thomasnormal/fewshot/main/static/circles.png)
