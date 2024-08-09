import pytest
from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor
import openai
import dotenv
from itertools import islice

from fewshot import Predictor


@pytest.fixture(scope="module")
def predictor():
    dotenv.load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())
    # Make a predictor without an optimizer
    return Predictor(client, "gpt-4o-mini", output_type=Answer)


class Question(BaseModel):
    """Answer questions with short factoid answers."""

    question: str


class Answer(BaseModel):
    reasoning: str = Field(description="reasoning for the answer")
    answer: str = Field(description="often between 1 and 5 words")


@pytest.fixture(scope="module")
def trainset():
    dataset = load_dataset("hotpot_qa", "fullwiki")
    return [
        (Question(question=x["question"]), x["answer"])
        for x in islice(dataset["train"], 10)  # Using 10 samples for quicker tests
    ]


@pytest.mark.asyncio
async def test_inspect_history(predictor, capsys):
    _ = await predictor.predict(Question(question="What is the capital of France?"))
    predictor.inspect_history()
    captured = capsys.readouterr()
    assert "Conversation" in captured.out
    assert "Answer questions with short factoid answers" in captured.out


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
