import pytest
import asyncio
from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor
import openai
import dotenv
from itertools import islice

from fewshot import Predictor, Example


@pytest.fixture(scope="module")
def client():
    dotenv.load_dotenv()
    print("Using OpenAI API key:", openai.api_key)
    return instructor.from_openai(openai.AsyncOpenAI())


@pytest.fixture(scope="module")
def predictor(client):
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
async def test_predict(predictor, trainset):
    input_question, expected_answer = trainset[0]
    result = await predictor.predict(input_question)

    assert isinstance(result, Answer)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.answer, str)


@pytest.mark.asyncio
async def test_as_completed(predictor, trainset):
    results = [result async for result in predictor.as_completed(trainset[:3])]

    assert len(results) == 3
    for (input_question, expected_answer), answer in results:
        assert isinstance(input_question, Question)
        assert isinstance(answer, Answer)


@pytest.mark.asyncio
async def test_backwards(predictor, trainset):
    input_question, expected_answer = trainset[0]
    answer = await predictor.predict(input_question)

    initial_log_length = len(predictor.log)
    predictor.backwards(input_question, answer, 1.0)

    assert len(predictor.log) == initial_log_length + 1
    assert isinstance(predictor.log[-1], Example)
    assert predictor.log[-1].input == input_question
    assert predictor.log[-1].output == answer
    assert predictor.log[-1].score == 1.0


@pytest.mark.asyncio
async def test_full_pipeline(predictor, trainset):
    correctness = []
    async for (input_question, expected_answer), answer in predictor.as_completed(
        trainset, max_concurrent=5
    ):
        score = int(answer.answer.lower() == expected_answer.lower())
        predictor.backwards(input_question, answer, score)
        correctness.append(score)

    assert len(correctness) == len(trainset)
    accuracy = sum(correctness) / len(correctness)
    print(f"Accuracy: {accuracy:.2f}")


@pytest.mark.asyncio
async def test_inspect_history(predictor, capsys):
    answer = await predictor.predict(
        Question(question="What is the capital of France?")
    )
    predictor.inspect_history()
    captured = capsys.readouterr()
    assert "Conversation" in captured.out
    assert "Answer questions with short factoid answers" in captured.out


@pytest.mark.asyncio
async def test_caching(predictor, trainset):
    input_question, _ = trainset[0]
    result1 = await predictor.predict(input_question)
    result2 = await predictor.predict(input_question)
    assert result1 == result2


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
