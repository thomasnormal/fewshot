import dotenv
import instructor
import openai
from pydantic import BaseModel, Field, create_model
import pytest
import string
import random

from fewshot import Predictor
from fewshot.utils import signature


model = "gpt-4o-mini"


@pytest.fixture(scope="module")
def client():
    dotenv.load_dotenv()
    return instructor.from_openai(openai.AsyncOpenAI())


class Question(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_cache(client):
    class Answer(BaseModel):
        number: int = Field(ge=0, lt=1000)

    pred = Predictor(
        client,
        model,
        output_type=Answer,
        system_message="You are a random number generator.",
        verbose=True,
    )
    _, a1 = await pred.predict()
    _, a2 = await pred.predict()
    _, a3 = await pred.predict()
    i1, i2, i3 = a1.number, a2.number, a3.number
    assert i1 == i2 == i3, "All numbers should be the same"

    pred = Predictor(
        client,
        model,
        output_type=Answer,
        system_message="You are a random number generator.",
        cache=None,
        verbose=True,
    )
    _, a1 = await pred.predict()
    _, a2 = await pred.predict()
    _, a3 = await pred.predict()
    i1, i2, i3 = a1.number, a2.number, a3.number
    assert len(set([i1, i2, i3])) == 3, "All numbers should be different"


@pytest.mark.asyncio
async def test_cache(client):
    questions_and_answers = [
        ("What is the capital of France?", "Paris"),
        ("What is the boiling point of water?", "100°C"),
        ("Who wrote 'Hamlet'?", "William Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many continents are there?", "Seven"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the square root of 64?", "Eight"),
        ("What is the currency of Japan?", "Yen"),
        ("How many states are in the USA?", "Fifty"),
    ]
    data = [(Question(text=q), a) for q, a in questions_and_answers]

    # Avoid cache on the first run
    seed = "".join(random.choices(string.ascii_lowercase, k=10))

    pred = Predictor(
        client,
        model,
        output_type=create_model("Answer", answer=(str, ...)),
        verbose=True,
        system_message=seed,
    )

    # First time we run, there should be no cache hits
    acc = 0
    async for t, (input, expected), answer in pred.gather(data, concurrent=3):
        acc += int(expected.lower() in answer.answer.lower())
    assert pred.cache_hits == 0
    assert acc >= 5

    # Second time we run, everything should be a cache hit
    async for t, (input, expected), answer in pred.gather(data, concurrent=3):
        pass
    assert pred.cache_hits == len(data)
