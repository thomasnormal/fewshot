import dotenv
import instructor
import openai
from pydantic import BaseModel, Field
import pytest

from fewshot import Predictor


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
        cache_dir=None,
        verbose=True,
    )
    _, a1 = await pred.predict()
    _, a2 = await pred.predict()
    _, a3 = await pred.predict()
    i1, i2, i3 = a1.number, a2.number, a3.number
    assert len(set([i1, i2, i3])) == 3, "All numbers should be different"
