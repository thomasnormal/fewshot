import pytest
from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor
import openai
import dotenv
import asyncio

from typing import Union, Optional, Set, Tuple

from fewshot import Predictor


@pytest.fixture(scope="module")
def client():
    dotenv.load_dotenv()
    # instructor.Mode.TOOLS,
    # instructor.Mode.JSON,
    # instructor.Mode.FUNCTIONS,
    # instructor.Mode.PARALLEL_TOOLS,
    # instructor.Mode.MD_JSON,
    return instructor.from_openai(
            openai.AsyncOpenAI(),
            #mode=instructor.Mode.MD_JSON,
            )

class Question(BaseModel):
    """ Return some interesting data that fits into the schema """


class Answer1(BaseModel):
    answer: dict[str, int]

class Answer2(BaseModel):
    answer: dict[str, str]

class Answer3(BaseModel):
    answer: list[list[float]]

class Answer4(BaseModel):
    answer: Union[int, str, float]

class Answer5(BaseModel):
    answer: Set[str]

class Answer6(BaseModel):
    answer: Tuple[int, float, str]

class Answer7(BaseModel):
    answer: Optional[dict[str, list[int]]]

class Answer8(BaseModel):
    answer: list[Optional[float]]

class Answer9(BaseModel):
    answer: dict[str, Optional[list[Tuple[int, str]]]]

class Answer10(BaseModel):
    answer: list[Union[str, dict[str, int]]]

@pytest.mark.asyncio
@pytest.mark.parametrize("AnswerType", [
    Answer1, Answer2, Answer3, Answer4, Answer5, Answer6, 
    Answer7, Answer8, Answer9, Answer10
])
async def test_output_types(client, AnswerType):
    pred = Predictor(client, "gpt-4o-mini", output_type=AnswerType)
    t, result = await pred.predict(Question())
    assert isinstance(result, AnswerType)

