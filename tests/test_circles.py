from dotenv import load_dotenv
import instructor
import openai
from pydantic import BaseModel
import pytest
from tqdm.asyncio import tqdm
from examples.image_datasets import generate_image_with_lines
from fewshot import Predictor
from optimizers import GreedyFewShot
from templates import Base64Image, image_to_base64


class ImageInput(BaseModel):
    image: Base64Image


class Answer(BaseModel):
    count: int


@pytest.fixture(scope="module")
def predictor():
    load_dotenv()
    return Predictor(
        instructor.from_openai(openai.AsyncOpenAI()),
        "gpt-4o-mini",
        output_type=Answer,
        optimizer=GreedyFewShot(3),
    )


@pytest.fixture(scope="module")
def dataset():
    dataset = [generate_image_with_lines() for _ in range(10)]
    dataset = [
        (ImageInput(image=image_to_base64(img)), count) for img, count in dataset
    ]
    return dataset


@pytest.mark.asyncio
async def test_train(predictor, dataset):
    with tqdm(predictor.as_completed(dataset, concurrent=len(dataset))) as pbar:
        async for t, (_, actual_count), answer in pbar:
            score = float(actual_count == answer.count)
            t.backwards(expected=Answer(count=actual_count), score=score)