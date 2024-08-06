import asyncio
from datasets import load_dataset
from pydantic import BaseModel, Field
import instructor
import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import random
from typing import Literal, Optional

from optimizers import OptunaFewShot
from fewshot import Predictor, Example
from templates import Base64Image, image_to_base64


# Mapping from numeric labels to string labels
LABEL_MAP = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}


class ImageInput(BaseModel):
    image: Base64Image = Field(description="Assess the health status of the bean plant")


class BeanClassification(BaseModel):
    # reasoning: str = Field(None, description="Reasoning for the classification")
    classification: Literal["healthy", "angular_leaf_spot", "bean_rust"]


print("Loading dataset...")
dataset = load_dataset("beans", split="train")
subset = random.sample(list(dataset), 60)
pairs = [
    (ImageInput(image=image_to_base64(item["image"])), LABEL_MAP[item["labels"]])
    for item in subset
]
train_subset = pairs[:10]
test_subset = pairs[10:]


async def main(client, max_examples: int):

    predictor = Predictor(
        client,
        "gpt-4o",
        output_type=BeanClassification,
        optimizer=OptunaFewShot(max_examples=3),
        system_message="You are an AI trained to classify the health status of bean plants. Classify the image as 'healthy', 'angular_leaf_spot', or 'bean_rust'.",
    )

    for input_data, output_data in train_subset:
        predictor.backwards(input_data, output_data, 1.0)

    correctness = []
    with tqdm(total=len(test_subset)) as pbar:
        async for (input_data, actual_label), answer in predictor.as_completed(
            test_subset, max_concurrent=15
        ):
            score = float(answer.classification == actual_label)
            predictor.backwards(input_data, answer, score)

            correctness.append(score)
            pbar.set_postfix(accuracy=sum(correctness) / len(correctness))
            pbar.update(1)

    final_accuracy = sum(correctness) / len(correctness)
    print(f"Final accuracy: {final_accuracy:.2f}")


if __name__ == "__main__":
    load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())
    asyncio.run(main(client, max_examples=3))
