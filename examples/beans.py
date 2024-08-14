import asyncio
from datasets import load_dataset
from pydantic import BaseModel, Field
import instructor
import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import random
from typing import Literal

from fewshot.optimizers import GreedyFewShot
from fewshot import Loss, Predictor
from fewshot.templates import Base64Image, image_to_base64


# Mapping from numeric labels to string labels
LABEL_MAP = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}


class ImageInput(BaseModel):
    image: Base64Image = Field(description="Assess the health status of the bean plant")


class BeanClassification(BaseModel):
    # reasoning: str = Field(None, description="Reasoning for the classification")
    classification: Literal["healthy", "angular_leaf_spot", "bean_rust"]


async def test_beans(predictor, test_subset):
    correctness = []
    with tqdm(predictor.gather(test_subset)) as pbar:
        async for t, (input, expected), answer in pbar:
            score = float(answer.classification == expected.classification)
            t.backwards(score=score)

            correctness.append(score)
            pbar.set_postfix(accuracy=sum(correctness) / len(correctness))
    final_accuracy = sum(correctness) / len(correctness)
    print(f"Final accuracy: {final_accuracy:.2f}")


async def main(client, max_examples: int):
    predictor = Predictor(
        client,
        "gpt-4o-mini",
        output_type=BeanClassification,
        optimizer=(optimizer := GreedyFewShot(max_examples=3)),
        system_message="You are an AI trained to classify the health status of bean plants.",
    )

    print("Loading dataset...")
    dataset = load_dataset("beans", split="train")
    random.seed(42)
    subset = random.sample(list(dataset), 60)
    pairs = [
        (
            ImageInput(image=image_to_base64(item["image"])),
            BeanClassification(classification=LABEL_MAP[item["labels"]]),
        )
        for item in subset
    ]
    train_subset = pairs[:10]
    test_subset = pairs[10:]

    print("Testing the model (without training)...")
    await test_beans(predictor, test_subset)

    # Give the optimizer some initial data to work with
    print("Training the optimizer...")
    for input, output in train_subset:
        optimizer.tell(None, Loss(input=input, output=output, score=1.0))

    print("Testing the model (after training)...")
    await test_beans(predictor, test_subset)


if __name__ == "__main__":
    load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())
    asyncio.run(main(client, max_examples=3))
