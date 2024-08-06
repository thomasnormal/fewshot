import asyncio
from pydantic import BaseModel, Field
import instructor
import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import random
from typing import Literal, Optional
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
import base64

from lib import Predictor, Example, Base64Image, image_to_base64
from optimizers import OptunaFewShot, GreedyFewShot


class ImageInput(BaseModel):
    image: Base64Image = Field(description="Count the number of circles in the image")


class CircleCount(BaseModel):
    count: int = Field(description="The number of circles in the image")


def generate_image_with_circles(width=300, height=300, max_circles=10):
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    num_circles = random.randint(1, max_circles)
    for _ in range(num_circles):
        x = random.randint(20, width - 20)
        y = random.randint(20, height - 20)
        radius = random.randint(10, 30)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=tuple(random.randint(0, 255) for _ in range(3)),
            outline="black",
        )
    draw.rectangle([0.5, 0.5, width - 1, height - 1], outline="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image_to_base64(Image.open(buffer)), num_circles


print("Generating dataset...")
random.seed(32)
dataset = [generate_image_with_circles() for _ in range(400)]
dataset = [(ImageInput(image=img), count) for img, count in dataset]
random.shuffle(dataset)


async def main(client, max_examples: int, optimizer):
    predictor = Predictor(
        client,
        "gpt-4o-mini",
        output_type=CircleCount,
        optimizer=optimizer(max_examples=max_examples),
        system_message="You are an AI trained to count the number of circles in images. Analyze the image and return the count of circles.",
    )

    correctness = []
    with tqdm(total=len(dataset)) as pbar:
        async for (input_data, actual_count), answer in predictor.as_completed(
            dataset, max_concurrent=20
        ):
            score = float(answer.count == actual_count)
            predictor.backwards(input_data, CircleCount(count=actual_count), score)
            correctness.append(score)
            pbar.set_postfix(accuracy=sum(correctness) / len(correctness))
            pbar.update(1)

    final_accuracy = sum(correctness) / len(correctness)
    print(f"Final accuracy: {final_accuracy:.2f}")

    return final_accuracy, predictor


if __name__ == "__main__":
    load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())
    N = 6
    xs = range(N)

    fig, axs = plt.subplots(N, N, figsize=(16, 8))
    fig.suptitle(
        f"FewShot Learning for Images: OptunaFewShot vs GreedyFewShot", fontsize=16
    )
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)

    accuracies_optuna = []
    accuracies_greedy = []

    for i in xs:
        accuracy_optuna, predictor_optuna = asyncio.run(main(client, i, OptunaFewShot))
        accuracy_greedy, predictor_greedy = asyncio.run(main(client, i, GreedyFewShot))
        accuracies_optuna.append(accuracy_optuna)
        accuracies_greedy.append(accuracy_greedy)

        # Plot examples for OptunaFewShot
        examples = predictor_optuna.optimizer.get_best(predictor_optuna.log)
        for j in range(len(examples)):
            img_data = base64.b64decode(examples[j].input.image)
            axs[N - 1 - j, i].imshow(Image.open(io.BytesIO(img_data)))
            axs[N - 1 - j, i].set_title(
                f"{examples[j].output.count}", y=0.98, x=0.9, pad=-8
            )
        for j in range(N):
            axs[j, i].axis("off")

    plot_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
    plot_ax.plot(xs, accuracies_optuna, marker="o", label="OptunaFewShot")
    plot_ax.plot(xs, accuracies_greedy, marker="s", label="GreedyFewShot")
    plot_ax.set_xticks(xs)
    plot_ax.patch.set_alpha(0)
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)

    plot_ax.set_xlabel("Few Shot Max Examples")
    plot_ax.set_ylabel("Accuracy")
    plot_ax.legend()

    plt.savefig("accuracy_vs_max_examples_comparison.png")
    plt.show()