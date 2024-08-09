import argparse
import asyncio
import base64
from pydantic import BaseModel
import instructor
import openai
import anthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import random
import io
import matplotlib.pyplot as plt
from PIL import Image

from examples.image_datasets import (
    generate_image_with_lines,
    generate_image_with_circles,
)
from fewshot import Predictor
from optimizers import (
    OptunaFewShot,
    GreedyFewShot,
    OptimizedRandomSubsets,
    HardCaseFewShot,
    GPCFewShot,
)
from templates import format_input_claude, format_input, Base64Image, image_to_base64


class ImageInput(BaseModel):
    image: Base64Image


class Answer(BaseModel):
    count: int


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default="gpt-4o-mini")
parser.add_argument("-n", type=int, default=40, help="Number of examples")
parser.add_argument(
    "-d", choices=["circles", "lines"], default="circles", help="What to count"
)
args = parser.parse_args()


async def runner(
    pbar, client, model, max_examples: int, Optimizer, dataset, system_message
):
    predictor = Predictor(
        client,
        model=model,
        max_tokens=256,  # Claude needs this for some reason
        max_retries=10,
        formatter=format_input_claude if "claude" in model else format_input,
        output_type=Answer,
        optimizer=Optimizer(max_examples),
        system_message=system_message,
    )

    correctness = []
    async for t, (_, actual_count), answer in predictor.as_completed(dataset):
        # l2 = -((actual_count - answer.count) ** 2)
        score = float(actual_count == answer.count)
        # It would be useful to be able to provide feedback to the optimizer.
        # Such as the "actual answer", or some other kind of message.
        # However, the kind of data needed might depend on the optimized used.
        # So maybe all of this should be part of `optimizer.step`, like in pytorch?
        t.backwards(expected=Answer(count=actual_count), score=score)
        correctness.append(score)
        pbar.set_postfix(accuracy=sum(correctness) / len(correctness))
        pbar.update(1)

    final_accuracy = sum(correctness) / len(correctness)

    return final_accuracy, predictor


async def main():
    load_dotenv()
    if "gpt" in args.model:
        client = instructor.from_openai(openai.AsyncOpenAI())
    elif "claude" in args.model:
        client = instructor.from_anthropic(anthropic.AsyncAnthropic())
    else:
        raise ValueError(f"Unknown model, {args.model}")

    print("Generating dataset...")
    random.seed(32)
    if args.d == "circles":
        generator = generate_image_with_circles
        system_message = "Analyze the image and return the count of circles."
    else:
        generator = generate_image_with_lines
        system_message = "How many times do the blue and red lines intersect?"

    dataset = [generator() for _ in range(args.n)]
    dataset = [
        (ImageInput(image=image_to_base64(img)), count) for img, count in dataset
    ]

    N = 6
    xs = range(N)

    fig, axs = plt.subplots(N, N, figsize=(16, 8))
    fig.suptitle(f"FewShot Learning for Images", fontsize=16)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)

    optimizers = [
        GPCFewShot,
        OptunaFewShot,
        GreedyFewShot,
        OptimizedRandomSubsets,
        HardCaseFewShot,
    ]
    accuracies = {opt.__name__: [] for opt in optimizers}

    for i in xs:
        print(f"Using {i} few-shot examples")
        pbars = [
            tqdm(total=args.n, desc=optimizer.__name__, position=oi)
            for oi, optimizer in enumerate(optimizers)
        ]
        results = await asyncio.gather(
            *[
                runner(pbar, client, args.model, i, optimizer, dataset, system_message)
                for pbar, optimizer in zip(pbars, optimizers)
            ]
        )
        for oi, (accuracy, predictor) in enumerate(results):
            accuracies[predictor.optimizer.__class__.__name__].append(accuracy)

            if oi == 0:
                for j, ex in enumerate(predictor.optimizer.best()):
                    img_data = base64.b64decode(ex.input.image)
                    axs[N - 1 - j, i].imshow(Image.open(io.BytesIO(img_data)))
                    axs[N - 1 - j, i].set_title(
                        f"{ex.output.count}", y=0.98, x=0.9, pad=-8
                    )
                for j in range(N):
                    axs[j, i].axis("off")

        for pbar in pbars:
            pbar.close()
        print()

    plot_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
    for optimizer in optimizers:
        plot_ax.plot(
            xs, accuracies[optimizer.__name__], marker="o", label=optimizer.__name__
        )
    plot_ax.set_xticks(xs)
    plot_ax.patch.set_alpha(0)
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.set_xlabel("Few Shot Max Examples")
    plot_ax.set_ylabel("Accuracy")
    plot_ax.legend()

    plt.savefig("accuracy_vs_max_examples_comparison.png")
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
