import argparse
import asyncio
import base64
from collections import defaultdict
import re
from typing import Literal
from pydantic import BaseModel, Field
import instructor
import openai
import anthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import datasets
import matplotlib.pyplot as plt
import numpy as np

from fewshot import Predictor
from optimizers import GPCFewShot, OptunaFewShot
from templates import format_input_claude, format_input, Base64Image

# Define types for the various types of tasks in the dataset


class Question(BaseModel):
    image: Base64Image


class SubwayQuestion(BaseModel):
    start: str
    end: str
    image: Base64Image


class AnswerCount(BaseModel):
    count: int


class AnswerYesNo(BaseModel):
    yesno: Literal["Yes", "No"]


class AnswerRowCol(BaseModel):
    rows: int
    columns: int


class AnswerLetter(BaseModel):
    letter: str = Field(min_length=1, max_length=1)


parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default="gpt-4o-mini")
parser.add_argument("-n", type=int, default=40, help="Number of data points")
parser.add_argument("-lo", type=int, default=0, help="lowest number of examples")
parser.add_argument("-hi", type=int, default=5, help="highest number of examples")
args = parser.parse_args()


async def runner(pbar, model, optimizer, dataset, prompt):
    if "gpt" in args.model:
        client = instructor.from_openai(openai.AsyncOpenAI())
    elif "claude" in args.model:
        client = instructor.from_anthropic(anthropic.AsyncAnthropic())
    else:
        raise ValueError(f"Unknown model, {args.model}")

    predictor = Predictor(
        client,
        model=model,
        max_tokens=256,  # Claude needs this for some reason
        max_retries=10,
        formatter=format_input_claude if "claude" in model else format_input,
        output_type=dataset[0][1].__class__,
        optimizer=optimizer,
        system_message=prompt,
    )

    correctness = []
    async for t, (_, expected), answer in predictor.as_completed(dataset):
        if answer.dict() != expected.dict() and str(answer) == str(expected):
            print(f"Expected: {expected}, Got: {answer}")
        score = float(answer.dict() == expected.dict())
        t.backwards(expected=expected, score=score)

        correctness.append(score)
        pbar.set_postfix(accuracy=sum(correctness) / len(correctness))
        pbar.update(1)

    final_accuracy = sum(correctness) / len(correctness)
    return final_accuracy


async def main():
    load_dotenv()

    print("Loading dataset...")
    ds = datasets.load_dataset("XAI/vlmsareblind")["valid"]
    tasks = defaultdict(list)
    prompts = {}
    for d in tqdm(ds.to_list()):
        image = base64.b64encode(d["image"]["bytes"]).decode("utf-8")
        # How many single-color paths go from B to D? Answer with a number in curly brackets e.g. {3}
        if d["task"] == "Subway Connections":
            a, b = re.search(r"go from (\w) to (\w)", d["prompt"]).groups()
            answer = AnswerCount(count=int(d["groundtruth"]))
            prompt = "How many single-color paths go from 'start' to 'end'?"
            question = Question(start=a, end=b, image=image)
        # Count total number of squares in the image. Answer with only the number in numerical format in curly brackets e.g. {3}.
        elif d["task"] == "Nested Squares":
            answer = AnswerCount(count=int(d["groundtruth"]))
            prompt = "Count total number of squares in the image."
            question = Question(image=image)
        # How many times do the blue and red lines touch each other? Answer with a number in curly brackets, e.g., {5}.
        elif d["task"] == "Line Plot Intersections":
            answer = AnswerCount(count=int(d["groundtruth"]))
            prompt = "How many times do the blue and red lines touch each other?"
            question = Question(image=image)
        # Are the two circles touching each other? Answer with Yes/No.
        elif d["task"] == "Touching Circles":
            answer = AnswerYesNo(yesno=d["groundtruth"])
            prompt = "Are the two circles touching each other?"
            question = Question(image=image)
        # Count the number of rows and columns and answer with numbers in curly brackets. For example, rows={5} columns={6}
        elif d["task"] in ("Counting Grid - Word Grids", "Counting Grid - Blank Grids"):
            a, b = re.search(r"(\d+),(\d+)", d["groundtruth"]).groups()
            answer = AnswerRowCol(rows=int(a), columns=int(b))
            prompt = "Count the number of rows and columns."
            question = Question(image=image)
        # How many circles are in the image? Answer with only the number in numerical format.
        # How many pentagons are in the image? Answer with only the number in numerical format.
        elif d["task"].startswith("Olympic Counting"):
            answer = AnswerCount(count=int(d["groundtruth"]))
            prompt = d["prompt"].split("?")[0] + "?"
            question = Question(image=image)
        # Which letter is being circled?
        elif d["task"] == "Circled Letter":
            answer = AnswerLetter(letter=d["groundtruth"])
            prompt = "Which letter is being circled?"
            question = Question(image=image)
        else:
            raise ValueError(f"Unknown task: {d['task']}")

        tasks[d["task"]].append((question, answer))
        prompts[d["task"]] = prompt

    for task, dataset in tasks.items():
        print(f"Task: {task}, Length: {len(dataset)}")

    accuracies = defaultdict(list)
    for nx in range(args.lo, args.hi + 1):
        print(f"Using {nx} few-shot examples")
        # opt = OptunaFewShot(nx)
        opt = GPCFewShot(nx)
        pbars = [
            tqdm(total=min(len(dataset), args.n), desc=task)
            for task, dataset in tasks.items()
        ]
        results = await asyncio.gather(
            *[
                runner(pbar, args.model, opt, tasks[t][: args.n], prompts[t])
                for pbar, t in zip(pbars, tasks)
            ]
        )
        for acc, task in zip(results, tasks):
            accuracies[task].append(acc)
        for pbar in pbars:
            pbar.close()
        print()

    print(accuracies)

    print_table(accuracies)
    plot_bars(accuracies)


def print_table(accuracies):
    print("Task".ljust(30), end="")
    for i in range(args.lo, args.hi + 1):
        print(f"{i}".rjust(10), end="")
    print()
    for task, accs in accuracies.items():
        print(task.ljust(30), end="")
        for acc in accs:
            print(f"{acc:.2f}".rjust(10), end="")
        print()


def plot_bars(accuracies):
    labels = list(accuracies.keys())
    x = np.arange(len(labels))
    width = 0.2
    spacing = 0.1
    group_size = len(accuracies[labels[0]])

    # Base colors for each group
    base_colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots()
    for i, (nx, accs) in enumerate(zip(range(group_size), zip(*accuracies.values()))):
        # Apply a slight variation to the base color for each bar in the group
        variation = 0.3 * (1 - i / group_size)
        colors = base_colors * (1 - variation) + variation
        ax.bar(
            x * spacing + (x * group_size + i) * width,
            accs,
            width,
            color=colors,
            label=f"{nx} examples",
        )

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x * (width * group_size + spacing) + width * (group_size - 1) / 2)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
