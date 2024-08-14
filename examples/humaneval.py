import argparse
from pydantic import BaseModel, field_validator, Field, ValidationError
import asyncio
import instructor
import openai
from dotenv import load_dotenv
import datasets
from tqdm.asyncio import tqdm
import contextlib
import io

from fewshot import Predictor
from fewshot.optimizers import OptunaFewShot


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n-tasks", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--n-examples", type=int, default=1)
    return parser.parse_args()


# We define a pydantic type that automatically checks if it's argument is valid python code.
class PythonCode(BaseModel):
    code: str

    @field_validator("code")
    def check_syntax(cls, v):
        try:
            compile(v, "<string>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Code is not syntactically valid: {e}")
        return v


class Question(BaseModel):
    """Complete the code snippet to pass the test."""

    prompt: PythonCode
    test: PythonCode = Field(exclude=True)
    entry_point: str


class Answer(BaseModel):
    reasoning: str = ""
    solution: PythonCode


def evaluate(input: Question, output: Answer | ValidationError):
    if isinstance(output, ValidationError):
        return 0
    code = output.solution.code + "\n" + input.test.code
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, {})
    except AssertionError:
        return False
    except Exception as e:
        return False
    return True


def load_humaneval(n: int):
    ds = datasets.load_dataset("openai_humaneval")["test"]
    dataset = [
        (
            Question(
                prompt=PythonCode(code=ex["prompt"]),
                test=PythonCode(code=ex["test"]),
                entry_point=ex["entry_point"],
            ),
            Answer(
                solution=PythonCode(code=ex["prompt"] + "\n" + ex["canonical_solution"])
            ),
        )
        for ex in ds
    ]
    if n > 0:
        dataset = dataset[:n]
    return dataset


async def main():
    args = parse_arguments()
    load_dotenv()
    pred = Predictor(
        client=instructor.from_openai(openai.AsyncOpenAI()),
        model=args.model,
        output_type=Answer,
        optimizer=OptunaFewShot(args.n_examples),
        max_retries=args.max_retries,
    )

    # Load the dataset
    print("Loading dataset...")
    dataset = load_humaneval(args.n_tasks)
    train, test = dataset[: args.n_train], dataset[args.n_train :]

    # Sanity check
    assert (
        sum(evaluate(input, output) for input, output in dataset) / len(dataset) == 1.0
    ), "Canonical solutions should all be correct"

    print("Training...")
    correctness = []
    with tqdm(pred.as_completed(train, concurrent=10)) as pbar:
        async for t, (input, expected), answer in pbar:
            score = evaluate(input, answer)
            t.backwards(expected=expected, score=score)
            correctness.append(score)
            pbar.set_postfix(correctness=sum(correctness) / len(correctness))

    print("Evaluating...")
    correctness = []
    with tqdm(pred.as_completed(test, concurrent=100)) as pbar:
        async for t, (input, expected), answer in pbar:
            score = evaluate(input, answer)
            correctness.append(score)
            pbar.set_postfix(correctness=sum(correctness) / len(correctness))


if __name__ == "__main__":
    asyncio.run(main())
