import argparse
from pydantic import BaseModel, field_validator, Field, ValidationError
import asyncio
import instructor, openai, dotenv
from dotenv import load_dotenv
import datasets
from tqdm.asyncio import tqdm
from typing import Annotated
import contextlib, io

from fewshot import Predictor, Example


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n-examples", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=1)
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
    prompt: PythonCode
    test: PythonCode = Field(exclude=True)
    entry_point: str


class Answer(BaseModel):
    reasoning: str = ""
    solution: PythonCode


def evaluate(input: Question, output: Answer):
    code = output.solution.code + "\n" + input.test.code
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, {})
    except AssertionError:
        return False
    except Exception as e:
        return False
    return True


async def run(pred, exs, examples=(), args=None):
    correctness = []
    with tqdm(total=len(exs)) as pbar:
        async for (input, ex), answer in pred.as_completed(
            (ex.input, ex) for ex in exs
        ):
            if isinstance(answer, ValidationError):
                score = 0
            else:
                score = evaluate(input, answer)
            pred.backwards(input, answer, score)

            correctness.append(score)
            pbar.set_postfix(correctness=sum(correctness) / len(correctness))
            pbar.update(1)


async def main():
    args = parse_arguments()
    load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())

    # Load the dataset
    print("Loading dataset...")
    ds = datasets.load_dataset("openai_humaneval")
    inputs = [
        Question(
            prompt=PythonCode(code=ex["prompt"]),
            test=PythonCode(code=ex["test"]),
            entry_point=ex["entry_point"],
        )
        for ex in ds["test"]
    ]
    if args.n_tasks > 0:
        inputs = inputs[: args.n_tasks]

    # Create examples from canonical solutions given in the dataset
    examples = [
        Example(
            input,
            Answer(solution=PythonCode(code=ex["prompt"] + ex["canonical_solution"])),
        )
        for input, ex in zip(inputs, ds["test"])
    ]
    score = sum(evaluate(ex.input, ex.output) for ex in examples) / len(examples)
    print(f"Canonical score: {score}")

    # Train the model
    if args.n_train > 0:
        pred = Predictor(client, args.model, output_type=Answer)
        print("Training...")
        await run(pred, examples[: args.n_train], args=args)
        given_examples = [ex for ex in pred.log if ex.score is True]
        print(f"Generated {len(given_examples)} good examples")
    else:
        given_examples = examples[: args.n_examples]

    # Evaluate the model
    print("Evaluating...")
    pred = Predictor(client, args.model, output_type=Answer)
    await run(pred, examples, examples=given_examples, args=args)


if __name__ == "__main__":
    asyncio.run(main())
