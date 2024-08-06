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
from fewshot import Predictor, Example, Base64Image, image_to_base64


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
from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor, openai, dotenv
import asyncio
from tqdm import tqdm
from itertools import islice

from fewshot import Predictor, Example
from optimizers import OptunaFewShot


class Question(BaseModel):
    """Answer questions with short factoid answers."""

    question: str


class Answer(BaseModel):
    reasoning: str = Field(description="reasoning for the answer")
    answer: str = Field(description="often between 1 and 5 words")


async def main():
    dataset = load_dataset("hotpot_qa", "fullwiki")
    trainset = [
        (Question(question=x["question"]), x["answer"])
        for x in islice(dataset["train"], 100)
    ]

    dotenv.load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())
    pred = Predictor(
        client, "gpt-4o-mini", output_type=Answer, optimizer=OptunaFewShot(3)
    )

    correctness = []
    with tqdm(total=len(trainset)) as pbar:
        async for (input, expected), answer in pred.as_completed(
            trainset, max_concurrent=10
        ):
            score = int(answer.answer == expected)
            pred.backwards(input, answer, score)

            correctness.append(score)
            pbar.set_postfix(correctness=sum(correctness) / len(correctness))
            pbar.update(1)

    pred.inspect_history()


if __name__ == "__main__":
    asyncio.run(main())
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
import pytest
import asyncio
from pydantic import BaseModel, field_validator, Field, ValidationError
import contextlib
import io
import instructor
import openai
from dotenv import load_dotenv

from fewshot import Predictor, Example


# Fixtures and tests
@pytest.fixture(scope="module")
def client():
    load_dotenv()
    return instructor.from_openai(openai.AsyncOpenAI())


@pytest.fixture(scope="module")
def predictor(client):
    return Predictor(client, "gpt-4o-mini", output_type=Answer)


# Define the necessary classes and functions
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


@pytest.fixture(scope="module")
def sample_question():
    return Question(
        prompt=PythonCode(
            code="def add(a, b):\n    # TODO: implement this function\n    pass"
        ),
        test=PythonCode(code="assert add(1, 2) == 3\nassert add(-1, 1) == 0"),
        entry_point="add",
    )


@pytest.fixture(scope="module")
def sample_answer():
    return Answer(
        reasoning="To add two numbers, we simply use the '+' operator.",
        solution=PythonCode(code="def add(a, b):\n    return a + b"),
    )


def test_python_code_validation():
    valid_code = PythonCode(code="def hello(): print('Hello, World!')")
    assert valid_code.code == "def hello(): print('Hello, World!')"

    with pytest.raises(ValidationError):
        PythonCode(code="def invalid_syntax: print('This is not valid Python')")


@pytest.mark.asyncio
async def test_predictor_predict(predictor, sample_question):
    result = await predictor.predict(sample_question)
    assert isinstance(result, Answer)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.solution, PythonCode)


@pytest.mark.asyncio
async def test_predictor_as_completed(predictor):
    questions = [
        Question(
            prompt=PythonCode(code=f"def func{i}():\n    pass"),
            test=PythonCode(code=f"assert func{i}() is None"),
            entry_point=f"func{i}",
        )
        for i in range(3)
    ]
    results = [
        result async for result in predictor.as_completed((q, q) for q in questions)
    ]
    assert len(results) == 3
    for (question, _), answer in results:
        assert isinstance(question, Question)
        assert isinstance(answer, Answer)


def test_evaluate_function(sample_question, sample_answer):
    assert evaluate(sample_question, sample_answer) == True

    incorrect_answer = Answer(
        reasoning="This is intentionally wrong.",
        solution=PythonCode(code="def add(a, b):\n    return a - b"),
    )
    assert evaluate(sample_question, incorrect_answer) == False


@pytest.mark.asyncio
async def test_predictor_backwards(predictor, sample_question, sample_answer):
    initial_log_length = len(predictor.log)
    predictor.backwards(sample_question, sample_answer, 1.0)
    assert len(predictor.log) == initial_log_length + 1
    assert isinstance(predictor.log[-1], Example)
    assert predictor.log[-1].input == sample_question
    assert predictor.log[-1].output == sample_answer
    assert predictor.log[-1].score == 1.0


@pytest.mark.asyncio
async def test_full_pipeline(predictor, sample_question):
    result = await predictor.predict(sample_question)
    score = evaluate(sample_question, result)
    predictor.backwards(sample_question, result, score)

    assert len(predictor.log) > 0
    assert isinstance(predictor.log[-1], Example)
    assert predictor.log[-1].input == sample_question
    assert predictor.log[-1].output == result
    assert predictor.log[-1].score in [0, 1]


@pytest.mark.asyncio
async def test_multiple_questions(predictor):
    questions = [
        Question(
            prompt=PythonCode(
                code=f"def func{i}(x):\n    # TODO: implement this function\n    pass"
            ),
            test=PythonCode(code=f"assert func{i}(2) == {i*2}"),
            entry_point=f"func{i}",
        )
        for i in range(1, 4)
    ]

    correctness = []
    async for (question, _), answer in predictor.as_completed(
        (q, q) for q in questions
    ):
        score = evaluate(question, answer)
        predictor.backwards(question, answer, score)
        correctness.append(score)

    assert len(correctness) == 3
    print(f"Accuracy: {sum(correctness) / len(correctness):.2f}")


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
import pytest
import asyncio
from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor
import openai
import dotenv
from itertools import islice

from fewshot import Predictor, Example


@pytest.fixture(scope="module")
def client():
    dotenv.load_dotenv()
    print("Using OpenAI API key:", openai.api_key)
    return instructor.from_openai(openai.AsyncOpenAI())


@pytest.fixture(scope="module")
def predictor(client):
    return Predictor(client, "gpt-4o-mini", output_type=Answer)


class Question(BaseModel):
    """Answer questions with short factoid answers."""

    question: str


class Answer(BaseModel):
    reasoning: str = Field(description="reasoning for the answer")
    answer: str = Field(description="often between 1 and 5 words")


@pytest.fixture(scope="module")
def trainset():
    dataset = load_dataset("hotpot_qa", "fullwiki")
    return [
        (Question(question=x["question"]), x["answer"])
        for x in islice(dataset["train"], 10)  # Using 10 samples for quicker tests
    ]


@pytest.mark.asyncio
async def test_predict(predictor, trainset):
    input_question, expected_answer = trainset[0]
    result = await predictor.predict(input_question)

    assert isinstance(result, Answer)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.answer, str)


@pytest.mark.asyncio
async def test_as_completed(predictor, trainset):
    results = [result async for result in predictor.as_completed(trainset[:3])]

    assert len(results) == 3
    for (input_question, expected_answer), answer in results:
        assert isinstance(input_question, Question)
        assert isinstance(answer, Answer)


@pytest.mark.asyncio
async def test_backwards(predictor, trainset):
    input_question, expected_answer = trainset[0]
    answer = await predictor.predict(input_question)

    initial_log_length = len(predictor.log)
    predictor.backwards(input_question, answer, 1.0)

    assert len(predictor.log) == initial_log_length + 1
    assert isinstance(predictor.log[-1], Example)
    assert predictor.log[-1].input == input_question
    assert predictor.log[-1].output == answer
    assert predictor.log[-1].score == 1.0


@pytest.mark.asyncio
async def test_full_pipeline(predictor, trainset):
    correctness = []
    async for (input_question, expected_answer), answer in predictor.as_completed(
        trainset, max_concurrent=5
    ):
        score = int(answer.answer.lower() == expected_answer.lower())
        predictor.backwards(input_question, answer, score)
        correctness.append(score)

    assert len(correctness) == len(trainset)
    accuracy = sum(correctness) / len(correctness)
    print(f"Accuracy: {accuracy:.2f}")


@pytest.mark.asyncio
async def test_inspect_history(predictor, capsys):
    answer = await predictor.predict(
        Question(question="What is the capital of France?")
    )
    predictor.inspect_history()
    captured = capsys.readouterr()
    assert "Conversation" in captured.out
    assert "Answer questions with short factoid answers" in captured.out


@pytest.mark.asyncio
async def test_caching(predictor, trainset):
    input_question, _ = trainset[0]
    result1 = await predictor.predict(input_question)
    result2 = await predictor.predict(input_question)
    assert result1 == result2


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
