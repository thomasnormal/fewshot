import pytest
import asyncio
from pydantic import BaseModel, field_validator, Field, ValidationError
import contextlib
import io
import instructor
import openai
from dotenv import load_dotenv

from fewshot import Loss, OptimizationToken, Predictor, Example
from optimizers import GreedyFewShot


# Fixtures and tests
@pytest.fixture(scope="module")
def client():
    load_dotenv()
    return instructor.from_openai(openai.AsyncOpenAI())


@pytest.fixture(scope="module")
def predictor(client):
    return Predictor(
        client, "gpt-4o-mini", output_type=Answer, optimizer=GreedyFewShot(3)
    )


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
        return 0
    except Exception as e:
        return 0
    return 1


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
    t, result = await predictor.predict(sample_question)
    assert isinstance(t, OptimizationToken)
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
    results = [(q, a) async for t, q, a in predictor.as_completed(questions)]
    assert len(results) == 3
    for question, answer in results:
        assert isinstance(question, Question)
        assert isinstance(answer, Answer)


def test_evaluate_function(sample_question, sample_answer):
    assert evaluate(sample_question, sample_answer)

    incorrect_answer = Answer(
        reasoning="This is intentionally wrong.",
        solution=PythonCode(code="def add(a, b):\n    return a - b"),
    )
    assert not evaluate(sample_question, incorrect_answer)


@pytest.mark.asyncio
async def test_full_pipeline(predictor, sample_question):
    t, result = await predictor.predict(sample_question)
    score = evaluate(sample_question, result)
    t.backwards(score=score)

    losses = predictor.optimizer.losses
    assert len(losses) > 0
    assert isinstance(losses[0], Loss)
    assert losses[0].input == sample_question
    assert losses[0].output == result
    assert losses[0].score == score


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
    async for t, question, answer in predictor.as_completed(questions):
        score = evaluate(question, answer)
        t.backwards(score=score)
        correctness.append(score)

    assert len(correctness) == 3
    print(f"Accuracy: {sum(correctness) / len(correctness):.2f}")


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
