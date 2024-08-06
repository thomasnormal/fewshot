import pytest
import asyncio
from datasets import load_dataset
from pydantic import BaseModel, field_validator, Field, ValidationError
import instructor
import openai
import dotenv
from lib import Predictor, Example
from itertools import islice

class Question(BaseModel):
    """Answer questions with short factoid answers."""
    question: str

class Answer(BaseModel):
    reasoning: str = Field(description="reasoning for the answer")
    answer: str = Field(description="often between 1 and 5 words")


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

@pytest.fixture(scope="module")
def dataset():
    return load_dataset("hotpot_qa", "fullwiki")

@pytest.fixture(scope="module")
def trainset(dataset):
    return [
        (Question(question=x["question"]), x["answer"])
        for x in islice(dataset['train'], 10)  # Using 10 samples for quicker tests
    ]

@pytest.fixture(scope="module")
def client():
    dotenv.load_dotenv()
    return instructor.from_openai(openai.AsyncOpenAI())

@pytest.fixture(scope="module")
def predictor(client):
    return Predictor(client, "gpt-3.5-turbo", output_type=Answer, max_examples=3)

@pytest.mark.asyncio
async def test_predict(predictor, trainset):
    input_question, expected_answer = trainset[0]
    result = await predictor.predict(input_question)
    
    assert isinstance(result, Answer)
    assert isinstance(result.reasoning, str)
    assert isinstance(result.answer, str)

@pytest.mark.asyncio
async def test_as_completed(predictor, trainset):
    results = [result async for result in predictor.as_completed(trainset[:3], max_concurrent=2)]
    
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
    async for (input_question, expected_answer), answer in predictor.as_completed(trainset, max_concurrent=5):
        score = int(answer.answer.lower() == expected_answer.lower())
        predictor.backwards(input_question, answer, score)
        correctness.append(score)
    
    assert len(correctness) == len(trainset)
    accuracy = sum(correctness) / len(correctness)
    print(f"Accuracy: {accuracy:.2f}")

def test_inspect_history(predictor, capsys):
    # Add a message to the log to ensure there's something to inspect
    predictor.message_log.append([
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Test response"}
    ])
    
    predictor.inspect_history()
    captured = capsys.readouterr()
    print("Captured output:", captured.out)  # Debug print
    assert "Conversation" in captured.out
    assert "Test message" in captured.out
    assert "Test response" in captured.out

@pytest.mark.asyncio
async def test_caching(predictor, trainset):
    input_question, _ = trainset[0]
    
    # First call should hit the API
    result1 = await predictor.predict(input_question)
    
    # Second call should use the cache
    result2 = await predictor.predict(input_question)

    print(result1)
    print(result2)
    
    assert result1 == result2


@pytest.fixture(scope="module")
def sample_question():
    return Question(
        prompt=PythonCode(code="def add(a, b):\n    # TODO: implement this function\n    pass"),
        test=PythonCode(code="assert add(1, 2) == 3\nassert add(-1, 1) == 0"),
        entry_point="add"
    )

@pytest.fixture(scope="module")
def sample_answer():
    return Answer(
        reasoning="To add two numbers, we simply use the '+' operator.",
        solution=PythonCode(code="def add(a, b):\n    return a + b")
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
            entry_point=f"func{i}"
        ) for i in range(3)
    ]
    results = [result async for result in predictor.as_completed((q, q) for q in questions)]
    assert len(results) == 3
    for (question, _), answer in results:
        assert isinstance(question, Question)
        assert isinstance(answer, Answer)


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


def test_evaluate_function(sample_question, sample_answer):
    assert evaluate(sample_question, sample_answer) == True

    incorrect_answer = Answer(
        reasoning="This is intentionally wrong.",
        solution=PythonCode(code="def add(a, b):\n    return a - b")
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
            question=f"Implement a function func{i} that multiplies its input by {i}",
            prompt=PythonCode(code=f"def func{i}(x):\n    # TODO: implement this function\n    pass"),
            test=PythonCode(code=f"assert func{i}(2) == {i*2}"),
            entry_point=f"func{i}"
        ) for i in range(1, 4)
    ]
    
    correctness = []
    async for (question, _), answer in predictor.as_completed((q, q) for q in questions):
        score = evaluate(question, answer)
        predictor.backwards(question, answer, score)
        correctness.append(score)
    
    assert len(correctness) == 3
    print(f"Accuracy: {sum(correctness) / len(correctness):.2f}")

if __name__ == "__main__":
    pytest.main(["-v", "-s"])
