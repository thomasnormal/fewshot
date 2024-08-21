import pytest
import dotenv
from fewshot.experimental.tinytextgrad import (
    LMConfig,
    Variable,
    Optimizer,
    complete,
    wikipedia_summary,
    concat,
    equality_loss,
    MultihopModel,
)


@pytest.fixture
def lm_config():
    dotenv.load_dotenv()
    return LMConfig("gpt-4o-mini")  # Using a faster model for tests


@pytest.mark.asyncio
async def test_lm_config_call(lm_config):
    response = await lm_config.call([{"role": "user", "content": "Say 'Test response'"}])
    assert isinstance(response, str)
    assert "Test response" == response


@pytest.mark.asyncio
async def test_variable_backward(lm_config):
    loss = Variable("test")
    # Calling backward on a loss assumes the value of the loss is already feedback
    await loss.backward(lm_config)
    assert loss.feedback == ["test"]


@pytest.mark.asyncio
async def test_optimizer_step(lm_config):
    v1 = Variable("test1")
    v2 = Variable("test2")
    optimizer = Optimizer([v1, v2])

    await optimizer.step(lm_config)
    assert isinstance(v1.value, str)
    assert isinstance(v2.value, str)
    assert v1.value != "test1"  # Ensure the value has been updated
    assert v2.value != "test2"  # Ensure the value has been updated


@pytest.mark.asyncio
async def test_complete(lm_config):
    v = Variable("What is the capital of France?")
    result = await complete(lm_config, question=v)
    assert isinstance(result, Variable)
    assert "Paris" in result.value


@pytest.mark.asyncio
async def test_wikipedia_summary():
    v = Variable("Python (programming language)")
    result = await wikipedia_summary(v)
    assert isinstance(result, Variable)
    assert "programming language" in result.value.lower()


def test_concat():
    v1 = Variable("Hello")
    v2 = Variable("World")
    result = concat([v1, v2])
    assert result.value == "Hello\n\nWorld"
    assert result.preds == [v1, v2]


@pytest.mark.asyncio
async def test_equality_loss(lm_config):
    answer = Variable("Paris")
    expected = Variable("Paris")
    result = await equality_loss(lm_config, answer, expected)
    assert isinstance(result, Variable)
    assert "equal" in result.value.lower() or "same" in result.value.lower()


@pytest.mark.asyncio
async def test_multihop_model(lm_config):
    model = MultihopModel(hops=2)
    question = Variable("What is the capital of the country where the Eiffel Tower is located?")

    result = await model.forward(lm_config, question)
    assert isinstance(result, Variable)
    assert "Paris" in result.value


@pytest.mark.asyncio
async def test_multiple_questions(lm_config):
    model = MultihopModel(hops=2)
    questions = [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the largest planet in our solar system?",
    ]

    for question in questions:
        result = await model.forward(lm_config, Variable(question))
        assert isinstance(result, Variable)
        assert len(result.value) > 0  # Ensure we got a non-empty response


if __name__ == "__main__":
    pytest.main()
