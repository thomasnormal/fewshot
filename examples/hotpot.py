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
