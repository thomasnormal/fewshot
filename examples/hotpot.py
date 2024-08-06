from datasets import load_dataset
from pydantic import Field, BaseModel
import instructor, openai, dotenv
import asyncio
from tqdm.asyncio import tqdm

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
    trainset = [(Question(question=x["question"]), x["answer"]) for x in dataset["train"]]

    dotenv.load_dotenv()
    client = instructor.from_openai(openai.AsyncOpenAI())
    pred = Predictor(client, "gpt-4o-mini", output_type=Answer, optimizer=OptunaFewShot(3))

    with tqdm(pred.as_completed(trainset), total=len(trainset)) as pbar:
        async for (input, expected), answer in pbar:
            score = int(answer.answer == expected)
            pred.backwards(input, answer, score)
            pbar.set_postfix(correctness=sum(ex.score for ex in pred.log) / len(pred.log))

    pred.inspect_history()


if __name__ == "__main__":
    asyncio.run(main())
