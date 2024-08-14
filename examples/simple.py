import argparse
from pydantic import BaseModel
import asyncio
import instructor, openai, dotenv
from dotenv import load_dotenv

from fewshot import Predictor


def parse_arguments():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    return parser.parse_args()


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


async def main():
    args = parse_arguments()
    client = instructor.from_openai(openai.AsyncOpenAI())
    pred = Predictor(client, args.model, output_type=Answer)
    t, answer = await pred.predict(Question(question="What is the capital of France?"))
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
