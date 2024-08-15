from PIL import Image, ImageDraw
import instructor
import openai
import langfuse.openai
from pydantic import BaseModel, Field
import asyncio
import dotenv

from fewshot import Predictor
from fewshot.templates import PILImage


def circle(color):
    img = Image.new("RGB", (200, 200), color="white")
    ImageDraw.Draw(img).ellipse([50, 50, 150, 150], fill=color)
    return PILImage(img)


class Question(BaseModel):
    """Which of the following images is most like the query image?"""

    query: PILImage
    options: dict[str, PILImage] = Field(description="Images and their names")


class Answer(BaseModel):
    reasoning: str
    image_name: str


async def main():
    dotenv.load_dotenv()
    client = instructor.patch(langfuse.openai.AsyncOpenAI())
    pred = Predictor(client, model="gpt-4o", output_type=Answer)

    images = {"a": circle("red"), "b": circle("blue"), "c": circle("green")}
    question = Question(query=images["a"], options=images)

    _, answer = await pred.predict(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
