import instructor
from pydantic import (
    BaseModel,
    Field,
    Secret,
    ValidationError,
    PlainSerializer,
    WithJsonSchema,
)
from typing import TypeVar, Generator, Tuple, Type
import pprint
import asyncio
from typing import Any, Annotated
from termcolor import colored
import diskcache
import base64
from io import BytesIO

from PIL import Image


def image_to_base64(image):
    # image = image.resize((100, 100))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


Base64Image = Annotated[str, "base64image"]


def is_base64_image_field(model: type[BaseModel], field_name: str) -> bool:
    field = model.model_fields[field_name]
    return "base64image" in field.metadata


T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)


def format_input_simple(pydantic_object: T) -> Generator[dict[str, str], None, None]:
    # Note: Doesn't support images
    schema = pydantic_object.model_json_schema()
    if "description" in schema:
        yield {"role": "user", "content": schema["description"]}
    yield {"role": "user", "content": str(pydantic_object)}


def format_input(pydantic_object: T) -> Generator[dict[str, str], None, None]:

    schema = pydantic_object.model_json_schema()

    if "description" in schema:
        yield {"role": "user", "content": schema["description"]}

    properties = schema.get("properties", {})
    for field_name, field_info in pydantic_object.model_fields.items():
        if not field_info.exclude:
            value = getattr(pydantic_object, field_name)
            schema_info = properties.get(field_name, {})
            title = schema_info.get("title", field_name.title())
            desc = schema_info.get("description", "")

            if is_base64_image_field(type(pydantic_object), field_name):
                yield {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{value}"},
                        },
                        {"type": "text", "text": f"{title}: {desc}"},
                    ],
                }
            else:
                yield {
                    "role": "user",
                    "content": f"{title}: {desc}",
                }
                yield {"role": "user", "content": str(value)}


class Example[T: BaseModel, U: BaseModel]:
    def __init__(self, input: T, output: U, score: None | float = None):
        self.input = input
        self.output = output
        self.score = score


class Predictor[T: BaseModel, U: BaseModel]:
    def __init__(
        self,
        client: any,
        model: str,
        output_type: Type[U],
        system_message: str = None,
        format_input=format_input,
        max_retries=5,
        optimizer=None,
        cache_dir="./cache",
    ):
        """
        Initialize the Predictor.

        Args:
            client (any): The client used for making API calls.
            model (str): The name of the model to use for predictions.
            output_type (Type[U]): The Pydantic model type for the output.
            system_message (str, optional): A system message to include in each request.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 5.
            max_examples (int, optional): Maximum number of examples to include in each request. Defaults to 3.
            cache_dir (str, optional): Directory to use for caching. Defaults to "./cache".
        """
        self.client = client
        self.model = model
        self.output_type = output_type
        self.system_message = system_message
        self.format_input = format_input

        self.max_retries = max_retries
        self.optimizer = optimizer or DummyOptimizer()

        self.message_log = []
        self.log = []

        self.cache = diskcache.Cache(cache_dir) or {}

    def _example_to_messages(self, ex: Example[T, U]):
        yield from self.format_input(ex.input)
        yield {"role": "assistant", "content": str(ex.output)}

    async def predict(self, input: T) -> U:
        """
        Make a prediction for the given input.

        Args:
            input (T): The input data, of type T (a Pydantic model).

        Returns:
            U: The prediction result, of type U (a Pydantic model).
        """
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        # Add examples from the optimizer
        for example in self.optimizer.suggest(input, self.log):
            messages.extend(self._example_to_messages(example))

        # Add the current input
        messages.extend(self.format_input(input))

        # Check if the result is already cached
        key = (self.model, self.max_retries, self.output_type, messages)
        if (cached := self.cache.get(key)) is not None:
            output = self.output_type.model_validate_json(cached)

        else:
            output = await self.client.chat.completions.create(
                model=self.model,
                max_retries=self.max_retries,
                response_model=self.output_type,
                messages=messages,
            )
            if self.cache is not None:
                self.cache.set(key, output.model_dump_json())

        messages.append({"role": "assistant", "content": str(output)})
        self.message_log.append(messages)

        return output

    async def as_completed(
        self,
        inputs: list[Tuple[T, Any]],
        max_concurrent=10,
    ) -> Generator[Tuple[Tuple[T, Any], U], None, None]:
        """
        Process multiple inputs concurrently and yield results as they complete.

        Args:
            inputs (list[Tuple[T, Any]]): A list of input tuples, where each tuple contains
                                          an input of type T and any additional data.
            max_concurrent (int, optional): Maximum number of concurrent tasks. Defaults to 10.

        Yields:
            Tuple[Tuple[T, Any], U]: A tuple containing the original input tuple and the prediction result.
        """

        async def wrap(ex: Tuple[T, Any] | T):
            inp, _ = ex
            try:
                res = await self.predict(inp)
            except ValidationError as e:
                res = e
            return ex, res

        pending = set()
        iter_inputs = iter(inputs)

        while pending or inputs:
            while len(pending) < max_concurrent and inputs:
                try:
                    pending.add(asyncio.create_task(wrap(next(iter_inputs))))
                except StopIteration:
                    break

            if not pending:
                break

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                yield await task

    def backwards(self, input: T, output: U, score: float):
        """
        Update the predictor with feedback on a prediction.

        Args:
            input (T): The input data that was used for the prediction.
            output (U): The output (prediction) that was generated.
            score (float): A score indicating the quality of the prediction (typically between 0 and 1).
        """
        # TODO: Right now this doesn't do much. Eventually it'll be more like real backprop.
        self.log.append(Example(input, output, score))
        self.optimizer.step(input, output, score)

    def inspect_history(self, n=1):
        """
        Print the last n conversations from the message log.

        Args:
            n (int, optional): Number of recent conversations to display. Defaults to 1.
        """
        for i, messages in enumerate(self.message_log[-n:]):
            print(colored(f"\n=== Conversation {i+1} ===", "yellow", attrs=["bold"]))
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    print(colored(content, "cyan"))
                elif role == "user":
                    print(colored(content, "green"))
                elif role == "assistant":
                    print(colored(content, "magenta"))

                print()

            print(colored("=" * 50, "yellow"))
