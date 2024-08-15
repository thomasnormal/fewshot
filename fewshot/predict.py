from collections import defaultdict
import heapq
from pprint import pprint
import instructor
from pydantic import BaseModel, ValidationError
from typing import NamedTuple, TypeVar, Tuple, Type
import asyncio
from typing import Any
import tenacity
from termcolor import colored
import diskcache
from itertools import chain, repeat

from .templates import format_input, format_input_simple


T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)


class Example[T: BaseModel, U: BaseModel](NamedTuple):
    input: T
    output: U


class Loss[T: BaseModel, U: BaseModel](BaseModel):
    input: T | None = None
    output: U
    expected: U | None = None
    score: float | None = None
    feedback: str | None = None


class MessageLogItem[T: BaseModel, U: BaseModel](NamedTuple):
    input: T
    output: U
    messages: list[dict]


class OptimizationToken:
    def __init__(self, index, predictor, opt_token):
        self.index = index
        self.predictor = predictor
        self.opt_token = opt_token

    def backwards(self, expected: U = None, score: float = None, feedback: str = None):
        input, output, _ = self.predictor.message_log[self.index]
        if self.predictor.optimizer is not None:
            self.predictor.optimizer.tell(
                self.opt_token,
                Loss(
                    input=input,
                    output=output,
                    expected=expected,
                    score=score,
                    feedback=feedback,
                ),
            )


class Predictor[T: BaseModel, U: BaseModel]:
    def __init__(
        self,
        client: any,
        model: str,
        output_type: Type[U],
        system_message: str = None,
        formatter=format_input_simple,
        max_retries=5,
        optimizer=None,
        verbose=False,
        cache=diskcache.Cache("./cache"),
        **llm_kwargs,
    ):
        """
        Initialize the Predictor.

        Args:
            client (any): The client used for making API calls.
            model (str): The name of the model to use for predictions.
            output_type (Type[U]): The Pydantic model type for the output.
            system_message (str, optional): A system message to include in each request.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 5.
            cache_dir (str, optional): Directory to use for caching. Defaults to "./cache".
        """
        self.client = client
        self.model = model
        self.output_type = output_type
        self.system_message = system_message
        self.formatter = formatter
        self.llm_kwargs = llm_kwargs
        self.verbose = verbose
        self.cache = cache
        self.training = True
        self.cache_hits = 0

        self.max_retries = max_retries
        self.optimizer = optimizer

        self.message_log = []
        self.token_count = 0

    def train(self):
        self.training = True

    def test(self):
        self.training = False

    def _example_to_messages(self, ex: Example[T, U]):
        yield self.formatter(ex.input)
        yield {"role": "assistant", "content": ex.output.model_dump_json()}

    async def predict(self, input: T | None = None) -> Tuple[U, Any]:
        """
        Make a prediction for the given input.

        Args:
            input (T): The input data, of type T (a Pydantic model).

        Returns:
            U: The prediction result, of type U (a Pydantic model).
        """
        messages = []

        system_messages = []
        if self.system_message:
            system_messages.append({"type": "text", "text": self.system_message})
        if input is not None:
            system_messages.append({"type": "text", "text": "Input schema:"})
            system_messages.append({"type": "text", "text": str(input.model_json_schema())})
        messages.append({"role": "system", "content": system_messages})

        # Add examples from the optimizer
        opt_token = None
        if self.optimizer is not None:
            if self.training:
                examples = self.optimizer.best()
            else:
                examples, opt_token = self.optimizer.suggest()
            for example in examples:
                messages.extend(self._example_to_messages(example))

        # Add the current input
        if input is not None:
            messages.append(self.formatter(input))

        if self.verbose:
            print("\n--- Messages ---")
            pprint(messages)
            print("--- End Messages ---\n")

        # Check if the result is already cached
        output = None
        if self.cache is not None:
            key = (
                self.model,
                self.max_retries,
                self.output_type.model_json_schema(),
                str(messages),
            )
            if (cached := self.cache.get(key)) is not None:
                output = self.output_type.model_validate_json(cached)
                self.cache_hits += 1

        if output is None:
            output = await self.client.chat.completions.create(
                model=self.model,
                max_retries=self.max_retries,
                response_model=self.output_type,
                messages=messages,
                **self.llm_kwargs,
            )
            if self.cache is not None:
                self.cache[key] = output.model_dump_json()

        messages.append({"role": "assistant", "content": output.model_dump_json()})
        self.message_log.append(MessageLogItem(input, output, messages))

        token = OptimizationToken(self.token_count, self, opt_token)
        self.token_count += 1

        return token, output

    def as_completed(self, inputs: list[T | tuple[T, ...]], concurrent=10, epochs=1) -> "PredictionWorker":
        """Process multiple inputs concurrently and yield results as they complete."""
        return PredictionWorker(self, inputs, concurrent, as_completed=True, epochs=epochs)

    def gather(self, inputs: list[T | tuple[T, ...]], concurrent=10, epochs=1) -> "PredictionWorker":
        """Process multiple inputs concurrently and yield them in order."""
        return PredictionWorker(self, inputs, concurrent, as_completed=False, epochs=epochs)

    def inspect_history(self, n=1):
        """
        Print the last n conversations from the message log.

        Args:
            n (int, optional): Number of recent conversations to display. Defaults to 1.
        """
        for i, (input, output, messages) in enumerate(self.message_log[-n:]):
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


class PredictionWorker:
    def __init__(
        self,
        predictor,  # Instance of the Predictor class
        inputs: list[T | tuple[T, ...]],
        concurrent: int = 10,
        as_completed: bool = False,
        epochs: int = 1,
    ):
        self.predictor = predictor
        self.inputs = tuple(inputs)
        self.iter_inputs = chain.from_iterable(repeat(self.inputs, epochs))
        self.concurrent = concurrent
        self.as_completed = as_completed
        self.epochs = epochs

        self.started_count = 0
        self.completed_count = 0
        self.output_heap = []
        self.pending = set()

    def __len__(self):
        return len(self.inputs) * self.epochs

    async def __anext__(self) -> tuple[OptimizationToken, T, Any]:
        # Generate some results if none are available, or if we are using gather()
        # and the next result is not done yet.
        while not self.output_heap or (
            not self.as_completed and self.output_heap[0][0] != self.completed_count
        ):
            await self._generate_some()
        index, token, ex, res = heapq.heappop(self.output_heap)
        self.completed_count += 1
        return token, ex, res

    def __aiter__(self):
        return self

    async def _wrap(self, ex: Tuple[T, Any], index: int):
        # We support both single inputs and tuples of inputs
        inp = ex[0] if isinstance(ex, tuple) else ex
        token = None
        try:
            token, res = await self.predictor.predict(inp)
        except ValidationError as e:
            res = e
        return ex, res, token, index

    async def _generate_some(self):
        if self.completed_count == len(self.inputs):
            raise StopAsyncIteration

        # Start more tasks, up to the concurrency limit
        while len(self.pending) < self.concurrent:
            try:
                input, index = next(self.iter_inputs), self.started_count
                task = asyncio.create_task(self._wrap(input, index))
                self.started_count += 1
                self.pending.add(task)
            except StopIteration:
                break

        if not self.pending:
            raise StopAsyncIteration

        # Wait for some tasks to complete
        done, self.pending = await asyncio.wait(self.pending, return_when=asyncio.FIRST_COMPLETED)

        # Add them to the (priority) queue for __anext__ to return
        for task in done:
            try:
                ex, res, token, index = await task
            except ValidationError as e:
                print("Validation error:", e)
                continue
            except tenacity.RetryError as e:
                print("RetryError error:", e)
                continue
            except instructor.exceptions.InstructorRetryException as e:
                print("InstructorRetryException error:", e)
                continue
            heapq.heappush(self.output_heap, (index, token, ex, res))
