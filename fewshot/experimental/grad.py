import asyncio
from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel, Field, create_model


T = TypeVar("T")
U = TypeVar("U")


@dataclass
class LMConfig:
    model: str
    max_retries: int = 3
    llm_kwargs: dict[str, any] = None
    cache = None
    template = None
    message_log = None

    # examples here?
    def Function(self, input_type: "Variable", output_type: "Variable") -> "Variable":
        def inner(input: "Variable") -> "Variable":
            return LMCall(input, input_type, output_type, self)

        return inner


class Module:
    pass


class Variable:
    inner_type: Type[BaseModel]
    purpose: str
    feedback: list[str]  # Should we keep previous feedback? So we can make few-shot examples?
    preds: dict[str, "Variable"]
    requires_grad = False  # Used by the optimizer only

    def __getattr__(self, key: str):
        inner_type = self.__dict__.get("inner_type", None)  # Access directly from the instance's dictionary
        assert issubclass(inner_type, BaseModel), f"Variable should wrap a Pydantic model, not {type}"
        if key in type.model_fields:
            return VariableView(self, key)

    async def unwrap(self) -> T:
        # This is basically the forward pass.
        # Should we just call it that?
        raise NotImplementedError

    def add_feedback(self, feedback: str):
        self.feedback.append(feedback)

    async def push_feedback(self, config=None):
        # This default feedback mechanism uses an LLM to assign the credit of the feedback to each
        # input variable.
        # TODO: If we want to do "minibatch" optimization, we should be able to show more than one
        # previous input at a time, and we should keep track of feedback from multiple rounds too.

        # We want to prompt using something like:
        # User: input_0
        # Agent: output_0
        # User: feedback_0
        # User: input_1
        # Agent: ...

        input = create_model(
            "FunctionCall",
            **{k: (type, Field(description=v.purpose)) for k, v in self.preds.items()}
            | {"output": (self.type, Field(description=self.purpose)), "feedback": list[str]},
        )(
            # Note: We're assuming these unwrappings will be cached
            **{k: await child.unwrap() for k, child in self.preds.items()},
            output=await self.unwrap(),
            feedback=self.feedback,
        )
        output_type = create_model("Feedback", **{k: (str, ...) for k in self.preds.keys()})
        output = await LMCall(input, output_type, config).unwrap()
        for k, v in self.preds.items():
            v.add_feedback(getattr(output, k))


class VariableView(Variable):
    """A view of an attribute of a variable."""

    def __init__(self, parent, key):
        self.parent = parent
        self.preds = [parent]
        self.key = key
        self.type = parent.model_fields[key].annotation

    async def unwrap(self):
        parent_val = await self.parent.unwrap()
        return getattr(parent_val, self.key)

    def view(self, k):
        assert issubclass(self.inner_type, BaseModel)
        return VariableView(self, k)

    async def push_feedback(self, config=None):
        # This is an example of a scenario where we can do the credit assignment directly
        for feedback in self.feedback:
            self.parent.add_feedback(f"Feedback for {self.key}: {feedback}")


class Make(Variable):
    def __init__(self, inner_type, **kwargs: Variable):
        self.preds = kwargs
        self.inner_type = inner_type

    async def unwrap(self):
        values = await asyncio.gather(v.unwrap() for v in self.preds.values())
        return self.inner_type(**{k: v for k, v in zip(self.preds.keys(), values)})

    def view(self, key) -> Variable:
        return self.preds[key]


class SchemaVariable(Variable):
    def __init__(self, inner_type):
        print(type(inner_type))
        assert issubclass(inner_type, BaseModel), f"Variable should wrap a Pydantic model, not {inner_type}"
        self.inner_type = inner_type
        self.purpose = "A prompt used to define a task"
        self.preds = []

    def __call__(self, **kwargs: Variable):
        return Make(self.inner_type, **kwargs)

    async def push_feedback(self, config=None):
        # Actually nothing should happen here, since there are no predecessors.
        # The optimizer is the one that's supposed to take care of changing the value.
        pass


class LMCall(Variable):
    # It's not that an LMcall _is_ a variable, it's just a function that returns
    # a tensor. But the tensor needs to be coupled with a particular push_feedback
    # method, so it's easier to just keep it

    def __init__(self, input: Variable, input_type: Variable, output_type: Variable, config: LMConfig):
        assert input.inner_type == self.input_type.inner_type
        self.inner_type = output_type.inner_type
        self.preds = [input, input_type, output_type]
        self.config = config
        self.value = None
        self.lock = asyncio.Lock()

    async def unwrap(self):
        # Make sure that we don't have two callers sending the API call at the
        # same time.
        async with self.lock:
            if self.value is not None:
                return self.value
            input, output_type = asyncio.gather(*[self.input.unwrap(), self.inner_type.unwrap()])
            # TODO: Where do I get few-shot examples from?
            messages = self.config.template(input)
            output = await self.client.chat.completions.create(
                model=self.config.model,
                max_retries=self.config.max_retries,
                response_model=await output_type,
                messages=messages,
                **self.config.llm_kwargs,
            )
            self.value = output
            # TODO: Handle cache, message_log, etc.
            return output

    async def push_feedback(self, config=None):
        pass
        # The input and the output_type should get textual feedback.
        # But we also need to give feedback to the few-shot examples,
        # assuming we are actually using them? Maybe for this "tiny-textgrad"
        # we'll just skip few-shots?
        # I think we can just make a custom push-feedback here, that sends
        # the "Examples" input type the (input, output) pair, which is the
        # "gradient" that it needs.


def var(type_name: str, **fields) -> Type[BaseModel]:
    """Helper function to create a Pydantic model and wrap it in a Variable."""

    fields = {k: (t if isinstance(t, tuple) else (t, ...)) for k, t in fields.items()}
    pydantic_type = create_model(type_name, **fields)

    variable = SchemaVariable(pydantic_type)

    def with_description(description: str) -> Type[BaseModel]:
        pydantic_type.__doc__ = description
        return SchemaVariable(pydantic_type)

    variable.with_description = staticmethod(with_description)
    return variable


class Model(Module):
    def __init__(self, lm: LMConfig, steps: int):
        self.Question = var("Question", question=str, context=list[str])
        self.WikipediaSearch = lm.Function(self.Question, var("WikipediaSearch", resoning=str, query=str))
        self.FinalAnswer = lm.Function(self.Question, var("Answer", reasoning=str, answer=str))
        self.Query = var("WikipediaSearch", query=str)
        self.SearchEngine = lm.Function(self.Query, var("SearchResult", page_title=str, content=str))
        self.steps = steps

    def forward(self, question: str) -> Variable:
        context = []
        for _ in range(self.steps):
            # Make a question, using the current context
            input = self.Question(question=question, context=context)
            query_string = self.WikipediaSearch(input).query  # This will be a wrapped string

            # Search Wikipedia (we immitate being a search engine)
            query = self.Query(query=query_string)
            content = self.SearchEngine(query).content
            context.append(content)

        # Final answer to the question based on the context
        return self.FinalAnswer(self.Question(question=question, context=context)).answer


async def main():
    lm = LMConfig("gpt-4o-mini")
    # These variables will store the "parameters" (prompts) of the model:
    model = Model(lm, steps=2)
    # A: She made her film debut in the 1995 teen drama "Kids".
    answer = await model.forward("What was Rosario Dawson of Josie and the Pussycats film debut").unwrap()
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
