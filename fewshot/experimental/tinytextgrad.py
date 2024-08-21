import asyncio
from concurrent.futures import ThreadPoolExecutor
import pydantic


################################################################################
# Fundemental classes
################################################################################


class LMConfig:
    """Simple LLM API wrapper."""

    def __init__(self, model: str):
        self.model = model
        if model.startswith("gpt"):
            from openai import AsyncOpenAI
            # from langfuse.openai import AsyncOpenAI

            self.client = AsyncOpenAI()

    async def call(self, messages, **kwargs) -> str | dict:
        """Call the language model with the given messages. Performs CoT if no response_format is specified."""
        cot = False
        if "response_format" not in kwargs:
            kwargs["response_format"] = pydantic.create_model(
                "Response",
                thoughts=(str, pydantic.Field(..., description="Summarize your thoughts")),
                answer=(str, pydantic.Field(..., description="Consise final answer")),
            )
            cot = True
        response = await self.client.beta.chat.completions.parse(
            model=self.model, messages=messages, **kwargs
        )
        if cot:
            return response.choices[0].message.parsed.answer
        return response.choices[0].message.parsed


class Variable:
    """A `Variable` is to a string, like `Tensor` is to a numpy.array in Pytorch.
    It is a node in the a computation graph, which keeps track of feedback (gradients) and predecessors."""

    def __init__(self, value: str, preds=(), push_grad=None, requires_grad=True, name="anon"):
        self.value = value
        self.preds = preds
        self.push_grad = push_grad
        self.requires_grad = requires_grad
        self.name = name
        self.feedback = []

    async def backward(self, lm: LMConfig):
        # Good interview question: Change the topological sort to Kahn's algorithm
        # and use asyncio.wait(return_when=asyncio.FIRST_COMPLETED) to parallelize.
        visited, order = set(), []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node.preds:
                    dfs(child)
                order.append(node)

        dfs(self)
        self.feedback.append(self.value)
        for node in reversed(order):
            if node.push_grad is not None and node.requires_grad:
                await node.push_grad(lm, "\n".join(node.feedback))


class Optimizer:
    """The optimizer performs "textual gradient descent" on the parameters.
    It keeps track of the past values of the variables, which is similar to momentum in pytorch."""

    def __init__(self, vs: list[Variable], max_past_values=10):
        self.vs = vs
        self.max_past_values = max_past_values
        self.past_values = []  # past_values[t][i] is the t-th value of v_i
        self.past_feedbacks = []

    async def step(self, lm: LMConfig):
        self.past_values.append([])
        self.past_feedbacks.append([])
        if len(self.past_values) > self.max_past_values:
            self.past_values.pop(1)
            self.past_feedbacks.pop(1)
        for v in self.vs:
            self.past_values[-1].append(v.value)
            self.past_feedbacks[-1].append("\n".join(v.feedback))
            del v.feedback[:]  # optimizer.step automatically does model.zero_grad()

        tasks = []
        # All the optimization tasks can be done in parallel
        for i in range(len(self.vs)):
            messages = [{"role": "system", "content": "Past strings from assistant and feedback from user:"}]
            for vals, feedbacks in zip(self.past_values, self.past_feedbacks):
                messages += [
                    {"role": "assistant", "content": f"String: {vals[i]}"},
                    {"role": "user", "content": f"Feedback: {feedbacks[i]}"},
                ]
            tasks.append(
                lm.call(
                    messages
                    + [
                        {
                            "role": "system",
                            "content": (
                                "Suggest a new string, similar to the best past ones from the assistant, "
                                "but which incorporates a bit of the feedback from the user. "
                                "Don't overfit on a particular feedback, but capture the general pattern."
                                "Make sure it captures the intent and output format of the original string."
                            ),
                        },
                    ],
                    response_format=pydantic.create_model(
                        "Response",
                        thoughts=(
                            str,
                            pydantic.Field(
                                description="What's the general pattern in the feedback? "
                                "Which past strings received the most positive response?"
                            ),
                        ),
                        new_string=(str, pydantic.Field(description="New suggested string")),
                    ),
                )
            )
        # Update the values of all the variables
        for v, c in zip(self.vs, await asyncio.gather(*tasks)):
            v.value = c.new_string


################################################################################
# Basic functions
################################################################################


async def complete(lm: LMConfig, **kwargs: Variable) -> Variable:
    """This function calls the language model with a prompt and returns the response."""

    assert all(isinstance(v, Variable) for v in kwargs.values()), "All inputs should be Variables"
    user_content = str({k: v.value for k, v in kwargs.items()})  # Simple multi-input template here
    agent_response = await lm.call([{"role": "user", "content": user_content}])

    async def push_grad(lm: LMConfig, feedback: str):
        response = await lm.call(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": agent_response},
                {"role": "user", "content": f"Feedback: {feedback}"},
                {
                    "role": "system",
                    "content": (
                        "Analyze the user's initial inputs (provided as {key: value} pairs) and the overall feedback received. "
                        "Generate specific, targeted feedback for each input value. Return this as a JSON object with "
                        "matching keys. Focus on how adjusting each specific input could improve the assistant's "
                        "response. Ensure that feedback for each input is isolated and relevant only to that input, "
                        "avoiding 'contamination' from aspects controlled by other inputs. Consider the direct impact "
                        "and influence of each input on the final output."
                    ),
                },
            ],
            # Create a response format that matches the input kwargs
            response_format=pydantic.create_model(
                "Response",
                explanation=(
                    str,
                    pydantic.Field(
                        ..., description="Analysis of the feedback and its implications for each input"
                    ),
                ),
                individual_feedback=(
                    pydantic.create_model("Feedback", **{k: (str, ...) for k in kwargs.keys()}),
                    ...,
                ),
            ),
        )
        for k, v in kwargs.items():
            v.feedback.append(getattr(response.individual_feedback, k))

    return Variable(agent_response, list(kwargs.values()), push_grad, name="complete")


async def wikipedia_summary(query: Variable) -> str:
    """This function calls  Get a summary of a Wikipedia page."""
    import wikipedia

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            summary = await loop.run_in_executor(pool, wikipedia.summary, query.value)
        except Exception as e:
            summary = str(e)

    async def push_grad(lm: LMConfig, feedback: str):
        response = await lm.call(
            [
                {"role": "user", "content": "Selected Wikipedia page: " + query.value},
                {"role": "assistant", "content": "Retrieved summary: " + summary},
                {"role": "user", "content": f"Feedback:\n{feedback}"},
                {
                    "role": "system",
                    "content": (
                        "Based on the feedback received on the retrieved summary, evaluate the choice of Wikipedia page. "
                        "Was this the most appropriate page to look up for the given context? "
                        "Or would a different Wikipedia page have been more suitable? "
                        "Consider if the content meets the user's needs or if another page might have been more relevant or comprehensive."
                    ),
                },
            ],
            response_format=pydantic.create_model(
                "Response",
                thoughts=(
                    str,
                    pydantic.Field(description="How does the feedback affect the choice of Wikipedia page?"),
                ),
                feedback=(str, pydantic.Field(description="Feedback on the choice of Wikipedia page")),
            ),
        )
        query.feedback.append(response.feedback)

    return Variable(summary, [query], push_grad, name="wikipedia_summary")


def concat(vs: list[Variable]) -> Variable:
    """Concatenate a list of variables."""

    vs = vs[:]  # Make a copy, nasty bug if we don't
    value = "\n\n".join(v.value for v in vs)

    # We do the lazy thing here, and just give the same feedback to all the inputs.
    async def push_grad(lm: LMConfig, feedback: str):
        for v in vs:
            v.feedback.append(feedback)

    return Variable(value, vs, push_grad, name=f"concat({', '.join(v.name for v in vs)})")


async def equality_loss(lm, answer: Variable, expected: Variable) -> Variable:
    """Simple loss functions that checks for equality between the answer and the expected value."""
    prompt = Variable("Is `answer` equal to `expected`? What's wrong?", requires_grad=False)
    response = await complete(lm, prompt=prompt, answer=answer, expected=expected)

    # We only want to give feedback to the answer, not the expected value.
    async def push_grad(lm, feedback):
        answer.feedback.append(feedback)

    response.push_grad = push_grad
    response.name = "equality_loss"
    return response


################################################################################
# Visualization
################################################################################


def visualize_variable_graph(loss: Variable, filename: str = "variable_graph"):
    """Visualize the computation graph of a Variable using Graphviz."""
    import graphviz

    dot = graphviz.Digraph(comment="Variable Call Graph")
    dot.attr(rankdir="LR", size="12,8")
    dot.attr(dpi="300")

    nodes: dict[int, str] = {}
    edges: set[tuple] = set()

    def add_node(var: Variable):
        node_label = (
            f"{var.name}\\n" f"Value: {var.value[:30]}...\\n" f"Feedback: {', '.join(var.feedback)[:30]}..."
        )
        nodes[id(var)] = f"node_{id(var)}"
        dot.node(nodes[id(var)], node_label, shape="box", style="rounded")

    def add_edge(from_var: Variable, to_var: Variable):
        if (id(from_var), id(to_var)) not in edges:
            dot.edge(nodes[id(from_var)], nodes[id(to_var)])
            edges.add((id(from_var), id(to_var)))

    seen = set()

    def traverse(var: Variable):
        if id(var) in seen:
            return
        seen.add(id(var))
        add_node(var)
        for pred in var.preds:
            traverse(pred)
            add_edge(pred, var)

    traverse(loss)

    print("Saving graph to", filename)
    dot.render(filename, view=True, format="png")


################################################################################
# Multihop QA model
################################################################################


class MultihopModel:
    """A simple multi-hop QA model that uses Wikipedia to answer questions."""

    def __init__(self, hops=2):
        self.prompts = [
            Variable("Based on question and context, suggest one relevant Wikipedia page title")
            for _ in range(hops)
        ] + [Variable("Based on question and context, give a short factual answer")]
        self.hops = hops

    async def forward(self, lm: LMConfig, q: Variable) -> Variable:
        passages = []
        for i in range(self.hops):
            query = await complete(lm, question=q, context=concat(passages), prompt=self.prompts[i])
            passages.append(await wikipedia_summary(query))
        return await complete(lm, question=q, context=concat(passages), prompt=self.prompts[-1])


async def main():
    import datasets

    lm = LMConfig("gpt-4o-2024-08-06")
    model = MultihopModel(hops=2)
    optimizer = Optimizer(model.prompts)
    dataset = datasets.load_dataset("hotpot_qa", "fullwiki")

    accuracy = 0
    for i, x in enumerate(dataset["train"]):
        answer = await model.forward(lm, Variable(x["question"], requires_grad=False))
        loss = await equality_loss(lm, answer, Variable(x["answer"], requires_grad=False))
        await loss.backward(lm)
        visualize_variable_graph(loss, f"variable_graph_question_{i}")
        await optimizer.step(lm)

        print(f"\n\n---\nQuestion {i}: {x['question']}")
        print(f"Answer: {answer.value}")
        print(f"Expected: {x['answer']}")
        print("Prompts after optimizing:")
        for j, p in enumerate(model.prompts):
            print(f"{j}: {p.value}")
        accuracy += answer.value == x["answer"]
        print(f"Accuracy: {accuracy / (i + 1)}")


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(main())
