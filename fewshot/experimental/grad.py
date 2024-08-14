
class Function:
    pass


class Variable:
    def __init__(self, requires_grad: bool):
        self.requires_grad = requires_grad
        self.feedback = defaultdict(list)

    # In pytorch the Variable/Tensor class would hold both the current value and the gradient.
    # Here it's not clear how I want to handle concurrency,
    def __getattr__(self, name) -> "View":
        return View(self, name, self.requires_grad)

    async def unwrap(self):
        pass

    def add_feedback(self, run_id: str, feedback: "Variable"):
        self.feedback[run_id].append(feedback)
        # In pytorch backprop just adds the gradient to the existing gradient.
        # This allows the "dynamic programming" implementation, where we process the
        # computation graph in topological order.
        # Here we instead accumulate it as a list, which allows us to use string feedback,
        # while still being able to process the graph in topological order.

        # The hard thing to figure out is how to handle variable fields, which themselves
        # become variables. Each field receive feedback from where it is used, but these
        # all have to come together in the end to form the combined feedback for the variable.
        pass


class View:
    def __init__(self, parent: Variable):
        self.data = data


class LMFunction[T: BaseModel, U: BaseModel](Function):
    def __init__(
        self,
        client: any,
        model: str,
        output_type: Type[U],
    ):
        pass

    def __call__(self, T) -> U:
        # Stitch together:
        # - System prompt
        # - Input description / schema
        # - Example pairs (user input, assistant output)
        # - Input
        pass

