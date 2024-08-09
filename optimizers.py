import optuna
import random

from fewshot import Example
from experimental.opt import GPC


class Optimizer:
    def __repr__(self):
        return self.__class__.__name__

    async def step(self):
        # For doing more complex things that require async.
        # For example, asking an LLM for a new prompt.
        # TODO: It would be nice if that could happen in the background, while we keep processing more examples?
        pass

    def tell(self, token, loss):
        raise NotImplementedError

    def suggest(self):
        raise NotImplementedError

    def best(self):
        return self.suggest()


class GreedyFewShot(Optimizer):
    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        self.losses = []

    def tell(self, token, loss):
        self.losses.append(loss)

    def suggest(self):
        # Include the input in the key for reproducibility
        self.losses.sort(
            key=lambda loss: (loss.score, loss.input.model_dump_json()), reverse=True
        )
        exs = [
            Example(
                loss.input, loss.expected if loss.expected is not None else loss.output
            )
            for loss in self.losses[: self.max_examples]
        ]
        return exs, None

    def best(self):
        return self.suggest()


class OptunaFewShot(Optimizer):
    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        sampler = optuna.samplers.TPESampler(seed=10)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.completed_trials = 0
        self.losses = []

    def _subset_from_trial(self, trial):
        subset_size = self.max_examples
        subset = []
        for i in range(subset_size):
            index = trial.suggest_int(f"index_{i}", 0, len(self.losses) - 1)
            loss = self.losses[index]
            answer = loss.expected if loss.expected is not None else loss.output
            subset.append(Example(loss.input, answer))
        return subset

    def suggest(self):
        if not self.losses or self.max_examples == 0:
            return [], None
        trial = self.study.ask()
        subset = self._subset_from_trial(trial)
        return subset, trial

    def best(self):
        if not self.losses or self.max_examples == 0:
            return []
        if self.completed_trials == 0:
            return self.suggest()[0]
        return self._subset_from_trial(self.study.best_trial)

    def tell(self, token, loss):
        self.losses.append(loss)
        if token is not None:
            if loss.score is None:
                raise ValueError("OptunaFewShot requires a score to be provided.")
            self.study.tell(token, loss.score)
            self.completed_trials += 1


class GPCFewShot(Optimizer):
    def __init__(self, n_examples: int, strategy="random"):
        self.n_examples = n_examples
        self.model = GPC(strategy=strategy)
        self.losses = []

    def __repr__(self):
        return f"GPCFewShot(strategy={self.model.strategy})"

    def suggest(self):
        if not self.losses or self.n_examples == 0:
            return [], None
        indices = self.model.ask(self.n_examples)
        assert len(indices) == self.n_examples
        subset = [Example(self.losses[i].input, self.losses[i].output) for i in indices]
        return subset, indices

    def best(self):
        if not self.losses or self.n_examples == 0:
            return []
        indices = self.model.best(self.n_examples)
        subset = [Example(self.losses[i].input, self.losses[i].output) for i in indices]
        return subset

    def tell(self, indices, loss):
        self.losses.append(loss)
        # The GPC model needs to know about the choices for indices, even if we don't
        # also have a value for a new example subset.
        self.model.add_index_value(loss.score)
        if indices is not None:
            self.model.tell(indices, loss.score)


class OptimizedRandomSubsets(Optimizer):
    def __init__(self, n_examples: int):
        self.n_examples = n_examples
        self.n_subsets = 10
        sampler = optuna.samplers.TPESampler(seed=10)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.subsets = []
        self.losses = []

    def _examples_from_losses(self):
        return [
            Example(
                loss.input,
                loss.expected if loss.expected is not None else loss.output,
            )
            for loss in self.losses
            if loss.input is not None
            and (loss.expected is not None or loss.output is not None)
        ]

    def suggest(self):
        if not self.losses or self.n_examples == 0:
            return [], None
        exs = self._examples_from_losses()
        # In the beginning we pick random subsets
        if len(exs) < self.n_examples + self.n_subsets:
            return exs[: self.n_examples], None
        # Once we have enough data, we create the actual subsets to optimize over
        if not self.subsets:
            self.subsets = [
                random.sample(exs, self.n_examples) for _ in range(self.n_subsets)
            ]
        trial = self.study.ask()
        index = trial.suggest_categorical("subset", range(self.n_subsets))
        return self.subsets[index], trial

    def best(self):
        if not self.losses or self.n_examples == 0:
            return []
        exs = self._examples_from_losses()
        if not self.subsets:
            return exs[: self.n_examples]
        trial = self.study.best_trial
        index = trial.suggest_categorical("subset", range(self.n_subsets))
        return self.subsets[index]

    def tell(self, trial, loss):
        self.losses.append(loss)
        if trial is not None:
            self.study.tell(trial, loss.score)


class HardCaseFewShot(Optimizer):
    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        self.losses = []

    def tell(self, token, loss):
        self.losses.append(loss)

    def suggest(self):
        subset = (
            [
                Example(loss.input, loss.expected)
                for loss in self.losses
                if loss.score == 0
            ]
            + [
                Example(loss.input, loss.output)
                for loss in self.losses
                if loss.score == 1
            ]
        )[: self.max_examples]
        return subset, None

    def best(self):
        subset, _ = self.suggest()
        return subset
