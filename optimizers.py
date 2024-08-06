from collections import defaultdict
import optuna


class Optimizer:
    def step(self):
        raise NotImplementedError

    def suggest(self):
        raise NotImplementedError


class OptunaFewShot(Optimizer):
    def __init__(self, max_examples:int):
        self.max_examples = max_examples
        self.ongoing_trials = defaultdict(list)
        sampler = optuna.samplers.TPESampler(seed=10)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

    def _subset_from_trial(self, trial, exs):
        #subset_size = trial.suggest_int("subset_size", 1, self.max_examples)
        subset_size = self.max_examples
        subset = []
        for i in range(subset_size):
            index = trial.suggest_int(f"index_{i}", 0, len(exs) - 1)
            subset.append(exs[index])
        return subset

    def suggest(self, input, log):
        if not log or self.max_examples == 0:
            return []
        trial = self.study.ask()
        self.ongoing_trials[str(input)].append(trial)
        return self._subset_from_trial(trial, log)

    def get_best(self, log):
        if not log or self.max_examples == 0:
            return []
        trial = self.study.best_trial
        return self._subset_from_trial(trial, log)

    def step(self, input, output, score):
        try:
            trial = self.ongoing_trials[str(input)].pop()
        except IndexError:
            return
        self.study.tell(trial, score)


class GreedyFewShot(Optimizer):
    def __init__(self):
        pass

    def step(self, input, output, score):
        pass

    def suggest(self, input, log, max_examples:int):
        def key(ex):
            # Include the input json in the key for reproducibility
            return ex.score, ex.input.model_dump_json()

        # Add the best examples from the log
        log = [ex for ex in log if ex.score == 1]

        return sorted(log, key=key, reverse=True)[: max_examples]

class DummyOptimizer:
    def suggest(self, input, log, max_examples:int):
        return []

