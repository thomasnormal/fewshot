################################################################################
# Experimental optimizers that didn't really work
################################################################################


class OptunaFewShotV2(Optimizer):
    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        self.ongoing_trials = defaultdict(list)
        sampler = optuna.samplers.TPESampler(seed=10)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

    def _subset_from_trial(self, trial, exs):
        exs = exs[:]
        subset = []
        for i in range(len(exs)):
            if trial.suggest_int(f"index_{i}", 0, 1):
                subset.append(i)
        # Ensure we have enough examples
        while len(subset) < self.max_examples:
            i = random.randrange(len(exs))
            if i not in subset:
                subset.append(i)
                # trial.set_system_attr(f"index_{i}", 1)
                trial._cached_frozen_trial.params[f"index_{i}"] = 1
        # Ensure we don't have too many examples
        if len(subset) > self.max_examples:
            unset = random.sample(subset, len(subset) - self.max_examples)
            for i in unset:
                # trial.set_system_attr(f"index_{i}", 0)
                trial._cached_frozen_trial.params[f"index_{i}"] = 0
                subset.remove(i)
        subset = [exs[i] for i in subset]
        return trial, subset

    def suggest(self, input, log):
        if not log or self.max_examples == 0:
            return []
        trial = self.study.ask()
        trial, subset = self._subset_from_trial(trial, log)
        self.ongoing_trials[str(input)].append(trial)
        return subset

    def get_best(self, log):
        if not log or self.max_examples == 0:
            return []
        trial = self.study.best_trial
        trial, subset = self._subset_from_trial(trial, log)
        return subset

    def step(self, input, output, score):
        try:
            trial = self.ongoing_trials[str(input)].pop(0)
        except IndexError:
            return
        self.study.tell(trial, score)


class OptunaFewShotV3(Optimizer):
    def __init__(self, max_examples: int):
        self.max_examples = max_examples
        self.ongoing_trials = defaultdict(list)
        sampler = optuna.samplers.TPESampler(seed=10, multivariate=True)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

    def _subset_from_trial(self, trial, exs):
        exs = exs[:]
        subset = []
        for i in range(len(exs)):
            try:
                if trial.suggest_int(f"index_{i}", 0, 1):
                    subset.append(i)
            except ValueError as e:
                print("Warning:", e)
                pass
        # Ensure we have enough examples
        while len(subset) < self.max_examples:
            i = random.randrange(len(exs))
            if i not in subset:
                subset.append(i)
        # Ensure we don't have too many examples
        if len(subset) > self.max_examples:
            unset = random.sample(subset, len(subset) - self.max_examples)
            for i in unset:
                subset.remove(i)
        params = {f"index_{i}": int(i in subset) for i in range(len(exs))}
        subset = [exs[i] for i in subset]
        return params, subset

    def suggest(self, input, log):
        if not log or self.max_examples == 0:
            return []
        trial = self.study.ask()
        params, subset = self._subset_from_trial(trial, log)
        self.ongoing_trials[str(input)].append(params)
        return subset

    def get_best(self, log):
        if not log or self.max_examples == 0:
            return []
        trial = self.study.best_trial
        _, subset = self._subset_from_trial(trial, log)
        return subset

    def step(self, input, output, score):
        try:
            params = self.ongoing_trials[str(input)].pop(0)
        except IndexError:
            return
        self.study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions={
                    key: optuna.distributions.IntDistribution(0, 1) for key in params
                },
                value=score,
            )
        )
