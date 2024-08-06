import optuna
import random
import math

def create_subset(trial):
    subset_size = trial.suggest_int("subset_size", 1, max_subset_size)
    subset_indices = []
    for i in range(subset_size):
        index = trial.suggest_int(f"index_{i}", 0, len(exs) - 1)
        if index not in subset_indices:
            subset_indices.append(index)
    return [exs[i] for i in subset_indices]

# Define your examples and function f here
exs = ["example1", "example2", "example3", "example4", "example5"]
max_subset_size = 3

vs = {key: random.normalvariate() for key in exs}
denum = sum(map(math.exp, vs.values()))
def p(subset):
    return sum(math.exp(vs[key]) for key in subset) / denum
def f(subset):
    return float(random.random() < p(subset))

# Create a study object
study = optuna.create_study(direction="maximize")

n_trials = 100
for _ in range(n_trials):
    # Create a new trial
    trial = study.ask()
    
    # Create a subset based on the trial's suggestions
    subset = create_subset(trial)
    
    # Evaluate the subset
    num_evaluations = 1
    successes = sum(f(subset) for _ in range(num_evaluations))
    objective_value = successes / num_evaluations

    print("Subset:", subset, "Value:", objective_value)
    
    # Report the result back to Optuna
    study.tell(trial, objective_value)

print(vs)

# Print the best parameters and value
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Get the best subset
best_subset_indices = [
    value for key, value in trial.params.items() if key.startswith("index_")
]
best_subset = [exs[i] for i in best_subset_indices]
print("Best subset:", best_subset)
print("Best value:", trial.value)
print(p(best_subset))
sorted_values = sorted(vs.keys(), key=lambda x: vs[x], reverse=True)
print(sorted_values)
print([p(sorted_values[:i]) for i in range(1, len(sorted_values) + 1)])
