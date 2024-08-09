import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

################################################################################
# Code for Gaussian Process Classification
################################################################################


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, mu_dim):
        super(LogisticRegressionModel, self).__init__()
        self.mu = nn.Parameter(torch.randn(mu_dim))
        self.A = nn.Parameter(torch.randn(input_dim, input_dim))
        with torch.no_grad():
            self.A[:] = torch.eye(input_dim)

    def forward(self, S):
        # S is expected to be of shape [batch_size, k]
        M = self.mu[S].sum(dim=1)  # Summing over the second dimension
        V = (
            self.A[S].sum(dim=1).pow(2).sum(dim=1)
        )  # Summing over the second dimension then squaring and summing
        return torch.sigmoid(M / torch.sqrt(S.size(1) + V))


class GPC:
    def __init__(self):
        self.observations = []
        self.values = []
        self.model = None
        self.dirty = True
        self.index_values = []

    def _retrain(self, steps=10):
        # N = max(max(obs) for obs in self.observations) + 1
        N = len(self.index_values)
        model = LogisticRegressionModel(N, N)

        # Copy the old model parameters for a warm start
        with torch.no_grad():
            if self.model is not None:
                n = self.model.A.size(0)
                model.A[:n, :n] = self.model.A
                model.mu[:n] = self.model.mu
            else:
                n = 0
            # We might want to scale the mu value here in some way
            for i in range(n, min(N, len(self.index_values))):
                model.mu[i] = self.index_values[i]

        if self.observations:
            S = torch.tensor(self.observations)
            v = torch.tensor(self.values)
            optimizer = optim.Adam(model.parameters(), lr=1e-2)

            for _ in range(steps):
                Ps = model(S)
                loss = F.binary_cross_entropy(Ps.view(-1, 1), v.view(-1, 1))
                loss.backward()
                optimizer.step()

        self.model = model

    def tell(self, indices, value):
        self.observations.append(indices)
        self.values.append(value)
        self.dirty = True

    def add_index_value(self, value):
        self.index_values.append(value)

    def ask(self, N: int):
        if self.model is None or self.dirty:
            self._retrain()
        # Sample from the normal distribution (model.mu and model.A)
        with torch.no_grad():
            xs = self.model.A @ torch.randn_like(self.model.mu) + self.model.mu
            # xs *= torch.tensor(self.index_values[: len(xs)])
            S = torch.argsort(xs, descending=True)[:N]

            # xs *= 1 / (1 + torch.arange(len(xs)).float()) ** 0.5
            # S = torch.argsort(self.model.mu, descending=True)[:N]
            # ws = torch.sigmoid(xs)
            # S = torch.multinomial(ws, N, replacement=False)
        return S.tolist()

    def best_(self, N: int):
        if self.alpha is None or self.dirty:
            self._evaluate_alpha()
        # TODO: This should actually return the subset S, that maximizes
        # torch.sigmoid(M / torch.sqrt(S.size(1) + V))
        # So we can take covariance into account
        with torch.no_grad():
            S = torch.argsort(self.model.mu, descending=True)[:N]
        return S.tolist()

    def best(self, N: int):
        if not self.observations:
            if len(self.index_values) >= N:
                xs = torch.tensor(self.index_values)
                return torch.argsort(xs, descending=True)[:N]
            raise ValueError("No observations have been made yet.")
        if self.model is None or self.dirty:
            self._retrain()

        mu = self.model.mu
        A = self.model.A

        selected = set()
        for _ in range(N):
            best_value = -float("inf")
            best_index = -1
            for i in range(len(mu)):
                if i in selected:
                    continue
                temp_S = list(selected) + [i]
                M = mu[temp_S].sum()
                V = A[temp_S].sum().pow(2).sum()
                value = torch.sigmoid(M / torch.sqrt(len(temp_S) + V))
                if value > best_value:
                    best_value = value
                    best_index = i
            selected.add(best_index)

        return list(selected)
