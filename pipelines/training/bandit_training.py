"""
Multi-Armed Bandit Training
Train contextual bandit for challenge selection
"""

import numpy as np
from typing import List, Dict
import json
from pathlib import Path
import pickle
from deepiri_modelkit.logging import get_logger

logger = get_logger("helox.bandit")


class ContextualBandit:
    """
    Contextual multi-armed bandit with Thompson sampling.

    Uses Beta-Bernoulli Thompson sampling per arm:
      - α (successes) and β (failures) are scalar counts initialised to 1
        giving a uniform Beta(1,1) prior.
      - At selection time, θ ~ Beta(α, β) is drawn per arm and dotted with
        the normalised context vector to rank arms contextually.
      - At update time, α or β is incremented by 1 (not by the context
        vector) to maintain valid Beta posterior parameters.  Adding the
        raw context vector would corrupt the posterior because Beta(α, β)
        requires α > 0 and β > 0 with magnitude independent of context scale.

    For reward threshold r > 0.5 → success (α += 1); otherwise failure (β += 1).
    """

    def __init__(self, challenge_types: List[str], context_dim: int = 10):
        self.challenge_types = challenge_types
        self.context_dim = context_dim
        # Scalar Beta parameters per arm — Beta(1,1) uniform prior
        self.alpha = {ct: 1.0 for ct in challenge_types}
        self.beta = {ct: 1.0 for ct in challenge_types}
        self.counts = {ct: 0 for ct in challenge_types}

    def select_challenge(self, context: np.ndarray) -> str:
        """
        Select challenge using Thompson sampling.

        Draws θ ~ Beta(α, β) per arm (scalar posterior sample representing
        the arm's estimated success probability) and returns the arm with
        the highest sample.  Context is not used at selection time in this
        implementation; the Beta posterior alone encodes past success/failure.
        """
        scores = {
            ct: float(np.random.beta(self.alpha[ct], self.beta[ct])) for ct in self.challenge_types
        }
        return max(scores, key=scores.get)

    def update(self, challenge_type: str, reward: float, context: np.ndarray):
        """
        Update Beta posterior with scalar ±1 increments.

        Reward > 0.5 → success: α += 1 (arm gets more likely to be sampled)
        Reward ≤ 0.5 → failure: β += 1 (arm gets less likely)

        The context is NOT used in the update to keep Beta parameters valid
        (must remain > 0 and independent of context magnitude).
        """
        self.counts[challenge_type] += 1

        if reward > 0.5:
            self.alpha[challenge_type] += 1.0
        else:
            self.beta[challenge_type] += 1.0

    def train(self, training_data: List[Dict]):
        """Train bandit on historical data."""
        logger.info("Training contextual bandit", samples=len(training_data))

        for sample in training_data:
            challenge_type = sample["challenge_type"]
            reward = sample["reward"]
            context = np.array(sample.get("context", []))

            self.update(challenge_type, reward, context)

        logger.info("Bandit training complete")

    def save(self, path: str):
        """Save trained bandit."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "challenge_types": self.challenge_types,
                    "context_dim": self.context_dim,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "counts": self.counts,
                },
                f,
            )
        logger.info("Bandit saved", path=path)

    def load(self, path: str):
        """Load trained bandit."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.challenge_types = data["challenge_types"]
            self.context_dim = data["context_dim"]
            self.alpha = data["alpha"]
            self.beta = data["beta"]
            self.counts = data["counts"]
        logger.info("Bandit loaded", path=path)


def train_bandit_from_data(dataset_path: str, output_path: str):
    """Train bandit from dataset."""
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    challenge_types = list(set([d["challenge_type"] for d in data]))
    bandit = ContextualBandit(challenge_types)

    bandit.train(data)
    bandit.save(output_path)

    return bandit
