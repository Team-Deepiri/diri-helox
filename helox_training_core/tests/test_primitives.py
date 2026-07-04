"""Tests for the generic training primitives (framework-agnostic)."""

from typing import Any, Dict, List

import pytest

from deepiri_helox_training import (
    BaseTrainer,
    BaseTrainingConfig,
    CheckpointStrategy,
    InMemoryCheckpointStrategy,
    MetricAggregator,
    StepResult,
    TrainerCallback,
)
from deepiri_helox_training.data import StaticDataPipeline


class _ToyTrainer(BaseTrainer):
    """Minimal numeric trainer: 'loss' = abs(weight - target), step nudges weight."""

    def __init__(self, target: float, **kwargs: Any) -> None:
        super().__init__(BaseTrainingConfig(epochs=10, eval_interval=1, **kwargs))
        self.weight = 0.0
        self.target = target
        self._mode = "eval"

    def set_train_mode(self) -> None:
        self._mode = "train"

    def set_eval_mode(self) -> None:
        self._mode = "eval"

    def train_step(self, batch: float) -> StepResult:
        # Nudge toward target, snapping once close so the loss plateaus (lets
        # best-tracking / early-stopping reach a steady state in tests).
        if abs(self.target - self.weight) < 0.1:
            self.weight = self.target
        else:
            self.weight += 0.5 * (self.target - self.weight)
        return StepResult(loss=abs(self.target - self.weight), batch_size=1)

    def eval_step(self, batch: float) -> StepResult:
        return StepResult(loss=abs(self.target - self.weight), batch_size=1)

    def collect_state(self) -> Dict[str, Any]:
        return {"weight": self.weight, "epoch": self.state.epoch}

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.weight = state["weight"]
        self.state.epoch = state["epoch"]


def _pipeline() -> StaticDataPipeline:
    return StaticDataPipeline(train=[0.0, 0.0, 0.0], val=[0.0])


def test_metric_aggregator_weighted_mean():
    agg = MetricAggregator()
    agg.update(StepResult(loss=2.0, batch_size=1, metrics={"acc": 0.0}))
    agg.update(StepResult(loss=4.0, batch_size=3, metrics={"acc": 1.0}))
    out = agg.compute()
    assert out["loss"] == pytest.approx((2.0 * 1 + 4.0 * 3) / 4)
    assert out["acc"] == pytest.approx((0.0 * 1 + 1.0 * 3) / 4)


def test_fit_runs_and_converges():
    trainer = _ToyTrainer(target=10.0)
    history = trainer.fit(_pipeline())
    assert "train_loss" in history and "val_loss" in history
    assert len(history["train_loss"]) == 10
    # loss should decrease as weight approaches target
    assert history["val_loss"][-1] < history["val_loss"][0]


def test_best_tracking_and_early_stopping():
    trainer = _ToyTrainer(target=1.0, early_stopping_patience=2)
    trainer.fit(_pipeline())
    # converges fast then stops improving -> early stop before all 10 epochs
    assert trainer.state.epoch < 9
    assert trainer.state.best_metric is not None


def test_checkpoint_roundtrip():
    strat = InMemoryCheckpointStrategy()
    trainer = _ToyTrainer(target=5.0)
    trainer.checkpoint_strategy = strat
    trainer.fit(_pipeline())
    assert "best" in strat.list_tags()

    restored = _ToyTrainer(target=5.0)
    restored.checkpoint_strategy = strat
    restored.load_checkpoint("best")
    assert restored.weight == pytest.approx(trainer.weight, abs=1.0)


def test_checkpoint_cleanup_keeps_best_and_recent():
    strat = InMemoryCheckpointStrategy()
    for i in range(5):
        strat.save({"i": i}, f"epoch_{i}")
    strat.save({"i": "b"}, "best")
    strat.cleanup(keep_last=2)
    tags = strat.list_tags()
    assert "best" in tags
    assert "epoch_4" in tags and "epoch_3" in tags
    assert "epoch_0" not in tags


def test_callbacks_receive_events():
    events: List[str] = []

    class Recorder(TrainerCallback):
        def on_train_begin(self, state):
            events.append("begin")

        def on_epoch_end(self, state, metrics):
            events.append("epoch")

        def on_train_end(self, state):
            events.append("end")

    trainer = _ToyTrainer(target=2.0)
    trainer.callbacks.add(Recorder())
    trainer.fit(_pipeline())
    assert events[0] == "begin"
    assert events[-1] == "end"
    assert events.count("epoch") == 10


def test_checkpoint_strategy_is_abstract():
    with pytest.raises(TypeError):
        CheckpointStrategy()  # type: ignore[abstract]
