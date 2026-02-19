"""Training utilities for BoltzKinema."""

from boltzkinema.training.losses import BoltzKinemaLoss
from boltzkinema.training.scheduler import get_warmup_constant_scheduler

__all__ = [
    "BoltzKinemaLoss",
    "get_warmup_constant_scheduler",
    "TrainConfig",
    "train",
]


def __getattr__(name: str):
    if name in ("TrainConfig", "train"):
        from boltzkinema.training.trainer import TrainConfig, train

        globals()["TrainConfig"] = TrainConfig
        globals()["train"] = train
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
