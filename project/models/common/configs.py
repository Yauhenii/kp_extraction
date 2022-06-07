# source https://github.com/PyTorchLightning/lightning-transformers/blob/0.1.0/lightning_transformers/core/config.py

from dataclasses import dataclass


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class OptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    num_training_steps: int = -1
    num_warmup_steps: float = 0.1


@dataclass
class TrainerConfig:
    ...


@dataclass
class TaskConfig:
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
