import torch

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SAVE_NAME = "checkpoint"
EPOCH = "epoch"
MODEL_KEY = "model_state_dict"
OPTIMIZER_KEY = "optimizer_state_dict"
SCHEDULER_KEY = "scheduler_state_dict"
LOSS_KEY = "loss"
TORCH_DEFAULT_DEVICE = torch.device("cpu")


@dataclass(slots=True)
class Checkpoint:
    path: Path
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epoch: int = 0
    best: float = float("inf")

    def step(self, loss: float, epoch: int):
        self.epoch = epoch
        if loss < self.best:
            self.best = loss
            self.save()

    def save(self, name: str = DEFAULT_SAVE_NAME):
        torch.save(
            {
                EPOCH: self.epoch,
                MODEL_KEY: self.model.state_dict(),
                OPTIMIZER_KEY: self.optimizer.state_dict(),
                SCHEDULER_KEY: self.scheduler.state_dict(),
                LOSS_KEY: self.best,
            },
            self.path / f"{name}.pt",
        )

    @staticmethod
    def __load(
        target: str,
        model: torch.nn.Module,
        path: Path = None,
        loaded_data: dict = None,
        device: torch.device = TORCH_DEFAULT_DEVICE,
    ):
        if loaded_data is None and path is not None:
            model.load_state_dict(torch.load(path, map_location=device)[target])
            return model
        elif loaded_data is not None:
            model.load_state_dict(loaded_data[target])
            return model
        else:
            raise ValueError("Either path or loaded_data must be provided.")

    @staticmethod
    def load_model(
        model: torch.nn.Module,
        path: Path = None,
        loaded_data: dict = None,
        device: torch.device = TORCH_DEFAULT_DEVICE,
    ):
        return Checkpoint.__load(MODEL_KEY, model, path, loaded_data, device)

    @staticmethod
    def load_optimizer(
        model: torch.nn.Module,
        path: Path = None,
        loaded_data: dict = None,
        device: torch.device = TORCH_DEFAULT_DEVICE,
    ):
        return Checkpoint.__load(OPTIMIZER_KEY, model, path, loaded_data, device)

    @staticmethod
    def load_scheduler(
        model: torch.nn.Module,
        path: Path = None,
        loaded_data: dict = None,
        device: torch.device = TORCH_DEFAULT_DEVICE,
    ):
        return Checkpoint.__load(SCHEDULER_KEY, model, path, loaded_data, device)

    @staticmethod
    def load(
        path: Path,
        pt_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device = torch.device("cpu"),
    ):
        ckpt = torch.load(pt_path, map_location=device)
        model = Checkpoint.load_model(model, path, ckpt, device)
        optimizer = Checkpoint.load_optimizer(optimizer, path, ckpt, device)
        scheduler = Checkpoint.load_scheduler(scheduler, path, ckpt, device)
        return Checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=ckpt["epoch"],
            best=ckpt["loss"],
        )
