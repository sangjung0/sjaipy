import torch
import time

from dataclasses import dataclass
from pathlib import Path


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

    def save(self):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": self.best,
            },
            self.path / f"checkpoint_epoch_{self.epoch}_{int(time.time())}.pt",
        )

    @staticmethod
    def load(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device = torch.device("cpu"),
    ):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return Checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=ckpt["epoch"],
            best=ckpt["loss"],
        )
