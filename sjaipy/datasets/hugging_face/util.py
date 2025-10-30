from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    pass

DEFAULT_DOWNLOAD_PATH = Path("/workspaces/dev/datasets")


def download(
    repo_id: str,
    repo_type: str,
    *args,
    local_dir: Path = DEFAULT_DOWNLOAD_PATH,
    **kwargs,
) -> None:
    data_dir = local_dir / f"{repo_type}--{repo_id.replace('/', '--')}"
    snapshot_download(
        repo_id, *args, repo_type=repo_type, local_dir=local_dir, **kwargs
    )
    return data_dir


__all__ = ["download"]
