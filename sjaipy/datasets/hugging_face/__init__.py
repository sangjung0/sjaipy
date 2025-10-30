# sjaipy/datasets/hugging_face/__init__.py

from sjaipy.datasets.hugging_face.ami import AMI
from sjaipy.datasets.hugging_face.tedlium import Tedlium
from sjaipy.datasets.hugging_face.vox_populi import VoxPopuli
from sjaipy.datasets.hugging_face.dataset_loader import DatasetLoader
from sjaipy.datasets.hugging_face.zeroth_korean import ZerothKorean
from sjaipy.datasets.hugging_face.ksponspeech import KSPonSpeech

__all__ = [
    "AMI",
    "Tedlium",
    "VoxPopuli",
    "DatasetLoader",
    "ZerothKorean",
    "KSPonSpeech",
]
