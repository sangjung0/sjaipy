from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from typing import Callable

from sj_ai_utils.evaluator.sclite_utils import TRNFormat
from sj_ai_utils.datasets.libri_speech_asr_corpus.file_type import X, Y
from sj_ai_utils.datasets.libri_speech_asr_corpus.service import select_file_from_dir

if TYPE_CHECKING:
    pass


def trans_txt_to_sclite_trn(
    src: Path, normalizer: Callable[[str], str] = lambda x: x
) -> list[TRNFormat]:
    result = []
    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            uid, *words = line.rstrip().split()
            sent = normalizer(" ".join(words))
            result.append(TRNFormat(id=uid, text=sent))
    return result


def generate_trn(
    data_paths: list[Path],
    transcribe: Callable[[Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    max_count: int = -1,
) -> dict[str, dict[str, list[TRNFormat]]]:
    if not data_paths:
        raise ValueError("No data paths provided.")
    if not all(p.exists() for p in data_paths):
        raise ValueError("One or more data paths do not exist.")

    result = {}
    for i, dirpath in enumerate(data_paths):
        if max_count != -1 and i >= max_count:
            break

        trans_txt = select_file_from_dir(dirpath, Y)
        ref_items = trans_txt_to_sclite_trn(trans_txt, normalizer=normalizer)
        hyp_items = [
            TRNFormat(id=flac.stem, text=normalizer(transcribe(flac)))
            for flac in sorted(select_file_from_dir(dirpath, X))
        ]

        result[trans_txt.stem] = {"ref": ref_items, "hyp": hyp_items}

    return result


__all__ = ["trans_txt_to_sclite_trn", "generate_trn"]
