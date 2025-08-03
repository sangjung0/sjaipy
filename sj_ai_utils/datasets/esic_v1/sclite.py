from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

from sj_ai_utils.evaluator.sclite_utils import TRNFormat
from sj_ai_utils.datasets.esic_v1.service import select_file_from_dir
from sj_ai_utils.datasets.esic_v1.file_type import MP4, VERBATIM, ORTO

if TYPE_CHECKING:
    from typing import Callable


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
    for i, data_path in enumerate(data_paths):
        if max_count != -1 and i >= max_count:
            break

        key = data_path.absolute()

        src = select_file_from_dir(data_path, ORTO)
        txt = src.read_text(encoding="utf-8")
        txt = normalizer(txt)
        ref = TRNFormat(id=key, text=txt)

        mp4 = select_file_from_dir(data_path, MP4)
        pred_txt = transcribe(mp4)
        pred_txt = normalizer(pred_txt)
        hyp = TRNFormat(id=key, text=pred_txt)

        result[key] = {
            "ref": [ref],
            "hyp": [hyp],
        }

    return result


__all__ = [
    "generate_trn",
]
