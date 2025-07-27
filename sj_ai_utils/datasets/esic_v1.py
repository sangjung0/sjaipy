from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

from sj_ai_utils.evaluator.sclite_utils import TRNFormat

if TYPE_CHECKING:
    from typing import Callable


def search_file_from_dir(dir: Path, file_type: str) -> Path:
    """Search for a specific file type in the given directory.

    Args:
        dir (Path): The directory to search in.
        file_type (str): The type of file to search for. Valid types are:
            - "txt": Manual translation by the interpreter.
            - "vert+ts": Contains start time, end time, original word, symbol-included word,
              sentence segment number, and other tags.
            - "o": Refined sentence with special tags removed.
            - "o+ts": Refined sentence with special tags removed and timestamps.
            - "v": Lowercase, punctuation removed, numbers as words, tags removed,
              incomplete utterances included.
            - "pv": Includes uppercase and punctuation, numbers as numbers, tags removed,
              incomplete utterances included.
            - "mp4": Original video.

    Raises:
        ValueError: If the file type is invalid.
        FileNotFoundError: If the file is not found.

    Returns:
        Path: The path to the found file.
    """
    FILE_TYPE = {
        "txt": {"file": "en.OSt.man.txt", "explain": "통역사의 수동 번역"},
        "vert+ts": {
            "file": "en.OSt.man.vert+ts",
            "explain": "시작시간, 종료시간, 원형단어, 기호 포함 단어, 문장 세그먼트 번호, 기타 태그",
        },
        "o": {"file": "en.OSt.man.orto.txt", "explain": "특수 태그 제거된 정제 문장"},
        "o+ts": {
            "file": "en.OSt.man.orto+ts.txt",
            "explain": "특수 태그 제거 + 타임스탬프",
        },
        "v": {
            "file": "en.OSt.man.verbatim.txt",
            "explain": "소문자, 구두점 제거, 숫자는 문자로, 태그 제거, 불완전 발화 포함",
        },
        "pv": {
            "file": "en.OSt.man.punct-verbatim.txt",
            "explain": "대소문자 및 구두점 포함, 숫자는 숫자로, 태그 제거, 불완전 발화 포함",
        },
        "mp4": {"file": "en.OS.man-diar.mp4", "explain": "원본영상"},
    }

    if file_type not in FILE_TYPE:
        raise ValueError(
            f"Invalid file type: {file_type}. Valid types are: {', '.join(FILE_TYPE.keys())}"
        )

    file_path = dir / FILE_TYPE[file_type]["file"]
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}. Expected file for type '{file_type}'."
        )

    return file_path


def search_all_data(source: Path, verbose=True) -> list[Path]:
    data_dirs = []
    for dirpath in (p for p in source.rglob("*") if p.is_dir()):
        video = dirpath / "en.OS.man-diar.mp4"
        if not video.exists():
            continue
        data_dirs.append(dirpath)
    return data_dirs


def search_all_ref_and_hyp(
    source: Path,
    transcribe: Callable[[Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    max_count: int = -1,
    verbose: bool = True,
) -> dict[str, dict[str, list[TRNFormat]]]:

    if not source.exists() or not source.is_dir():
        print(f"Source path {source} does not exist or is not a directory.")
        return {}

    result = {}
    count = 0

    dirs = search_all_data(source, verbose=verbose)
    for data_path in dirs:
        if max_count != -1 and count >= max_count:
            break
        count += 1

        key = data_path.parent.stem + "_" + data_path.stem
        src = search_file_from_dir(data_path, "v")
        txt = src.read_text(encoding="utf-8")
        txt = normalizer(txt)
        ref = TRNFormat(id=key, text=txt)

        mp4 = search_file_from_dir(data_path, "mp4")
        pred_txt = transcribe(mp4)
        pred_txt = normalizer(pred_txt)
        hyp = TRNFormat(id=key, text=pred_txt)

        result[key] = {
            "ref": [ref],
            "hyp": [hyp],
        }

    return result


__all__ = [
    "txt_to_sclite_trn",
    "search_file_from_dir",
    "search_all_data",
    "search_all_ref_and_hyp",
]
