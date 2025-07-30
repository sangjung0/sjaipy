from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

from sj_ai_utils.evaluator.sclite_utils import TRNFormat, make_trn_file

if TYPE_CHECKING:
    from typing import Callable


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


def generate_all_ref_and_hyp_file(
    source: Path,
    destination: Path,
    transcribe: Callable[[Path], TRNFormat],
    normalizer: Callable[[str], str] = lambda x: x,
    max_count: int = -1,
    verbose: bool = True,
) -> None:
    """source 폴더에 있는 모든 *.train.txt 파일을 *.ref.trn 파일로 destination위치에 같은 상대경로로 저장하고, *.flac 파일을 찾아, transcribe 함수를 통해 음성 데이터를 list[TRN] 형태로 변환하여 동일한 상대경로로 *.hyp.trn 파일로 저장하는 함수

    Args:
        source (Path): *.train.txt와 *.flac 파일이 있는 폴더 경로
        destination (Path): 결과가 저장될 폴더 경로
        transcribe (Callable[[np.ndarray, int], list[TRN]]): 음성 데이터를 받아 list[TRN] 형태로 변환하는 함수
    """

    if destination.exists() and not destination.is_dir():
        print(f"Destination path {destination} is not a directory.")
        return

    data_paths = search_all_data(source, verbose=verbose)
    destination.mkdir(parents=True, exist_ok=True)

    for idx, dirpath in enumerate(data_paths):
        if max_count != -1 and idx >= max_count:
            break

        trans_txt = next(dirpath.glob("*.train.txt"))

        # 목적지 경로 만들기
        rel_dir = dirpath.relative_to(source)
        dst_dir = destination / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        # REF 변환
        ref_trn = dst_dir / (trans_txt.stem.replace(".trans", "") + ".ref.trn")
        hyp_trn = dst_dir / (trans_txt.stem.replace(".trans", "") + ".hyp.trn")

        ref_items = trans_txt_to_sclite_trn(trans_txt, normalizer=normalizer)
        make_trn_file(ref_items, ref_trn)

        # ── 2) 같은 폴더의 flac 순차 변환 ──────────────────
        hyp_items = [
            TRNFormat(id=flac.stem, text=normalizer(transcribe(flac)))
            for flac in sorted(dirpath.glob("*.flac"))
        ]
        make_trn_file(hyp_items, hyp_trn)

        verbose and print(f"✅ {ref_trn} and {hyp_trn} created successfully.")

    verbose and print(f"✅ {destination} 경로에 REF/HYP .trn 모두 생성 완료")


def make_ref_and_hyp(
    data_paths: list[Path],
    transcribe: Callable[[Path], str],
    normalizer: Callable[[str], str] = lambda x: x,
    max_count: int = -1,
    verbose: bool = True,
) -> dict[str, dict[str, list[TRNFormat]]]:
    if not data_paths:
        verbose and print("No data paths provided.")
        return {}
    if not all(p.exists() for p in data_paths):
        verbose and print("One or more data paths do not exist.")
        return {}

    result = {}
    for i, dirpath in enumerate(data_paths):
        if max_count != -1 and i >= max_count:
            break

        trans_txt = next(dirpath.glob("*.trans.txt"))
        ref_items = trans_txt_to_sclite_trn(trans_txt, normalizer=normalizer)
        hyp_items = [
            TRNFormat(id=flac.stem, text=normalizer(transcribe(flac)))
            for flac in sorted(dirpath.glob("*.flac"))
        ]

        result[trans_txt.stem] = {"ref": ref_items, "hyp": hyp_items}

        verbose and print(f"✅ {trans_txt.stem} read successfully.")

    return result


def search_all_data(source: Path, verbose: bool = True) -> list[Path]:
    if not source.exists():
        verbose and print(f"Source path {source} does not exist.")
        return []
    if not source.is_dir():
        verbose and print(f"Source path {source} is not a directory.")
        return []

    data_dirs = []
    for dirpath in (p for p in source.rglob("*") if p.is_dir()):
        trans_files = list(dirpath.glob("*.trans.txt"))
        if len(trans_files) == 0:
            continue
        elif len(trans_files) > 1:
            verbose and print(
                f"Skipping {dirpath} - expected one trans file, found {len(trans_files)}"
            )
            continue
        data_dirs.append(dirpath)

    return data_dirs


__all__ = [
    "trans_txt_to_sclite_trn",
    "generate_all_ref_and_hyp_file",
    "make_ref_and_hyp",
    "search_all_data",
]
