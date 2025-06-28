from __future__ import annotations
from typing import TYPE_CHECKING
import re
from dataclasses import dataclass, field
import subprocess
from pathlib import Path

if TYPE_CHECKING:
    pass


@dataclass
class TRNFormat:
    id: str
    text: str
    # id: str = field(...)
    # text: str = field(...)


def __subprocess_run(cmd: list[str], workdir: Path = None) -> None:
    try:
        process = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, check=True
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        raise e


def sclite_trn_file(
    ref: Path,
    hyp: Path,
    output: Path,
    ignore_case: bool = True,
    output_format: list[str] = ["sum", "prf", "dtl", "sgml"],
    verbose: bool = True,
) -> None:
    """sclite의 trn 평가 파일을 생성하는 함수

    Args:
        ref (Path): 정답 trn 파일
        hyp (Path): 예측된 trn 파일
        output (Path): 생성 결과가 저장될 폴더 경로
        ignore_case (bool, optional): 대소문자 구분 여부. Defaults to True.
        output_format (list[str], optional): 출력 형식. Defaults to ["sum", "prf", "dtl", "sgml"].

    Raises:
        IsADirectoryError: 출력 경로가 파일이 아닌 디렉토리일 때
    """

    if output.exists() and not output.is_dir():
        raise IsADirectoryError(f"Output path {output} is not a directory.")
    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sclite",
        "-r",
        str(ref.resolve()),
        "trn",
        "-h",
        str(hyp.resolve()),
    ]

    ignore_case and cmd.extend(["-i", "rm"])
    cmd.extend([o for f in output_format for o in ["-o", f]])
    __subprocess_run(cmd, workdir=output)

    verbose and print(f"Sclite evaluation completed. Results saved to {output}")


def sclite_trn_run(
    ref: Path, hyp: Path, ignore_case: bool = True, output_format: list[str] = ["sum"]
) -> str:
    """sclite의 trn 평가를 반환하는 함수

    Args:
        ref (Path): 정답 trn 파일
        hyp (Path): 예측된 trn 파일
        ignore_case (bool, optional): 대소문자 구분 여부. Defaults to True.

    Returns:
        str: sclite 평가 결과 문자열
    """

    cmd = [
        "sclite",
        "-r",
        str(ref.resolve()),
        "trn",
        "-h",
        str(hyp.resolve()),
    ]
    ignore_case and cmd.extend(["-i", "rm"])
    cmd.extend(["-o", "stdout"])
    cmd.extend([o for f in output_format for o in ["-o", f]])
    return __subprocess_run(cmd)


def concat_trn_file(source: list[Path], dest: Path) -> None:
    """trn 리스트를 받아 하나의 trn 파일로 합치는 함수

    Args:
        source (list[Path]): 합칠 trn 파일들의 리스트
        dest (Path): 생성 결과가 저장될 파일 경로

    Raises:
        NotADirectoryError: dest가 디렉토리일 때 발생
    """
    if dest.exists() and not dest.is_file():
        raise NotADirectoryError(f"Destination {dest} is a directory, not a file.")

    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("w", encoding="utf-8", newline="\n") as fout:
        for item in source:
            with item.open("r", encoding="utf-8") as fin:
                fout.write(fin.read().replace("\r\n", "\n").replace("\r", "\n"))


def make_trn_file(trn_list: list[TRNFormat], path: Path) -> None:
    """trn 리스트를 받아 sclite의 trn 평가 파일을 생성하는 함수

    Args:
        trn_list (list[TRNFormat]): trn 평가 항목 리스트
        path (Path): 생성 결과가 저장될 파일 경로
    """

    if isinstance(path, Path) and path.exists() and path.is_dir():
        raise NotADirectoryError(f"Path {path} is a directory, not a file.")

    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", encoding="utf-8", newline="\n") as fout:
        for item in trn_list:
            fout.write(f"{item.text.strip().upper()} ({item.id})\n")


def parse_sclite_summary(output: str) -> dict[str, int | float]:
    """
    sclite 의 sum 출력 파싱

    Args:
        output (str): sclite 명령의 전체 출력 문자열.

    Returns:
        dict[str, int | float]: 파싱된 결과를 포함하는 딕셔너리.
    """

    summary_pattern = re.compile(
        r"Sum/Avg\s+\|\s*(\d+)\s*(\d+)\s+\|\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)"
    )

    match = summary_pattern.search(output)

    if match:
        return {
            "num_sentences": int(match.group(1)),
            "num_words": int(match.group(2)),
            "correct_percent": float(match.group(3)),
            "substitution_percent": float(match.group(4)),
            "deletion_percent": float(match.group(5)),
            "insertion_percent": float(match.group(6)),
            "wer_percent": float(match.group(7)),  # WER을 명확히 명시
            "sentence_error_percent": float(match.group(8)),
        }
    else:
        raise ValueError("Could not parse sclite summary output.")
