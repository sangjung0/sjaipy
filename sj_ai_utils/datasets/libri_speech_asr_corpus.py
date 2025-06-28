from pathlib import Path


def trans_txt_to_sclite_trn(src: Path, dst: Path):
    folder = dst / src.parent.parent.name / src.parent.name
    folder.mkdir(parents=True, exist_ok=True)
    dst_file = folder / src.name.replace(".txt", ".trn").replace(".trans", ".ref")

    with src.open(encoding="utf-8") as fin, dst_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            uid, *words = line.rstrip().split()
            sent = " ".join(words).upper()
            fout.write(f"{sent} ({uid})\n")
    print(f"Converted {src} to {dst_file}")


def search_trans_txt(root: Path):
    return [p for p in root.rglob("*.trans.txt") if p.is_file()]
