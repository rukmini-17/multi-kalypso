import csv
import shutil
import warnings
from pathlib import Path


def text_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.txt"), key=lambda p: int(p.stem))


def count_rows(path: str | Path) -> int:
    path = Path(path)
    if path.is_dir():
        return len(text_files(path))
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "data" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: CSV requires a data column")
        return sum(1 for _ in reader)


def materialize_table(
    source: Path, rows: int, out_dir: Path, *, allow_fewer_rows: bool = False
) -> Path:
    if rows < 1:
        raise ValueError("rows must be positive")
    available = count_rows(source)
    if available < rows:
        message = f"{source}: requested {rows} rows, found {available}"
        if not allow_fewer_rows or available == 0:
            raise ValueError(message)
        warnings.warn(message + "; using available rows", stacklevel=2)
    out_dir.mkdir(parents=True, exist_ok=False)
    if source.is_dir():
        for path in text_files(source)[:rows]:
            shutil.copy2(path, out_dir / path.name)
        return out_dir.resolve()
    target = out_dir / "input.csv"
    with (
        source.open(newline="", encoding="utf-8") as src,
        target.open("w", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for index, row in enumerate(reader):
            if index >= rows:
                break
            writer.writerow(row)
    return target.resolve()
