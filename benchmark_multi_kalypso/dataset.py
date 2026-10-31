import csv
import shutil
import warnings
from pathlib import Path

from .config import Experiment, WorkloadConfig
from .plans import WORKLOADS, Workload, to_request


def text_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.txt"), key=lambda p: int(p.stem))


def count_rows(path: str | Path) -> int:
    path = Path(path)
    if path.is_dir():
        return len(text_files(path))
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV requires a header")
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
        source.open(newline="", encoding="utf-8-sig") as src,
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


def source_paths(
    entry: WorkloadConfig, repo_root: str | Path
) -> tuple[Workload, Path, dict[int, Path]]:
    try:
        workload = WORKLOADS[entry.name]
    except KeyError:
        raise ValueError(
            f"Unknown workload: {entry.name}. Valid workloads: {', '.join(WORKLOADS)}"
        ) from None
    root = (
        Path(repo_root) / entry.data_root
        if entry.data_root is not None
        else workload.data_root
    ).resolve()
    return (
        workload,
        root / workload.data_path,
        {index: root / join.table for index, join in workload.joins},
    )


def preflight(
    config: Experiment, repo_root: str | Path
) -> dict[str, dict[str, tuple[int, int]]]:
    counts = {}
    for entry in config.workloads:
        workload, data, tables = source_paths(entry, repo_root)
        checks = [("rows", data, entry.rows)] + [
            (f"table_{index}", tables[index], join.rows)
            for index, join in workload.joins
        ]
        counts[entry.name] = {}
        for label, path, needed in checks:
            try:
                available = count_rows(path)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Data preflight failed for {entry.name} {label}: {path}: {exc}"
                ) from exc
            counts[entry.name][label] = (available, needed)
    return counts


def prepare_inputs(
    config: Experiment, repo_root: str | Path, out: Path
) -> dict[str, tuple[dict, int, int | None]]:
    prepared = {}
    for entry in config.workloads:
        workload, source, tables = source_paths(entry, repo_root)
        folder = out / "data" / entry.name
        data = materialize_table(
            source,
            entry.rows,
            folder / "input",
            allow_fewer_rows=config.allow_fewer_rows,
        )
        right_tables = {
            index: materialize_table(
                tables[index],
                join.rows,
                folder / f"table_{index}",
                allow_fewer_rows=config.allow_fewer_rows,
            )
            for index, join in workload.joins
        }
        body = to_request(workload, data, right_tables, config.model)
        right_rows = (
            sum(count_rows(path) for path in right_tables.values())
            if right_tables
            else None
        )
        prepared[entry.name] = body, count_rows(data), right_rows
    return prepared
