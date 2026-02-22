"""Migrate legacy group labels in CSV files.

Converts group column values:
- OUR / our -> 자사
- COMPETITOR / competitor -> 경쟁사
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

REPLACEMENTS = {
    "OUR": "자사",
    "our": "자사",
    "COMPETITOR": "경쟁사",
    "competitor": "경쟁사",
}


def find_group_index(header: list[str]) -> int | None:
    for idx, col in enumerate(header):
        name = str(col).replace("\ufeff", "").strip().lower()
        if name == "group":
            return idx
    return None


def migrate_csv(path: Path, dry_run: bool = False) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return 0, 0

    group_idx = find_group_index(rows[0])
    if group_idx is None:
        return 0, 0

    changed = 0
    for row in rows[1:]:
        if group_idx >= len(row):
            continue
        old = row[group_idx].strip()
        new = REPLACEMENTS.get(old)
        if new and new != row[group_idx]:
            row[group_idx] = new
            changed += 1

    if changed > 0 and not dry_run:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tf:
            writer = csv.writer(tf)
            writer.writerows(rows)
            temp_path = Path(tf.name)
        temp_path.replace(path)

    return changed, len(rows) - 1


def collect_csv_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.csv") if p.is_file()))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate group labels in CSV files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests", "data"],
        help="CSV file(s) or directories (recursive) to process. Default: tests data",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    args = parser.parse_args()

    targets = collect_csv_files([Path(p) for p in args.paths])
    if not targets:
        print("No CSV files found.")
        return

    total_changed = 0
    changed_files = 0

    for file_path in targets:
        changed, _ = migrate_csv(file_path, dry_run=args.dry_run)
        if changed > 0:
            changed_files += 1
            total_changed += changed
            print(f"[UPDATED] {file_path}: {changed} row(s)")

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"[{mode}] files_changed={changed_files}, rows_changed={total_changed}")


if __name__ == "__main__":
    main()
