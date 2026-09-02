#!/usr/bin/env python3
"""Rename Re_<integer>.csv files to Re_<integer>.0.csv in a folder."""

import argparse
import re
from pathlib import Path


INTEGER_RE_CSV = re.compile(r"^(Re_)([+-]?\d+)(\.csv)$")


def rename_files(folder: Path, dry_run: bool) -> int:
    renamed = 0
    for source in sorted(folder.iterdir()):
        if not source.is_file():
            continue

        match = INTEGER_RE_CSV.fullmatch(source.name)
        if not match:
            continue

        destination = source.with_name(f"{match.group(1)}{match.group(2)}.0{match.group(3)}")
        if destination.exists():
            print(f"Skipping {source.name}: {destination.name} already exists")
            continue

        print(f"{source.name} -> {destination.name}")
        if not dry_run:
            source.rename(destination)
        renamed += 1

    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename Re_<integer>.csv files, e.g. Re_38.csv to Re_38.0.csv."
    )
    parser.add_argument("folder", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true", help="Show intended renames only.")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Not a folder: {args.folder}")

    count = rename_files(args.folder, args.dry_run)
    action = "Would rename" if args.dry_run else "Renamed"
    print(f"{action} {count} file(s).")


if __name__ == "__main__":
    main()