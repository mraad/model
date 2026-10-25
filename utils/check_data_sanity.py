"""Find unreadable or malformed chip .npz files, and optionally delete them.

    python utils/check_data_sanity.py DIR [--chips 128] [--delete] [--workers 24]

Dry run by default: lists invalid files. A file is invalid when it cannot be
read, has no `pixels` array, or (with --chips) holds another number of chips.
utils/split_npz.py writes 32-chip files, so pass --chips 32 after splitting.
"""

import argparse
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def problem(file_path, chips):
    """Return why the file is invalid, or None when it is valid."""
    try:
        with np.load(file_path) as data:
            if "pixels" not in data:
                return "'pixels' key missing"
            n = data["pixels"].shape[0]
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        return f"unreadable: {e}"
    if chips is not None and n != chips:
        return f"{n} chips, expected {chips}"
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Find (and delete) invalid chip .npz files."
    )
    ap.add_argument("directory")
    ap.add_argument("--chips", type=int, help="expected chips per file")
    ap.add_argument("--delete", action="store_true", help="delete invalid files")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.directory)
        for f in files
        if f.endswith(".npz")
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        reasons = executor.map(lambda p: problem(p, args.chips), paths)
        invalid = [(p, r) for p, r in zip(paths, reasons) if r]

    for path, reason in invalid:
        if args.delete:
            os.remove(path)
        print(f"{'Deleted' if args.delete else 'Invalid'}: {path} ({reason})")
    print(f"{len(invalid)} of {len(paths)} .npz files invalid", end="")
    print("; deleted." if args.delete and invalid else ".")


if __name__ == "__main__":
    main()
