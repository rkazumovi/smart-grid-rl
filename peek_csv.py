"""
Tiny cross-shell utility: print the header + first few data rows of a CSV.
Works identically in cmd.exe and PowerShell (unlike Get-Content, head, etc.)
since it's just a Python script.

Usage:
    python peek_csv.py data\\raw\\elia_load.csv
    python peek_csv.py data\\raw\\elia_load.csv --lines 5
"""
import argparse
import sys


def peek(path: str, n_lines: int = 3):
    try:
        # utf-8-sig strips a leading BOM if present -- common in CSVs exported from
        # European government/utility open-data portals (Elia included).
        with open(path, encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                print(f"[line {i}] {line.rstrip()}")
    except FileNotFoundError:
        print(f"File not found: {path}")
        print("Check the path is correct and the file was actually saved there.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--lines", type=int, default=3)
    args = parser.parse_args()
    peek(args.path, args.lines)