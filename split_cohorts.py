#!/usr/bin/env python3
"""Split a merged HRS respondent-wave CSV into alternating psychosocial cohorts.

Cohort A uses HRS waves 8, 10, and 12, relabeled T1, T2, and T3.
Cohort B uses HRS waves 9, 11, and 13, relabeled T1, T2, and T3.

A participant is retained in a cohort only when loneliness3 is nonmissing in at
least two of that cohort's three waves. Every retained participant receives
exactly three output rows. If a target respondent-wave row is absent from the
source, the script creates a row containing hhidpn and cycle while leaving all
other variables blank. No missing values are imputed.

The source columns ``wave`` and ``year`` are removed. Every other source
variable is retained in its original order, with ``cycle`` inserted immediately
after ``hhidpn``.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# define the cohorts, and the corresponding wave to cycle mapping. 
# there are 2 cohorts A and B, 
# cohort A has wave 8 -> T1, wave 10 -> T2, wave 12 -> T3
# cohort B has wave 9 -> T1, wave 11 -> T2, wave 13 -> T3
COHORTS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "A": (("8", "T1"), ("10", "T2"), ("12", "T3")),
    "B": (("9", "T1"), ("11", "T2"), ("13", "T3")),
}


def read_source(
    input_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, str]]]]:
    """Read and index source rows by hhidpn and wave."""
    # rows_by_id is in hierarchical dict of hhidpn->wave->variables, e.g.  
    # hhidpn1: {{wave8:variables dict}, {wave10:variables dict}, {wave12:variables dict}}
    # hhidpn2: {{wave8:variables dict}, {wave10:variables dict}, {wave12:variables dict}}
    # ...
    rows_by_id: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)

    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # CSV sanity check
        if reader.fieldnames is None:
            raise ValueError("The input CSV does not contain a header row.")

        # get the original fields names in the CSV.
        fieldnames = list(reader.fieldnames)

        # required fields sanity check
        required = {"hhidpn", "wave", "year", "loneliness3"}
        missing_required = sorted(required.difference(fieldnames))
        if missing_required:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(missing_required)
            )

        # read CSV row by row, covert CSV to dict
        for line_number, row in enumerate(reader, start=2):
            hhidpn = (row.get("hhidpn") or "").strip()
            wave = (row.get("wave") or "").strip()
            if not hhidpn:
                raise ValueError(f"Missing hhidpn on source line {line_number}.")
            if not wave:
                raise ValueError(f"Missing wave on source line {line_number}.")
            if wave in rows_by_id[hhidpn]:
                raise ValueError(
                    f"Duplicate respondent-wave key ({hhidpn}, {wave}) "
                    f"on source line {line_number}."
                )
            rows_by_id[hhidpn][wave] = dict(row)

    return fieldnames, dict(rows_by_id)


def output_fieldnames(source_fieldnames: Sequence[str]) -> List[str]:
    """
    Replace wave/year with cycle and retain all other source columns.
    Input is the original fields names in the CSV
    Output is the names with cycle
    """
    retained = [
        name
        for name in source_fieldnames
        if name not in {"hhidpn", "wave", "year"}
    ]
    return ["hhidpn", "cycle", *retained]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split merged HRS data into alternating psychosocial cohorts."
    )
    parser.add_argument("input_csv", type=Path, help="Merged respondent-wave CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for the two cohort CSV files (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_fields, rows_by_id = read_source(args.input_csv)
    fields = output_fieldnames(source_fields)


if __name__ == "__main__":
    main()
