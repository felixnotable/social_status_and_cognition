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
from pathlib import Path
from typing import Mapping, Tuple

# define the cohorts, and the corresponding wave to cycle mapping. 
# there are 2 cohorts A and B, 
# cohort A has wave 8 -> T1, wave 10 -> T2, wave 12 -> T3
# cohort B has wave 9 -> T1, wave 11 -> T2, wave 13 -> T3
COHORTS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "A": (("8", "T1"), ("10", "T2"), ("12", "T3")),
    "B": (("9", "T1"), ("11", "T2"), ("13", "T3")),
}


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


if __name__ == "__main__":
    main()
