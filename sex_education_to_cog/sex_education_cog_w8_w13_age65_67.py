#!/usr/bin/env python3
"""
Create a one-row-per-participant HRS dataset for participants who:

1. Have a starting age of 65–67, where starting age is the age at their
   earliest available wave in the original merged data.
2. Have a non-missing Cog value in every wave from 8 through 13.

Output columns:
    sex
    education_group
    hhidpn
    Cog_wave8 ... Cog_wave13
    starting_age

Education groups:
    Less than 12 years
    12 years
    13-15 years
    16 or more years
    Missing
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


COG_WAVES = tuple(range(8, 14))


def is_missing(value: Any) -> bool:
    """Return True for blank, NA-like, or non-finite values."""
    if value is None:
        return True
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "n/a", "nan", "none", "null", "."}:
        return True
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


def as_float(value: Any) -> float | None:
    """Convert a value to float; return None when missing or invalid."""
    if is_missing(value):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def first_nonmissing(rows: list[dict[str, str]], column: str) -> str:
    """Return the first non-missing value ordered by wave."""
    for row in sorted(rows, key=lambda r: int(float(r["wave"]))):
        value = row.get(column, "")
        if not is_missing(value):
            return str(value).strip()
    return ""


def education_group(education_value: str) -> str:
    """Convert education years to the requested categories."""
    years = as_float(education_value)
    if years is None:
        return "Missing"
    if years < 12:
        return "Less than 12 years"
    if years == 12:
        return "12 years"
    if years < 16:
        return "13-15 years"
    return "16 or more years"


def normalize_sex(value: str) -> str:
    """Standardize sex labels to Male/Female when possible."""
    text = value.strip().lower()
    if text in {"1", "male", "m"}:
        return "Male"
    if text in {"2", "female", "f"}:
        return "Female"
    return value.strip() if value.strip() else "Missing"


def id_sort_key(hhidpn: str) -> tuple[int, str]:
    """Sort numeric IDs numerically while preserving the original text."""
    try:
        return (0, f"{int(float(hhidpn)):020d}")
    except ValueError:
        return (1, hhidpn)


def build_dataset(input_csv: Path, output_csv: Path) -> dict[str, int]:
    participants: dict[str, list[dict[str, str]]] = defaultdict(list)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"hhidpn", "wave", "age", "Cog", "sex", "education"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Input is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            hhidpn = row["hhidpn"].strip()
            if hhidpn:
                participants[hhidpn].append(row)

    output_rows: list[dict[str, str | float]] = []

    for hhidpn, rows in participants.items():
        rows_by_wave: dict[int, dict[str, str]] = {}
        for row in rows:
            wave_value = as_float(row.get("wave"))
            if wave_value is None:
                continue
            wave = int(wave_value)

            # The source is expected to have one row per participant-wave.
            # If duplicates occur, retain the first row, but replace it when
            # the later duplicate supplies a missing Cog value.
            if wave not in rows_by_wave:
                rows_by_wave[wave] = row
            elif is_missing(rows_by_wave[wave].get("Cog")) and not is_missing(row.get("Cog")):
                rows_by_wave[wave] = row

        # Starting age = valid age at the participant's earliest available wave.
        age_candidates = []
        for row in rows:
            wave_value = as_float(row.get("wave"))
            age_value = as_float(row.get("age"))
            if wave_value is not None and age_value is not None:
                age_candidates.append((int(wave_value), age_value))

        if not age_candidates:
            continue

        starting_age = min(age_candidates, key=lambda item: item[0])[1]
        if not (65 <= starting_age <= 67):
            continue

        # Require valid Cog in every wave 8 through 13.
        if any(
            wave not in rows_by_wave or as_float(rows_by_wave[wave].get("Cog")) is None
            for wave in COG_WAVES
        ):
            continue

        education_value = first_nonmissing(rows, "education")
        sex_value = first_nonmissing(rows, "sex")

        output_row: dict[str, str | float] = {
            "sex": normalize_sex(sex_value),
            "education_group": education_group(education_value),
            "hhidpn": hhidpn,
        }

        for wave in COG_WAVES:
            # Preserve the source representation of Cog rather than rounding it.
            output_row[f"Cog_wave{wave}"] = rows_by_wave[wave]["Cog"].strip()

        output_row["starting_age"] = (
            int(starting_age) if starting_age.is_integer() else starting_age
        )
        output_rows.append(output_row)

    output_rows.sort(key=lambda row: id_sort_key(str(row["hhidpn"])))

    fieldnames = [
        "sex",
        "education_group",
        "hhidpn",
        *[f"Cog_wave{wave}" for wave in COG_WAVES],
        "starting_age",
    ]

    with output_csv.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    sex_counts: dict[str, int] = defaultdict(int)
    education_counts: dict[str, int] = defaultdict(int)
    for row in output_rows:
        sex_counts[str(row["sex"])] += 1
        education_counts[str(row["education_group"])] += 1

    return {
        "participants": len(output_rows),
        "male": sex_counts.get("Male", 0),
        "female": sex_counts.get("Female", 0),
        "education_missing": education_counts.get("Missing", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select age-65–67 participants with complete Cog waves 8–13."
    )
    parser.add_argument("input_csv", type=Path, help="Original merged HRS CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("HRS_age65_67_complete_Cog_w8_w13.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    summary = build_dataset(args.input_csv, args.output)
    print(f"Created: {args.output}")
    print(f"Participants: {summary['participants']}")
    print(f"Male: {summary['male']}")
    print(f"Female: {summary['female']}")
    print(f"Missing education group: {summary['education_missing']}")


if __name__ == "__main__":
    main()
