#!/usr/bin/env python3
"""Create a participant-level HRS cohort dataset.

Cohort eligibility is based on observed loneliness3 values in the alternating
psychosocial waves:
    Cohort A: waves 8, 10, and 12
    Cohort B: waves 9, 11, and 13

A participant is retained only if at least two of the cohort's three
psychosocial waves have non-missing loneliness3 values.

Participant-level output wave mapping:
    Cohort A: T1 = wave 8,  T2 = wave 10, T3 cognition = wave 11
    Cohort B: T1 = wave 9,  T2 = wave 11, T3 cognition = wave 12

T2 covariates and demographics are taken from the participant's actual T2
source row. No synthetic values are created. If a requested row or value is
absent, the corresponding output field is blank. Difference fields are "n/a"
when either required value is missing.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# define the cohorts waves
COHORTS = {
    "A": {
        "eligibility_waves": (8, 10, 12),
        "t1_wave": 8,
        "t2_wave": 10,
        "t3_cog_wave": 11,
    },
    "B": {
        "eligibility_waves": (9, 11, 13),
        "t1_wave": 9,
        "t2_wave": 11,
        "t3_cog_wave": 12,
    },
}

# define the output columns of the loneliness-cog table
OUTPUT_COLUMNS = [
    "cohort",
    "hhidpn",
    "T1 loneliness",
    "T2 loneliness",
    "Loneliness T2-T1",
    "lives_alone at T2",
    "cog T2",
    "cog T3",
    "cog T3-T2",
    "age T2",
    "sex",
    "education",
    "race",
    "hispanic",
    "race_ethnicity",
    "marital_status T2",
]

SUMMARY_COLUMNS = [
    "cohort",
    "retained_participants",
    "complete_T1_T2_loneliness",
    "complete_T2_T3_cog",
    "complete_both_loneliness_and_cog_pairs",
]

REQUIRED_SOURCE_COLUMNS = {
    "hhidpn",
    "wave",
    "loneliness3",
    "lives_alone",
    "Cog",
    "age",
    "sex",
    "education",
    "race",
    "hispanic",
    "race_ethnicity",
    "marital_status",
}


def is_present(value: object) -> bool:
    """Return True when a CSV value is not blank."""
    return value is not None and str(value).strip() != ""


def clean(value: object) -> str:
    """Return a stripped CSV string, or an empty string for missing values."""
    return "" if value is None else str(value).strip()


def decimal_difference(later: object, earlier: object) -> str:
    """Return later - earlier, or 'n/a' when either value is missing/non-numeric."""
    if not is_present(later) or not is_present(earlier):
        return "n/a"

    try:
        result = Decimal(clean(later)) - Decimal(clean(earlier))
    except InvalidOperation:
        return "n/a"

    # Avoid scientific notation and remove unnecessary trailing zeroes.
    text = format(result, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def participant_sort_key(hhidpn: str) -> Tuple[int, object, str]:
    """Sort numeric HRS IDs numerically while preserving their original text."""
    try:
        return (0, int(hhidpn), hhidpn)
    except ValueError:
        return (1, hhidpn, hhidpn)


def read_source(path: Path) -> Tuple[List[str], Dict[Tuple[str, int], Dict[str, str]]]:
    """Read the source CSV and index rows by (hhidpn, wave)."""
    rows_by_key: Dict[Tuple[str, int], Dict[str, str]] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The input CSV has no header row.")

        missing_columns = REQUIRED_SOURCE_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for line_number, row in enumerate(reader, start=2):
            hhidpn = clean(row.get("hhidpn"))
            wave_text = clean(row.get("wave"))
            if not hhidpn or not wave_text:
                continue

            try:
                wave = int(Decimal(wave_text))
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(
                    f"Invalid wave value {wave_text!r} on CSV line {line_number}."
                ) from exc

            key = (hhidpn, wave)
            if key in rows_by_key:
                raise ValueError(
                    "Duplicate participant-wave record found for "
                    f"hhidpn={hhidpn}, wave={wave}."
                )
            rows_by_key[key] = {name: clean(value) for name, value in row.items()}

    return list(reader.fieldnames), rows_by_key


def determine_membership(
    rows_by_key: Mapping[Tuple[str, int], Mapping[str, str]]
) -> Dict[str, List[str]]:
    """Return eligible participant IDs for each cohort."""
    all_ids = sorted(
        {hhidpn for hhidpn, _wave in rows_by_key}, key=participant_sort_key
    )
    members: Dict[str, List[str]] = {}

    for cohort, config in COHORTS.items():
        eligibility_waves: Sequence[int] = config["eligibility_waves"]
        eligible: List[str] = []

        for hhidpn in all_ids:
            valid_loneliness_count = sum(
                is_present(rows_by_key.get((hhidpn, wave), {}).get("loneliness3"))
                for wave in eligibility_waves
            )
            if valid_loneliness_count >= 2:
                eligible.append(hhidpn)

        members[cohort] = eligible

    overlap = set(members["A"]).intersection(members["B"])
    if overlap:
        examples = ", ".join(sorted(overlap, key=participant_sort_key)[:10])
        raise ValueError(
            f"{len(overlap)} participants qualify for both cohorts, which would "
            "violate the one-record-per-hhidpn requirement. Examples: " + examples
        )

    return members


def build_participant_row(
    cohort: str,
    hhidpn: str,
    rows_by_key: Mapping[Tuple[str, int], Mapping[str, str]],
) -> Dict[str, str]:
    """Build one participant-level output record."""
    config = COHORTS[cohort]
    t1 = rows_by_key.get((hhidpn, config["t1_wave"]), {})
    t2 = rows_by_key.get((hhidpn, config["t2_wave"]), {})
    t3 = rows_by_key.get((hhidpn, config["t3_cog_wave"]), {})

    t1_loneliness = clean(t1.get("loneliness3"))
    t2_loneliness = clean(t2.get("loneliness3"))
    cog_t2 = clean(t2.get("Cog"))
    cog_t3 = clean(t3.get("Cog"))

    return {
        "cohort": cohort,
        "hhidpn": hhidpn,
        "T1 loneliness": t1_loneliness,
        "T2 loneliness": t2_loneliness,
        "Loneliness T2-T1": decimal_difference(t2_loneliness, t1_loneliness),
        "lives_alone at T2": clean(t2.get("lives_alone")),
        "cog T2": cog_t2,
        "cog T3": cog_t3,
        "cog T3-T2": decimal_difference(cog_t3, cog_t2),
        "age T2": clean(t2.get("age")),
        "sex": clean(t2.get("sex")),
        "education": clean(t2.get("education")),
        "race": clean(t2.get("race")),
        "hispanic": clean(t2.get("hispanic")),
        "race_ethnicity": clean(t2.get("race_ethnicity")),
        "marital_status T2": clean(t2.get("marital_status")),
    }


def create_outputs(
    input_csv: Path,
    output_csv: Path,
    summary_csv: Path,
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]]]:
    """Create the participant-level data and cohort completeness summary."""
    _headers, rows_by_key = read_source(input_csv)
    members = determine_membership(rows_by_key)

    output_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, object]] = []

    for cohort in ("A", "B"):
        cohort_rows = [
            build_participant_row(cohort, hhidpn, rows_by_key)
            for hhidpn in members[cohort]
        ]
        output_rows.extend(cohort_rows)

        complete_loneliness = sum(
            is_present(row["T1 loneliness"]) and is_present(row["T2 loneliness"])
            for row in cohort_rows
        )
        complete_cog = sum(
            is_present(row["cog T2"]) and is_present(row["cog T3"])
            for row in cohort_rows
        )
        complete_both = sum(
            is_present(row["T1 loneliness"])
            and is_present(row["T2 loneliness"])
            and is_present(row["cog T2"])
            and is_present(row["cog T3"])
            for row in cohort_rows
        )

        summary_rows.append(
            {
                "cohort": cohort,
                "retained_participants": len(cohort_rows),
                "complete_T1_T2_loneliness": complete_loneliness,
                "complete_T2_T3_cog": complete_cog,
                "complete_both_loneliness_and_cog_pairs": complete_both,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    return output_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create participant-level HRS psychosocial cohorts."
    )
    parser.add_argument("input_csv", type=Path, help="Merged HRS source CSV")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("HRS_participant_level_cohorts.csv"),
        help="Combined participant-level cohort CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("HRS_cohort_completeness_summary.csv"),
        help="Cohort completeness summary CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_rows, summary_rows = create_outputs(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
    )

    print(f"Wrote {len(output_rows):,} participant records to {args.output_csv}")
    print(f"Wrote cohort summary to {args.summary_csv}")
    for row in summary_rows:
        print(
            f"Cohort {row['cohort']}: retained={row['retained_participants']:,}, "
            f"complete loneliness pair={row['complete_T1_T2_loneliness']:,}, "
            f"complete cognition pair={row['complete_T2_T3_cog']:,}, "
            f"complete both={row['complete_both_loneliness_and_cog_pairs']:,}"
        )


if __name__ == "__main__":
    main()
