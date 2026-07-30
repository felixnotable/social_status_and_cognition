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
# define the text the represent missing data
MISSING_TEXT = {"", ".", "na", "n/a", "nan", "none", "null"}


def is_nonmissing(value: object) -> bool:
    """Return True when a CSV value represents observed data."""
    if value is None:
        return False
    return str(value).strip().lower() not in MISSING_TEXT


def hhidpn_sort_key(value: str) -> Tuple[int, object]:
    """Sort numeric HRS identifiers numerically while preserving their text."""
    stripped = value.strip()
    try:
        return (0, int(stripped))
    except ValueError:
        return (1, stripped)


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


def qualifying_ids(
    rows_by_id: Mapping[str, Mapping[str, Mapping[str, str]]],
    wave_cycle_pairs: Sequence[Tuple[str, str]],
    minimum_loneliness_waves: int = 2,
) -> List[str]:
    """
    Return IDs with loneliness3 observed in at least the required waves.
    input args:
        rows_by_id: the hhidpn+wave indexed data
        wave_cycle_pairs: wave to cycle mapping. Cohort A and B have different mapping 
        minimum_loneliness_waves: select hhidpn who has at least 2 cycles.
    """
    # target waves for the current cohort. e.g. cohort A: target_waves= 8, 10, 12
    target_waves = [wave for wave, _ in wave_cycle_pairs]
    eligible: List[str] = []

    for hhidpn, wave_rows in rows_by_id.items():
        observed = sum(
            # for each wave, check whether loneliness3 has valid data
            is_nonmissing(wave_rows.get(wave, {}).get("loneliness3"))
            for wave in target_waves
        )
        if observed >= minimum_loneliness_waves:
            eligible.append(hhidpn)

    return sorted(eligible, key=hhidpn_sort_key)


def make_output_row(
    hhidpn: str,
    cycle: str,
    source_row: Mapping[str, str] | None,
    fields: Sequence[str],
) -> Dict[str, str]:
    """Create one cohort-cycle row without imputing absent values."""
    output = {field: "" for field in fields}
    output["hhidpn"] = hhidpn
    output["cycle"] = cycle

    if source_row is not None:
        for field in fields:
            if field not in {"hhidpn", "cycle"}:
                output[field] = source_row.get(field, "")

    return output


def write_cohort(
    cohort_name: str,
    wave_cycle_pairs: Sequence[Tuple[str, str]],
    rows_by_id: Mapping[str, Mapping[str, Mapping[str, str]]],
    fields: Sequence[str],
    output_path: Path,
) -> Dict[str, object]:
    """
    Write one cohort CSV and return validation statistics.
    input args:
        cohort_name: A or B
        wave_cycle_pairs: wave to cycle pairs, e.g. wave 8 -> T1
        rows_by_id: hhidpn+wave indexed data
        fields: output fields with wave+year being replaced with cycle
        output_path: the location for output file
    """
    # get the eligible hhidpns for this cohort
    ids = qualifying_ids(rows_by_id, wave_cycle_pairs)
    synthetic_rows = 0
    loneliness_counts: Counter[int] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for hhidpn in ids:
            wave_rows = rows_by_id[hhidpn]
            observed_loneliness = 0

            for wave, cycle in wave_cycle_pairs:
                source_row = wave_rows.get(wave)
                if source_row is None:
                    # this hhidpn misses the wave, make an empty row. 
                    synthetic_rows += 1
                elif is_nonmissing(source_row.get("loneliness3")):
                    observed_loneliness += 1

                writer.writerow(
                    make_output_row(
                        hhidpn=hhidpn,
                        cycle=cycle,
                        source_row=source_row,
                        fields=fields,
                    )
                )

            loneliness_counts[observed_loneliness] += 1

    expected_rows = len(ids) * 3
    return {
        "cohort": cohort_name,
        "participants": len(ids),
        "rows": expected_rows,
        "synthetic_missing_wave_rows": synthetic_rows,
        "loneliness_observations_per_participant": dict(sorted(loneliness_counts.items())),
        "output": str(output_path),
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
    source_fields, rows_by_id = read_source(args.input_csv)
    fields = output_fieldnames(source_fields)

    # filter the data to discard hhidpn with less than 2 cycles. 
    # also organize the eligible hhidpns into 2 cohorts
    eligible_by_cohort = {
        name: set(qualifying_ids(rows_by_id, pairs))
        for name, pairs in COHORTS.items()
    }
    # check whether cocort A and B have overlap. 
    overlap = eligible_by_cohort["A"].intersection(eligible_by_cohort["B"])
    if overlap:
        examples = ", ".join(sorted(overlap, key=hhidpn_sort_key)[:10])
        raise ValueError(
            f"{len(overlap)} participants qualify for both cohorts. "
            f"Examples: {examples}. A manual cohort-assignment rule is required."
        )

    summaries = []
    for cohort_name, wave_cycle_pairs in COHORTS.items():
        output_path = args.output_dir / f"HRS_psychosocial_cohort_{cohort_name}.csv"
        summaries.append(
            write_cohort(
                cohort_name=cohort_name,
                wave_cycle_pairs=wave_cycle_pairs,
                rows_by_id=rows_by_id,
                fields=fields,
                output_path=output_path,
            )
        )

    print(f"Source participants: {len(rows_by_id):,}")
    print(f"Output columns ({len(fields)}): {', '.join(fields)}")


if __name__ == "__main__":
    main()
