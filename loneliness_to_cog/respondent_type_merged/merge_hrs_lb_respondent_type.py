#!/usr/bin/env python3
"""
Merge six HRS Core Leave-Behind respondent files (2006-2016).

Required input files
--------------------
H06LB_R.da, H06LB_R.dct
H08LB_R.da, H08LB_R.dct
H10LB_R.da, H10LB_R.dct
H12LB_R.da, H12LB_R.dct
H14LB_R.da, H14LB_R.dct
H16LB_R.da, H16LB_R.dct

Main output
-----------
HRS_LB_respondent_type_w8_w13_wide.csv

The main output contains one row per respondent and includes:
- hhidpn
- hhidpn_padded
- hhid
- pn
- respondent_type_w8 through respondent_type_w13
- source and audit fields for each wave

The harmonized respondent_type_wN variables are based on the wave-specific
"WHO ANSWERED THE QUESTIONS" item. Only source values 1 and 2 are retained.
All other source values are written as blank.

Wave mapping
------------
2006 = wave 8
2008 = wave 9
2010 = wave 10
2012 = wave 11
2014 = wave 12
2016 = wave 13

Usage
-----
python merge_hrs_lb_respondent_type.py \
    --input-dir /path/to/lb_files \
    --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONFIG: dict[int, dict[str, Any]] = {
    2006: {
        "wave": 8,
        "stem": "H06LB_R",
        "who_answered": "KLB051",
        "lb_rtype": "KLBRTYPE",
        "eligibility": "KLBELIG",
        "completion_mode": "KLBCOMP",
    },
    2008: {
        "wave": 9,
        "stem": "H08LB_R",
        "who_answered": "LLB051",
        "lb_rtype": "LLBRTYPE",
        "eligibility": "LLBELIG",
        "completion_mode": "LLBCOMP",
    },
    2010: {
        "wave": 10,
        "stem": "H10LB_R",
        "who_answered": "MLB051",
        "lb_rtype": "MLBRTYPE",
        "eligibility": "MLBELIG",
        "completion_mode": "MLBCOMP",
    },
    2012: {
        "wave": 11,
        "stem": "H12LB_R",
        "who_answered": "NLB085",
        "lb_rtype": "NLBRTYPE",
        "eligibility": "NLBELIG",
        "completion_mode": "NLBCOMP",
    },
    2014: {
        "wave": 12,
        "stem": "H14LB_R",
        "who_answered": "OLB077",
        "lb_rtype": "OLBRTYPE",
        "eligibility": "OLBELIG",
        "completion_mode": "OLBCOMP",
    },
    2016: {
        "wave": 13,
        "stem": "H16LB_R",
        "who_answered": "PLB077",
        "lb_rtype": "PLBRTYPE",
        "eligibility": "PLBELIG",
        "completion_mode": "PLBCOMP",
    },
}

WAVES = tuple(range(8, 14))

DCT_PATTERN = re.compile(
    r"_column\((\d+)\)\s+"
    r"(\w+)\s+"
    r"(\w+)\s+"
    r"%(\d+)(?:\.\d+)?[A-Za-z]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HRS 2006-2016 Leave-Behind respondent files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Folder containing the six .da/.dct file pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Folder in which output files will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="HRS_LB_respondent_type_w8_w13",
        help="Filename prefix for the wide and long CSV outputs.",
    )
    return parser.parse_args()


def resolve_file(folder: Path, stem: str, extension: str) -> Path:
    """
    Resolve an input file case-insensitively.

    This accepts filenames such as:
    H16LB_R.da, H16LB_R.DA, H16LB_R.dct, or H16LB_R.DCT.
    """
    direct_candidates = [
        folder / f"{stem}{extension.lower()}",
        folder / f"{stem}{extension.upper()}",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    target_name = f"{stem}{extension}".lower()
    for candidate in folder.iterdir():
        if candidate.is_file() and candidate.name.lower() == target_name:
            return candidate

    raise FileNotFoundError(
        f"Could not find {stem}{extension} in {folder.resolve()}"
    )


def parse_dct(path: Path) -> dict[str, dict[str, Any]]:
    """
    Parse fixed-width positions from an HRS Stata dictionary file.

    Returns a dictionary keyed by uppercase variable name. Positions are
    converted from Stata's 1-based columns to Python's 0-based string indexes.
    """
    specs: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="latin-1", errors="replace") as source:
        for line in source:
            match = DCT_PATTERN.search(line)
            if not match:
                continue

            start, storage_type, name, width = match.groups()
            specs[name.upper()] = {
                "start": int(start) - 1,
                "width": int(width),
                "storage_type": storage_type,
            }

    if not specs:
        raise ValueError(f"No variable definitions were parsed from {path}")

    return specs


def read_fixed_width_field(
    line: str,
    specs: dict[str, dict[str, Any]],
    variable: str,
) -> str:
    variable = variable.upper()
    if variable not in specs:
        raise KeyError(f"{variable} is not defined in the dictionary.")

    spec = specs[variable]
    start = spec["start"]
    width = spec["width"]
    return line[start : start + width].strip()


def integer_or_blank(value: str) -> int | str:
    """
    Convert an integer-looking field to int; otherwise return a blank string.
    """
    value = value.strip()
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    return ""


def build_hhidpn(hhid: str, pn: str) -> tuple[int, str]:
    """
    Return both numeric-style and zero-preserving versions of HHIDPN.

    Example:
        HHID='000003', PN='010'
        numeric-style hhidpn = 3010
        hhidpn_padded = '000003010'
    """
    hhidpn_padded = f"{hhid}{pn}"

    if not hhidpn_padded.isdigit():
        raise ValueError(
            f"HHID and PN did not form a numeric identifier: "
            f"HHID={hhid!r}, PN={pn!r}"
        )

    return int(hhidpn_padded), hhidpn_padded


def validate_required_variables(
    year: int,
    specs: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    required = [
        "HHID",
        "PN",
        config["who_answered"],
        config["lb_rtype"],
        config["eligibility"],
        config["completion_mode"],
    ]

    missing = [name for name in required if name.upper() not in specs]
    if missing:
        raise KeyError(
            f"{year}: required variables missing from dictionary: {missing}"
        )


def read_wave(
    year: int,
    config: dict[str, Any],
    input_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stem = config["stem"]
    data_path = resolve_file(input_dir, stem, ".da")
    dct_path = resolve_file(input_dir, stem, ".dct")

    specs = parse_dct(dct_path)
    validate_required_variables(year, specs, config)

    rows: list[dict[str, Any]] = []
    identifiers_seen: set[str] = set()
    duplicate_identifiers = 0

    distributions = {
        "who_answered_raw": Counter(),
        "respondent_type": Counter(),
        "lb_rtype_raw": Counter(),
        "lb_eligible_raw": Counter(),
        "lb_completion_mode_raw": Counter(),
    }

    with data_path.open("r", encoding="latin-1", newline="") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")

            hhid = read_fixed_width_field(line, specs, "HHID")
            pn = read_fixed_width_field(line, specs, "PN")

            try:
                hhidpn, hhidpn_padded = build_hhidpn(hhid, pn)
            except ValueError as exc:
                raise ValueError(
                    f"{data_path.name}, line {line_number}: {exc}"
                ) from exc

            if hhidpn_padded in identifiers_seen:
                duplicate_identifiers += 1
            identifiers_seen.add(hhidpn_padded)

            who_answered_raw = integer_or_blank(
                read_fixed_width_field(
                    line, specs, config["who_answered"]
                )
            )
            lb_rtype_raw = integer_or_blank(
                read_fixed_width_field(
                    line, specs, config["lb_rtype"]
                )
            )
            lb_eligible_raw = integer_or_blank(
                read_fixed_width_field(
                    line, specs, config["eligibility"]
                )
            )
            lb_completion_mode_raw = integer_or_blank(
                read_fixed_width_field(
                    line, specs, config["completion_mode"]
                )
            )

            # Requested harmonization: preserve only source codes 1 and 2.
            respondent_type = (
                who_answered_raw if who_answered_raw in (1, 2) else ""
            )

            row = {
                "hhidpn": hhidpn,
                "hhidpn_padded": hhidpn_padded,
                "hhid": hhid,
                "pn": pn,
                "year": year,
                "wave": config["wave"],
                "respondent_type": respondent_type,
                "who_answered_raw": who_answered_raw,
                "lb_rtype_raw": lb_rtype_raw,
                "lb_eligible_raw": lb_eligible_raw,
                "lb_completion_mode_raw": lb_completion_mode_raw,
                "lb_record_present": 1,
                "source_who_answered_variable": config["who_answered"],
            }
            rows.append(row)

            values_to_count = {
                "who_answered_raw": who_answered_raw,
                "respondent_type": respondent_type,
                "lb_rtype_raw": lb_rtype_raw,
                "lb_eligible_raw": lb_eligible_raw,
                "lb_completion_mode_raw": lb_completion_mode_raw,
            }
            for key, value in values_to_count.items():
                count_key = "blank" if value == "" else str(value)
                distributions[key][count_key] += 1

    audit = {
        "year": year,
        "wave": config["wave"],
        "data_file": data_path.name,
        "dictionary_file": dct_path.name,
        "rows": len(rows),
        "unique_hhidpn": len(identifiers_seen),
        "duplicate_hhidpn": duplicate_identifiers,
        "source_variables": {
            "who_answered": config["who_answered"],
            "lb_rtype": config["lb_rtype"],
            "eligibility": config["eligibility"],
            "completion_mode": config["completion_mode"],
        },
        "distributions": {
            key: dict(counter)
            for key, counter in distributions.items()
        },
    }

    return rows, audit


def build_wide_rows(
    long_rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    respondent_data: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

    for row in long_rows:
        respondent_data[row["hhidpn_padded"]][row["wave"]] = row

    headers = [
        "hhidpn",
        "hhidpn_padded",
        "hhid",
        "pn",
    ]
    headers.extend(f"respondent_type_w{wave}" for wave in WAVES)

    for wave in WAVES:
        headers.extend(
            [
                f"lb_record_present_w{wave}",
                f"who_answered_raw_w{wave}",
                f"lb_rtype_raw_w{wave}",
                f"lb_eligible_raw_w{wave}",
                f"lb_completion_mode_raw_w{wave}",
            ]
        )

    headers.extend(
        [
            "n_lb_wave_records",
            "n_respondent_type_observed",
            "n_code1_waves",
            "n_code2_waves",
            "first_lb_wave",
            "last_lb_wave",
        ]
    )

    wide_rows: list[dict[str, Any]] = []

    for hhidpn_padded in sorted(
        respondent_data,
        key=lambda value: int(value),
    ):
        wave_map = respondent_data[hhidpn_padded]
        first_record = wave_map[min(wave_map)]

        output_row: dict[str, Any] = {
            "hhidpn": first_record["hhidpn"],
            "hhidpn_padded": hhidpn_padded,
            "hhid": first_record["hhid"],
            "pn": first_record["pn"],
        }

        for wave in WAVES:
            wave_row = wave_map.get(wave)
            output_row[f"respondent_type_w{wave}"] = (
                wave_row["respondent_type"] if wave_row else ""
            )

        for wave in WAVES:
            wave_row = wave_map.get(wave)
            output_row[f"lb_record_present_w{wave}"] = 1 if wave_row else 0
            output_row[f"who_answered_raw_w{wave}"] = (
                wave_row["who_answered_raw"] if wave_row else ""
            )
            output_row[f"lb_rtype_raw_w{wave}"] = (
                wave_row["lb_rtype_raw"] if wave_row else ""
            )
            output_row[f"lb_eligible_raw_w{wave}"] = (
                wave_row["lb_eligible_raw"] if wave_row else ""
            )
            output_row[f"lb_completion_mode_raw_w{wave}"] = (
                wave_row["lb_completion_mode_raw"] if wave_row else ""
            )

        observed_values = [
            row["respondent_type"]
            for row in wave_map.values()
            if row["respondent_type"] in (1, 2)
        ]
        record_waves = sorted(wave_map)

        output_row["n_lb_wave_records"] = len(record_waves)
        output_row["n_respondent_type_observed"] = len(observed_values)
        output_row["n_code1_waves"] = sum(
            value == 1 for value in observed_values
        )
        output_row["n_code2_waves"] = sum(
            value == 2 for value in observed_values
        )
        output_row["first_lb_wave"] = record_waves[0]
        output_row["last_lb_wave"] = record_waves[-1]

        wide_rows.append(output_row)

    return headers, wide_rows


def write_dict_csv(
    path: Path,
    headers: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=headers,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_codebook(path: Path) -> None:
    rows = [
        ["column_pattern", "description"],
        [
            "hhidpn",
            "Numeric-style concatenation of HHID and PN, with leading "
            "zeros removed. This resembles common merge IDs such as 3010.",
        ],
        [
            "hhidpn_padded",
            "Nine-character HHID+PN identifier preserving leading zeros.",
        ],
        [
            "hhid",
            "Six-character household identification number, stored as text.",
        ],
        [
            "pn",
            "Three-character respondent person number, stored as text.",
        ],
        [
            "respondent_type_w8 ... respondent_type_w13",
            "Harmonized WHO ANSWERED THE QUESTIONS value. Only source "
            "codes 1 and 2 are retained; other codes and missing values "
            "are output as blank.",
        ],
        [
            "lb_record_present_wN",
            "1 if the respondent has a record in that wave's LB file; "
            "0 otherwise.",
        ],
        [
            "who_answered_raw_wN",
            "Unchanged source value from KLB051, LLB051, MLB051, "
            "NLB085, OLB077, or PLB077.",
        ],
        [
            "lb_rtype_raw_wN",
            "Unchanged HRS RESPONDENT TYPE INDICATOR from XLBRTYPE. "
            "This is distinct from WHO ANSWERED THE QUESTIONS.",
        ],
        [
            "lb_eligible_raw_wN",
            "Unchanged Leave-Behind eligibility value from XLBELIG.",
        ],
        [
            "lb_completion_mode_raw_wN",
            "Unchanged Leave-Behind completion-mode value from XLBCOMP.",
        ],
        [
            "n_lb_wave_records",
            "Number of the six source LB files containing a record for "
            "the respondent.",
        ],
        [
            "n_respondent_type_observed",
            "Number of waves in which respondent_type_wN is 1 or 2.",
        ],
        [
            "n_code1_waves",
            "Number of harmonized respondent-type values equal to 1.",
        ],
        [
            "n_code2_waves",
            "Number of harmonized respondent-type values equal to 2.",
        ],
        [
            "first_lb_wave / last_lb_wave",
            "Earliest and latest source wave containing an LB record.",
        ],
        [
            "wave mapping",
            "2006=W8; 2008=W9; 2010=W10; 2012=W11; "
            "2014=W12; 2016=W13.",
        ],
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    all_long_rows: list[dict[str, Any]] = []
    wave_audits: dict[str, Any] = {}

    for year, config in CONFIG.items():
        rows, audit = read_wave(year, config, input_dir)
        all_long_rows.extend(rows)
        wave_audits[str(config["wave"])] = audit

        print(
            f"Loaded {year} / wave {config['wave']}: "
            f"{len(rows):,} rows"
        )

    all_long_rows.sort(
        key=lambda row: (row["hhidpn"], row["wave"])
    )

    long_headers = [
        "hhidpn",
        "hhidpn_padded",
        "hhid",
        "pn",
        "year",
        "wave",
        "respondent_type",
        "who_answered_raw",
        "lb_rtype_raw",
        "lb_eligible_raw",
        "lb_completion_mode_raw",
        "lb_record_present",
        "source_who_answered_variable",
    ]

    wide_headers, wide_rows = build_wide_rows(all_long_rows)

    wide_path = output_dir / f"{args.prefix}_wide.csv"
    long_path = output_dir / f"{args.prefix}_long.csv"
    codebook_path = output_dir / (
        f"{args.prefix.replace('_w8_w13', '')}_codebook.csv"
    )
    audit_path = output_dir / (
        f"{args.prefix.replace('_w8_w13', '')}_merge_audit.json"
    )

    write_dict_csv(wide_path, wide_headers, wide_rows)
    write_dict_csv(long_path, long_headers, all_long_rows)
    write_codebook(codebook_path)

    audit = {
        "overall": {
            "unique_respondents_wide": len(wide_rows),
            "long_rows": len(all_long_rows),
            "duplicates_within_wave_total": sum(
                wave_audit["duplicate_hhidpn"]
                for wave_audit in wave_audits.values()
            ),
            "wave_mapping": {
                str(config["wave"]): year
                for year, config in CONFIG.items()
            },
            "main_output": wide_path.name,
            "supplementary_long_output": long_path.name,
        },
        "waves": wave_audits,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Wide output:     {wide_path}")
    print(f"Long output:     {long_path}")
    print(f"Codebook:        {codebook_path}")
    print(f"Audit:           {audit_path}")
    print(
        f"Unique respondents: {len(wide_rows):,}; "
        f"respondent-wave rows: {len(all_long_rows):,}"
    )


if __name__ == "__main__":
    main()
