#!/usr/bin/env python3
"""
Update the HRS participant-level loneliness cohort master dataset.

Inputs
------
1. HRS_participant_level_cohorts.csv
   Existing one-row-per-participant cohort master.

2. merged_cognition_loneliness_HRS_w6_w13_with_demographics_wealth(3).csv
   Original respondent-wave merged dataset containing latent Cog,
   psychosocial variables, demographics, wealth, and marital status.

3. HRS_LB_respondent_type_w8_w13_wide.csv
   Leave-Behind/Psychosocial raw respondent and completion variables.

4. cogfinalimp_9522wide(1).dta
   Langa-Weir contributed cognition dataset containing:
   cogtot27_impYYYY, cogfunctionYYYY, proxyYYYY, and interviewYYYY.

Time mapping
------------
Cohort A: T1=wave 8 (2006), T2=wave 10 (2010), T3=wave 11 (2012)
Cohort B: T1=wave 9 (2008), T2=wave 11 (2012), T3=wave 12 (2014)

Definitions
-----------
psychosocial_completed_Tn:
    1 if raw LBCOMP is 1, 2, or 4
    0 if raw LBCOMP is 5
    missing if raw LBCOMP is missing or unexpected

self_completed_loneliness_Tn:
    1 if raw LBCOMP is 1 or 2 AND all three loneliness items are valid
    0 otherwise

cog27_Tn:
    cogtot27_imp for self respondents. It is forced to missing when
    proxy code is 1 (spouse) or 2 (other proxy).

cognitive_classification_Tn:
    cogfunction label retained for self and proxy respondents:
    1=Normal, 2=CIND, 3=Demented.

Eligibility flags are created but intentionally left blank.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WAVE_YEAR = {
    8: 2006,
    9: 2008,
    10: 2010,
    11: 2012,
    12: 2014,
    13: 2016,
}

TIME_MAP = {
    "A": {"T1": 8, "T2": 10, "T3": 11},
    "B": {"T1": 9, "T2": 11, "T3": 12},
}

LONG_VARIABLES = [
    "age",
    "wealth",
    "marital_status",
    "lives_alone",
    "lack_companionship",
    "left_out",
    "feels_isolated",
    "Cog",
]

COGFUNCTION_LABELS = {
    1: "Normal",
    2: "CIND",
    3: "Demented",
}

EXPECTED_LBCOMP_CODES = {1, 2, 4, 5}


def normalize_hhidpn(value: Any) -> str:
    """Normalize HRS participant IDs for reliable joins."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        stripped = text.lstrip("0")
        return stripped or "0"
    try:
        number = int(float(text))
        return str(number)
    except (TypeError, ValueError):
        return text


def valid_number(value: Any) -> bool:
    """True only for a finite numeric value."""
    if value is None or pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def nullable_integer(value: Any) -> Any:
    """Return an integer when value is integer-like; otherwise missing."""
    if not valid_number(value):
        return pd.NA
    number = float(value)
    if number.is_integer():
        return int(number)
    return pd.NA


def numeric_or_missing(value: Any) -> Any:
    """Return finite float or missing."""
    return float(value) if valid_number(value) else np.nan


def clean_text_or_missing(value: Any) -> Any:
    """Return nonblank text or missing."""
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return pd.NA
    return text


def build_long_lookup(long_df: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    """Create participant-wave lookup from the original long merged data."""
    long_df = long_df.copy()
    long_df["hhidpn"] = long_df["hhidpn"].map(normalize_hhidpn)
    long_df["wave"] = pd.to_numeric(long_df["wave"], errors="coerce")

    duplicate_mask = long_df.duplicated(["hhidpn", "wave"], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"Original merged data contains {duplicate_count} rows involved "
            "in duplicate hhidpn-wave combinations."
        )

    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in long_df.itertuples(index=False):
        wave = getattr(row, "wave")
        hhidpn = getattr(row, "hhidpn")
        if not hhidpn or pd.isna(wave):
            continue
        lookup[(hhidpn, int(wave))] = row._asdict()
    return lookup


def prepare_lb_lookup(lb_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Create participant-level lookup for LB raw variables."""
    lb_df = lb_df.copy()
    lb_df["hhidpn"] = lb_df["hhidpn"].map(normalize_hhidpn)

    if lb_df["hhidpn"].duplicated().any():
        duplicates = int(lb_df["hhidpn"].duplicated(keep=False).sum())
        raise ValueError(
            f"LB wide file contains {duplicates} rows involved in duplicate hhidpn values."
        )

    return {
        row["hhidpn"]: row
        for row in lb_df.to_dict(orient="records")
        if row.get("hhidpn")
    }


def load_langa_weir_lookup(dta_path: Path) -> dict[str, dict[str, Any]]:
    """Read only the Langa-Weir columns needed for waves 8-12."""
    years = sorted({WAVE_YEAR[w] for w in (8, 9, 10, 11, 12)})
    columns = ["hhid", "pn"]
    for year in years:
        columns.extend(
            [
                f"interview{year}",
                f"proxy{year}",
                f"cogtot27_imp{year}",
                f"cogfunction{year}",
            ]
        )

    cognition = pd.read_stata(
        dta_path,
        columns=columns,
        convert_categoricals=False,
    )

    cognition["hhidpn"] = (
        cognition["hhid"].astype(str).str.strip()
        + cognition["pn"].astype(str).str.strip()
    ).map(normalize_hhidpn)

    if cognition["hhidpn"].duplicated().any():
        duplicates = int(cognition["hhidpn"].duplicated(keep=False).sum())
        raise ValueError(
            f"Langa-Weir file contains {duplicates} rows involved in duplicate hhidpn values."
        )

    return {
        row["hhidpn"]: row
        for row in cognition.to_dict(orient="records")
        if row.get("hhidpn")
    }


def psychosocial_completed_from_code(code: Any) -> Any:
    """
    Derive binary psychosocial completion:
      1 for LBCOMP 1, 2, or 4
      0 for LBCOMP 5
      missing otherwise.
    """
    code_int = nullable_integer(code)
    if pd.isna(code_int):
        return pd.NA
    if code_int in {1, 2, 4}:
        return 1
    if code_int == 5:
        return 0
    return pd.NA


def self_completed_loneliness(
    completion_code: Any,
    lack_companionship: Any,
    left_out: Any,
    feels_isolated: Any,
) -> int:
    """
    1 when LBCOMP is 1 or 2 and all three loneliness items are valid;
    0 otherwise.
    """
    code_int = nullable_integer(completion_code)
    all_items_valid = all(
        valid_number(value)
        for value in (lack_companionship, left_out, feels_isolated)
    )
    return int(code_int in {1, 2} and all_items_valid)


def classify_cogfunction(code: Any) -> Any:
    """Map Langa-Weir cogfunction code to readable label."""
    code_int = nullable_integer(code)
    if pd.isna(code_int):
        return pd.NA
    return COGFUNCTION_LABELS.get(code_int, pd.NA)


def add_time_specific_fields(
    master: pd.DataFrame,
    long_lookup: dict[tuple[str, int], dict[str, Any]],
    lb_lookup: dict[str, dict[str, Any]],
    lw_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Add all requested T1/T2/T3 variables to the master."""
    records: list[dict[str, Any]] = []

    for master_row in master.to_dict(orient="records"):
        output = dict(master_row)
        hhidpn = normalize_hhidpn(master_row["hhidpn"])
        cohort = str(master_row["cohort"]).strip().upper()

        if cohort not in TIME_MAP:
            raise ValueError(f"Unexpected cohort {cohort!r} for hhidpn {hhidpn}.")

        output["hhidpn"] = hhidpn
        lb_row = lb_lookup.get(hhidpn, {})
        lw_row = lw_lookup.get(hhidpn, {})

        for timepoint, wave in TIME_MAP[cohort].items():
            long_row = long_lookup.get((hhidpn, wave), {})
            year = WAVE_YEAR[wave]

            # Original merged-data fields.
            for variable in LONG_VARIABLES:
                value = long_row.get(variable, pd.NA)
                column = f"{variable}_{timepoint}"
                if variable == "marital_status":
                    output[column] = clean_text_or_missing(value)
                else:
                    output[column] = numeric_or_missing(value)

            # Raw LB "who answered?" and completion mode.
            output[f"who_answered_raw_{timepoint}"] = nullable_integer(
                lb_row.get(f"who_answered_raw_w{wave}", pd.NA)
            )

            # Questionnaire completion and completion-derived fields are
            # requested only at T1 and T2.
            if timepoint in {"T1", "T2"}:
                completion = nullable_integer(
                    lb_row.get(f"lb_completion_mode_raw_w{wave}", pd.NA)
                )
                output[f"questionnaire_completion_{timepoint}"] = completion
                output[f"psychosocial_completed_{timepoint}"] = (
                    psychosocial_completed_from_code(completion)
                )
                output[f"self_completed_loneliness_{timepoint}"] = (
                    self_completed_loneliness(
                        completion,
                        output[f"lack_companionship_{timepoint}"],
                        output[f"left_out_{timepoint}"],
                        output[f"feels_isolated_{timepoint}"],
                    )
                )

            # Existing latent Cog availability.
            output[f"has_Cog_{timepoint}"] = int(
                valid_number(output[f"Cog_{timepoint}"])
            )

            # Langa-Weir fields.
            interview = nullable_integer(lw_row.get(f"interview{year}", pd.NA))
            proxy = nullable_integer(lw_row.get(f"proxy{year}", pd.NA))
            raw_cog27 = lw_row.get(f"cogtot27_imp{year}", pd.NA)
            cogfunction = lw_row.get(f"cogfunction{year}", pd.NA)

            if interview != 1:
                raw_cog27 = pd.NA
                cogfunction = pd.NA
                proxy = pd.NA

            # Direct 27-point score is blank for proxy respondents.
            if proxy in {1, 2}:
                cog27 = np.nan
            else:
                cog27 = numeric_or_missing(raw_cog27)

            output[f"cog27_{timepoint}"] = cog27
            output[f"cogfunction_{timepoint}"] = nullable_integer(cogfunction)
            output[f"cognitive_classification_{timepoint}"] = (
                classify_cogfunction(cogfunction)
            )
            output[f"has_cog27_{timepoint}"] = int(valid_number(cog27))

        # Difference score only when both direct 27-point scores are valid.
        if valid_number(output["cog27_T2"]) and valid_number(output["cog27_T3"]):
            output["cog27_T3_minus_T2"] = (
                float(output["cog27_T3"]) - float(output["cog27_T2"])
            )
        else:
            output["cog27_T3_minus_T2"] = np.nan

        # Intentionally unset pending final analysis-sample definitions.
        output["eligible_basic_analysis"] = pd.NA
        output["eligible_living_alone_model"] = pd.NA
        output["eligible_fully_adjusted_model"] = pd.NA

        records.append(output)

    updated = pd.DataFrame.from_records(records)

    # Use nullable integer dtype for code/flag columns while preserving blank cells.
    nullable_int_columns = [
        *[f"who_answered_raw_{t}" for t in ("T1", "T2", "T3")],
        *[f"questionnaire_completion_{t}" for t in ("T1", "T2")],
        *[f"psychosocial_completed_{t}" for t in ("T1", "T2")],
        *[f"self_completed_loneliness_{t}" for t in ("T1", "T2")],
        *[f"has_Cog_{t}" for t in ("T1", "T2", "T3")],
        *[f"has_cog27_{t}" for t in ("T1", "T2", "T3")],
        *[f"cogfunction_{t}" for t in ("T1", "T2", "T3")],
        "eligible_basic_analysis",
        "eligible_living_alone_model",
        "eligible_fully_adjusted_model",
    ]

    for column in nullable_int_columns:
        updated[column] = pd.array(updated[column], dtype="Int64")

    return updated


def order_columns(updated: pd.DataFrame, original_columns: list[str]) -> pd.DataFrame:
    """Keep original columns first and append new columns in logical sections."""
    new_columns = [
        "age_T1", "age_T2", "age_T3",
        "wealth_T1", "wealth_T2", "wealth_T3",
        "marital_status_T1", "marital_status_T2", "marital_status_T3",
        "lives_alone_T1", "lives_alone_T2", "lives_alone_T3",
        "who_answered_raw_T1", "who_answered_raw_T2", "who_answered_raw_T3",
        "questionnaire_completion_T1", "questionnaire_completion_T2",
        "psychosocial_completed_T1", "psychosocial_completed_T2",
        "lack_companionship_T1", "lack_companionship_T2", "lack_companionship_T3",
        "left_out_T1", "left_out_T2", "left_out_T3",
        "feels_isolated_T1", "feels_isolated_T2", "feels_isolated_T3",
        "self_completed_loneliness_T1", "self_completed_loneliness_T2",
        "Cog_T1", "Cog_T2", "Cog_T3",
        "has_Cog_T1", "has_Cog_T2", "has_Cog_T3",
        "cog27_T1", "cog27_T2", "cog27_T3",
        "cog27_T3_minus_T2",
        "cogfunction_T1", "cogfunction_T2", "cogfunction_T3",
        "cognitive_classification_T1",
        "cognitive_classification_T2",
        "cognitive_classification_T3",
        "has_cog27_T1", "has_cog27_T2", "has_cog27_T3",
        "eligible_basic_analysis",
        "eligible_living_alone_model",
        "eligible_fully_adjusted_model",
    ]

    missing = [column for column in new_columns if column not in updated.columns]
    if missing:
        raise ValueError(f"Expected output columns were not created: {missing}")

    return updated[original_columns + new_columns]


def build_availability_analysis(updated: pd.DataFrame) -> pd.DataFrame:
    """
    Create flag summaries and Cog/cog27 coexistence distributions
    by cohort and overall.
    """
    rows: list[dict[str, Any]] = []
    cohort_groups = [
        ("A", updated[updated["cohort"] == "A"]),
        ("B", updated[updated["cohort"] == "B"]),
        ("Total", updated),
    ]

    for cohort_label, group in cohort_groups:
        denominator = len(group)

        for timepoint in ("T1", "T2", "T3"):
            for flag in (f"has_Cog_{timepoint}", f"has_cog27_{timepoint}"):
                count = int(group[flag].fillna(0).sum())
                rows.append(
                    {
                        "section": "availability_flag",
                        "cohort": cohort_label,
                        "timepoint": timepoint,
                        "measure": flag,
                        "category": "present",
                        "count": count,
                        "denominator": denominator,
                        "percentage": round(100 * count / denominator, 2)
                        if denominator
                        else np.nan,
                    }
                )

            cog_present = group[f"has_Cog_{timepoint}"].fillna(0).astype(int) == 1
            cog27_present = (
                group[f"has_cog27_{timepoint}"].fillna(0).astype(int) == 1
            )

            categories = {
                "both_present": cog_present & cog27_present,
                "Cog_only": cog_present & ~cog27_present,
                "cog27_only": ~cog_present & cog27_present,
                "neither_present": ~cog_present & ~cog27_present,
            }

            for category, mask in categories.items():
                count = int(mask.sum())
                rows.append(
                    {
                        "section": "coexistence",
                        "cohort": cohort_label,
                        "timepoint": timepoint,
                        "measure": "Cog_vs_cog27",
                        "category": category,
                        "count": count,
                        "denominator": denominator,
                        "percentage": round(100 * count / denominator, 2)
                        if denominator
                        else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def build_validation_report(
    master: pd.DataFrame,
    updated: pd.DataFrame,
    lb_df: pd.DataFrame,
    lw_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Create compact validation and coverage report."""
    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: Any, expected: Any = "") -> None:
        checks.append({"check": name, "value": value, "expected": expected})

    add_check("input_participants", len(master))
    add_check("output_participants", len(updated), len(master))
    add_check(
        "output_unique_hhidpn",
        updated["hhidpn"].nunique(),
        len(updated),
    )
    add_check(
        "cohort_A_participants",
        int((updated["cohort"] == "A").sum()),
        int((master["cohort"] == "A").sum()),
    )
    add_check(
        "cohort_B_participants",
        int((updated["cohort"] == "B").sum()),
        int((master["cohort"] == "B").sum()),
    )

    lb_ids = set(lb_df["hhidpn"].map(normalize_hhidpn))
    lw_ids = set(lw_lookup)
    master_ids = set(updated["hhidpn"])

    add_check("matched_to_LB_file", len(master_ids & lb_ids), len(master_ids))
    add_check("matched_to_Langa_Weir_file", len(master_ids & lw_ids), len(master_ids))

    # Existing T2/T3 Cog fields should match the newly extracted long-data values.
    for timepoint, old_column in (("T2", "cog T2"), ("T3", "cog T3")):
        old_values = pd.to_numeric(updated[old_column], errors="coerce")
        new_values = pd.to_numeric(updated[f"Cog_{timepoint}"], errors="coerce")
        comparable = old_values.notna() & new_values.notna()
        mismatch = (
            (old_values[comparable] - new_values[comparable]).abs() > 1e-8
        ).sum()
        add_check(
            f"{old_column}_vs_Cog_{timepoint}_numeric_mismatches",
            int(mismatch),
            0,
        )

    # Check that all retained direct cog27 values are not proxy codes 1 or 2.
    # This is guaranteed by construction; report missing score + retained class counts.
    for timepoint in ("T1", "T2", "T3"):
        proxy_class_without_score = (
            (updated[f"has_cog27_{timepoint}"] == 0)
            & updated[f"cognitive_classification_{timepoint}"].notna()
        ).sum()
        add_check(
            f"classification_present_but_cog27_missing_{timepoint}",
            int(proxy_class_without_score),
        )

    # Unexpected completion codes in relevant T1/T2 mappings.
    unexpected_count = 0
    lb_lookup = prepare_lb_lookup(lb_df)
    for row in updated[["cohort", "hhidpn"]].to_dict(orient="records"):
        cohort = row["cohort"]
        hhidpn = row["hhidpn"]
        lb_row = lb_lookup.get(hhidpn, {})
        for timepoint in ("T1", "T2"):
            wave = TIME_MAP[cohort][timepoint]
            code = nullable_integer(
                lb_row.get(f"lb_completion_mode_raw_w{wave}", pd.NA)
            )
            if not pd.isna(code) and code not in EXPECTED_LBCOMP_CODES:
                unexpected_count += 1
    add_check("unexpected_LBCOMP_codes_at_T1_or_T2", unexpected_count, 0)

    return pd.DataFrame(checks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the HRS participant-level cohort master."
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("HRS_participant_level_cohorts.csv"),
    )
    parser.add_argument(
        "--merged-long",
        type=Path,
        default=Path(
            "merged_cognition_loneliness_HRS_w6_w13_with_demographics_wealth(3).csv"
        ),
    )
    parser.add_argument(
        "--lb-wide",
        type=Path,
        default=Path("HRS_LB_respondent_type_w8_w13_wide.csv"),
    )
    parser.add_argument(
        "--langa-weir",
        type=Path,
        default=Path("cogfinalimp_9522wide(1).dta"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HRS_participant_level_cohorts_updated.csv"),
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("HRS_cognition_availability_analysis.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("HRS_participant_level_update_validation.csv"),
    )
    args = parser.parse_args()

    master = pd.read_csv(args.master, dtype={"hhidpn": "string"})
    original_columns = list(master.columns)
    master["hhidpn"] = master["hhidpn"].map(normalize_hhidpn)

    if master["hhidpn"].duplicated().any():
        duplicate_count = int(master["hhidpn"].duplicated(keep=False).sum())
        raise ValueError(
            f"Master file contains {duplicate_count} rows involved in duplicate hhidpn values."
        )

    long_df = pd.read_csv(
        args.merged_long,
        dtype={"hhidpn": "string"},
        low_memory=False,
    )
    required_long = {"hhidpn", "wave", *LONG_VARIABLES}
    missing_long = required_long.difference(long_df.columns)
    if missing_long:
        raise ValueError(
            "Merged long file is missing required columns: "
            + ", ".join(sorted(missing_long))
        )

    lb_df = pd.read_csv(
        args.lb_wide,
        dtype={"hhidpn": "string"},
        low_memory=False,
    )

    long_lookup = build_long_lookup(long_df)
    lb_lookup = prepare_lb_lookup(lb_df)
    lw_lookup = load_langa_weir_lookup(args.langa_weir)

    updated = add_time_specific_fields(
        master=master,
        long_lookup=long_lookup,
        lb_lookup=lb_lookup,
        lw_lookup=lw_lookup,
    )
    updated = order_columns(updated, original_columns)

    analysis = build_availability_analysis(updated)
    validation = build_validation_report(
        master=master,
        updated=updated,
        lb_df=lb_df,
        lw_lookup=lw_lookup,
    )

    updated.to_csv(args.output, index=False, na_rep="")
    analysis.to_csv(args.analysis_output, index=False, na_rep="")
    validation.to_csv(args.validation_output, index=False, na_rep="")

    print(f"Created updated master: {args.output}")
    print(f"Created cognition availability analysis: {args.analysis_output}")
    print(f"Created validation report: {args.validation_output}")
    print(f"Participants: {len(updated):,}")
    print(f"Columns: {len(updated.columns)}")

    print("\nAvailability summary:")
    availability = analysis[
        (analysis["section"] == "availability_flag")
        & (analysis["cohort"] == "Total")
    ]
    for row in availability.itertuples(index=False):
        print(
            f"  {row.measure}: {row.count:,}/{row.denominator:,} "
            f"({row.percentage:.2f}%)"
        )

    print("\nCog versus cog27 coexistence, total sample:")
    coexistence = analysis[
        (analysis["section"] == "coexistence")
        & (analysis["cohort"] == "Total")
    ]
    for timepoint in ("T1", "T2", "T3"):
        print(f"  {timepoint}:")
        block = coexistence[coexistence["timepoint"] == timepoint]
        for row in block.itertuples(index=False):
            print(
                f"    {row.category}: {row.count:,} "
                f"({row.percentage:.2f}%)"
            )


if __name__ == "__main__":
    main()
