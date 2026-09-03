#!/usr/bin/env python3
"""Run the final-phase HRS loneliness, contact, and cognition analyses.
The input file is the input file is: 
HRS_participant_level_cohorts_current.csv

python run_final_phase_analysis.py \
    --input "HRS_participant_level_cohorts_current.csv" \
    --output-dir "final_phase_results"

This script reproduces the pooled primary Models 1-3 for friends, other
relatives, and children; applies the prespecified Benjamini-Hochberg correction
to the three Model 3 omnibus interaction tests; and runs the two sensitivity
analyses explicitly described in Peer Review.docx:

1. Model 3 with loneliness change categorized as decreasing, stable, or
   increasing (stable is the reference; 6-df interaction test).
2. Continuous-change Model 3 separately in Cohorts A and B, omitting the cohort
   indicator (3-df interaction test).

Following the professor's final instructions, the script also estimates a
secondary explanatory analysis of the overall association between continuous
loneliness change and subsequent cognition without contact-transition terms:

3. Pooled Models 1-3 predicting T3 cognition from loneliness change while
   adjusting for T2 cognition and T1 loneliness, then adding the same planned
   covariate sequence as the primary analysis.
4. The fully adjusted overall-association model separately in Cohorts A and B,
   omitting the cohort indicator.
5. A formal pooled comparison adding a loneliness-change-by-cohort interaction
   to fully adjusted Model 3.

All models are unweighted OLS with HC3 covariance. The frozen participant-level
input is read but never modified. No random, imputation, or resampling step is
used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


EXPECTED_INPUT_SHA256 = "d6f53a78d2f8140b6589a036c5b2abb11499b678eeef38a48cf3c30017685828"
EXPECTED_ROWS = 8041
EXPECTED_COLUMNS = 79
EXPECTED_COHORT_COUNTS = {"A": 4272, "B": 3769}
EXPECTED_OVERALL_N = 5243
EXPECTED_OVERALL_COHORT_COUNTS = {"A": 2722, "B": 2521}

DOMAIN_SPECS = {
    "friends": {
        "label": "Friends",
        "transition_col": "friends_contact_transition",
        "contact_t1_col": "weekly_contact_friends_T1",
        "contact_t2_col": "weekly_contact_friends_T2",
        "eligibility_col": "eligible_friends_contact_fully_adjusted_model",
        "expected_n": 5068,
        "expected_cohorts": {"A": 2640, "B": 2428},
        "expected_contact": {
            "maintained_frequent": 2785,
            "increased_contact": 622,
            "decreased_contact": 691,
            "maintained_infrequent": 970,
        },
        "expected_model3_f": 1.6056348558,
        "expected_model3_p": 0.185849265837,
    },
    "other_relatives": {
        "label": "Other relatives",
        "transition_col": "other_relatives_contact_transition",
        "contact_t1_col": "weekly_contact_other_relatives_T1",
        "contact_t2_col": "weekly_contact_other_relatives_T2",
        "eligibility_col": "eligible_other_relatives_contact_fully_adjusted_model",
        "expected_n": 5080,
        "expected_cohorts": {"A": 2638, "B": 2442},
        "expected_contact": {
            "maintained_frequent": 1873,
            "increased_contact": 794,
            "decreased_contact": 743,
            "maintained_infrequent": 1670,
        },
        "expected_model3_f": 0.413923872076,
        "expected_model3_p": 0.743011251356,
    },
    "children": {
        "label": "Children",
        "transition_col": "children_contact_transition",
        "contact_t1_col": "weekly_contact_children_T1",
        "contact_t2_col": "weekly_contact_children_T2",
        "eligibility_col": "eligible_children_contact_fully_adjusted_model",
        "expected_n": 4717,
        "expected_cohorts": {"A": 2465, "B": 2252},
        "expected_contact": {
            "maintained_frequent": 3536,
            "increased_contact": 342,
            "decreased_contact": 399,
            "maintained_infrequent": 440,
        },
        "expected_model3_f": 2.10942005729,
        "expected_model3_p": 0.0968394132133,
    },
}

CONTACT_ORDER = [
    "maintained_frequent",
    "increased_contact",
    "decreased_contact",
    "maintained_infrequent",
]
CONTACT_LABELS = {
    "maintained_frequent": "Maintained frequent",
    "increased_contact": "Increased contact",
    "decreased_contact": "Decreased contact",
    "maintained_infrequent": "Maintained infrequent",
}
CONTACT_COLORS = {
    "maintained_frequent": "#0072B2",
    "increased_contact": "#009E73",
    "decreased_contact": "#D55E00",
    "maintained_infrequent": "#CC79A7",
}
TRANSITION_PAIRS = {
    "maintained_frequent": (1, 1),
    "increased_contact": (0, 1),
    "decreased_contact": (1, 0),
    "maintained_infrequent": (0, 0),
}

RACE_LEVELS = [
    "White, non-Hispanic",
    "Black/African American, non-Hispanic",
    "Hispanic",
    "Other, non-Hispanic",
]

TERM_LABELS = {
    "Intercept": "Intercept",
    "cog_T2": "Cognition at T2",
    "ucla3_T1": "Loneliness at T1",
    "delta_loneliness": "Change in loneliness",
    "change_decreasing": "Decreasing loneliness vs stable",
    "change_increasing": "Increasing loneliness vs stable",
    "contact_increased": "Increased contact vs maintained frequent",
    "contact_decreased": "Decreased contact vs maintained frequent",
    "contact_maintained_infrequent": "Maintained infrequent vs maintained frequent",
    "delta_x_increased": "Loneliness change x increased contact",
    "delta_x_decreased": "Loneliness change x decreased contact",
    "delta_x_maintained_infrequent": "Loneliness change x maintained infrequent",
    "decreasing_x_increased": "Decreasing loneliness x increased contact",
    "decreasing_x_decreased": "Decreasing loneliness x decreased contact",
    "decreasing_x_maintained_infrequent": "Decreasing loneliness x maintained infrequent",
    "increasing_x_increased": "Increasing loneliness x increased contact",
    "increasing_x_decreased": "Increasing loneliness x decreased contact",
    "increasing_x_maintained_infrequent": "Increasing loneliness x maintained infrequent",
    "age_T2_centered": "Age at T2 (per year; centered at 75)",
    "male": "Male vs Female",
    "education": "Education (years)",
    "race_black": "Black/African American non-Hispanic vs White non-Hispanic",
    "race_hispanic": "Hispanic vs White non-Hispanic",
    "race_other": "Other non-Hispanic vs White non-Hispanic",
    "wealth_ihs_T2": "IHS wealth at T2 (wealth/$100,000)",
    "marital_separated_divorced": "Separated/divorced vs married/partnered",
    "marital_widowed": "Widowed vs married/partnered",
    "marital_never_married": "Never married vs married/partnered",
    "cohort_B": "Cohort B vs Cohort A",
    "delta_x_cohort_B": "Loneliness change x Cohort B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    path.write_text(json.dumps(value, indent=2, default=convert) + "\n", encoding="utf-8")


def marital_recode(series: pd.Series) -> pd.Series:
    mapping = {
        "Married": "Married/partnered",
        "Married, spouse absent": "Married/partnered",
        "Partnered": "Married/partnered",
        "Separated": "Separated/divorced",
        "Divorced": "Separated/divorced",
        "Separated/Divorced": "Separated/divorced",
        "Widowed": "Widowed",
        "Never married": "Never married",
    }
    result = series.map(mapping)
    bad = series.notna() & result.isna()
    if bad.any():
        raise ValueError(f"Unmapped marital-status labels: {sorted(series.loc[bad].unique())}")
    return result


def add_validation(checks: list[dict], check: str, passed: bool, observed, expected) -> None:
    checks.append(
        {
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "observed": str(observed),
            "expected": str(expected),
        }
    )
    if not passed:
        raise ValueError(f"{check}: observed {observed}; expected {expected}")


def prepare_domain_sample(
    master: pd.DataFrame,
    domain: str,
    spec: dict,
    checks: list[dict],
) -> pd.DataFrame:
    t1_items = ["lack_companionship_T1", "left_out_T1", "feels_isolated_T1"]
    t2_items = ["lack_companionship_T2", "left_out_T2", "feels_isolated_T2"]

    working = master.copy()
    working["ucla3_T1"] = working[t1_items].sum(axis=1, min_count=3) / 3.0
    working["ucla3_T2"] = working[t2_items].sum(axis=1, min_count=3) / 3.0
    working["delta_loneliness"] = working["ucla3_T2"] - working["ucla3_T1"]

    reconstructed = (
        working["self_completed_loneliness_T1"].eq(1)
        & working["self_completed_loneliness_T2"].eq(1)
        & working["cog T2"].notna()
        & working["cog T3"].notna()
        & working[spec["contact_t1_col"]].isin([0, 1])
        & working[spec["contact_t2_col"]].isin([0, 1])
        & working[
            ["age_T2", "sex", "education", "race_ethnicity", "wealth_T2", "marital_status_T2"]
        ].notna().all(axis=1)
    )
    stored = working[spec["eligibility_col"]].eq(1)
    add_validation(
        checks,
        f"{domain}: stored eligibility equals reconstructed rule",
        stored.equals(reconstructed),
        int((stored != reconstructed).sum()),
        0,
    )

    sample = working.loc[reconstructed].copy()
    sample["marital_group"] = marital_recode(sample["marital_status_T2"])
    sample["age_T2_centered"] = sample["age_T2"] - 75.0
    sample["wealth_ihs_T2"] = np.arcsinh(sample["wealth_T2"] / 100000.0)
    sample["change_category"] = np.select(
        [sample["delta_loneliness"].lt(0), sample["delta_loneliness"].gt(0)],
        ["decreasing", "increasing"],
        default="stable",
    )

    add_validation(checks, f"{domain}: analytical N", len(sample) == spec["expected_n"], len(sample), spec["expected_n"])
    observed_cohorts = sample["cohort"].value_counts().to_dict()
    add_validation(
        checks,
        f"{domain}: cohort counts",
        observed_cohorts == spec["expected_cohorts"],
        observed_cohorts,
        spec["expected_cohorts"],
    )
    observed_contact = sample[spec["transition_col"]].value_counts().to_dict()
    add_validation(
        checks,
        f"{domain}: transition counts",
        observed_contact == spec["expected_contact"],
        observed_contact,
        spec["expected_contact"],
    )

    max_diffs = {
        "T1": float((sample["ucla3_T1"] - sample["T1 loneliness"]).abs().max()),
        "T2": float((sample["ucla3_T2"] - sample["T2 loneliness"]).abs().max()),
        "change": float((sample["delta_loneliness"] - sample["Loneliness T2-T1"]).abs().max()),
    }
    for label, value in max_diffs.items():
        add_validation(checks, f"{domain}: reconstructed loneliness {label}", value <= 1e-6, value, "<= 1e-6")

    mismatches = 0
    for transition, (expected_t1, expected_t2) in TRANSITION_PAIRS.items():
        rows = sample[spec["transition_col"]].eq(transition)
        mismatches += int((sample.loc[rows, spec["contact_t1_col"]] != expected_t1).sum())
        mismatches += int((sample.loc[rows, spec["contact_t2_col"]] != expected_t2).sum())
    add_validation(checks, f"{domain}: transition reconstruction", mismatches == 0, mismatches, 0)

    return sample


def prepare_overall_sample(master: pd.DataFrame, checks: list[dict]) -> pd.DataFrame:
    """Create the contact-independent fully adjusted analytical sample.

    This follows the same longitudinal and complete-case rules used by the
    primary analyses, except that no contact variable is required. The frozen
    master is read only; all derived variables live in this working copy.
    """

    t1_items = ["lack_companionship_T1", "left_out_T1", "feels_isolated_T1"]
    t2_items = ["lack_companionship_T2", "left_out_T2", "feels_isolated_T2"]
    working = master.copy()
    working["ucla3_T1"] = working[t1_items].sum(axis=1, min_count=3) / 3.0
    working["ucla3_T2"] = working[t2_items].sum(axis=1, min_count=3) / 3.0
    working["delta_loneliness"] = working["ucla3_T2"] - working["ucla3_T1"]

    reconstructed = (
        working["self_completed_loneliness_T1"].eq(1)
        & working["self_completed_loneliness_T2"].eq(1)
        & working["cog T2"].notna()
        & working["cog T3"].notna()
        & working[
            ["age_T2", "sex", "education", "race_ethnicity", "wealth_T2", "marital_status_T2"]
        ].notna().all(axis=1)
    )
    stored = working["eligible_fully_adjusted_model"].eq(1)
    add_validation(
        checks,
        "overall: stored eligibility equals reconstructed rule",
        stored.equals(reconstructed),
        int((stored != reconstructed).sum()),
        0,
    )

    sample = working.loc[reconstructed].copy()
    sample["marital_group"] = marital_recode(sample["marital_status_T2"])
    sample["age_T2_centered"] = sample["age_T2"] - 75.0
    sample["wealth_ihs_T2"] = np.arcsinh(sample["wealth_T2"] / 100000.0)

    add_validation(
        checks,
        "overall: analytical N",
        len(sample) == EXPECTED_OVERALL_N,
        len(sample),
        EXPECTED_OVERALL_N,
    )
    observed_cohorts = sample["cohort"].value_counts().to_dict()
    add_validation(
        checks,
        "overall: cohort counts",
        observed_cohorts == EXPECTED_OVERALL_COHORT_COUNTS,
        observed_cohorts,
        EXPECTED_OVERALL_COHORT_COUNTS,
    )

    max_diffs = {
        "T1": float((sample["ucla3_T1"] - sample["T1 loneliness"]).abs().max()),
        "T2": float((sample["ucla3_T2"] - sample["T2 loneliness"]).abs().max()),
        "change": float((sample["delta_loneliness"] - sample["Loneliness T2-T1"]).abs().max()),
    }
    for label, value in max_diffs.items():
        add_validation(
            checks,
            f"overall: reconstructed loneliness {label}",
            value <= 1e-6,
            value,
            "<= 1e-6",
        )

    return sample


def contact_dummies(group: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        group.eq("increased_contact").astype(float),
        group.eq("decreased_contact").astype(float),
        group.eq("maintained_infrequent").astype(float),
    )


def append_demographic_columns(
    data: pd.DataFrame,
    columns: list[np.ndarray],
    names: list[str],
    include_wealth_marital: bool,
    include_cohort: bool,
) -> None:
    columns.extend(
        [
            data["age_T2_centered"].to_numpy(float),
            data["sex"].eq("Male").to_numpy(float),
            data["education"].to_numpy(float),
            data["race_ethnicity"].eq("Black/African American, non-Hispanic").to_numpy(float),
            data["race_ethnicity"].eq("Hispanic").to_numpy(float),
            data["race_ethnicity"].eq("Other, non-Hispanic").to_numpy(float),
        ]
    )
    names.extend(["age_T2_centered", "male", "education", "race_black", "race_hispanic", "race_other"])
    if include_wealth_marital:
        columns.extend(
            [
                data["wealth_ihs_T2"].to_numpy(float),
                data["marital_group"].eq("Separated/divorced").to_numpy(float),
                data["marital_group"].eq("Widowed").to_numpy(float),
                data["marital_group"].eq("Never married").to_numpy(float),
            ]
        )
        names.extend(
            [
                "wealth_ihs_T2",
                "marital_separated_divorced",
                "marital_widowed",
                "marital_never_married",
            ]
        )
        if include_cohort:
            columns.append(data["cohort"].eq("B").to_numpy(float))
            names.append("cohort_B")


def build_continuous_design(
    data: pd.DataFrame,
    transition_col: str,
    model_number: int,
    include_cohort: bool = True,
    forced_group: str | None = None,
    forced_delta: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    group = data[transition_col] if forced_group is None else pd.Series(forced_group, index=data.index)
    delta = data["delta_loneliness"].astype(float) if forced_delta is None else pd.Series(float(forced_delta), index=data.index)
    inc, dec, inf = contact_dummies(group)
    columns = [
        np.ones(len(data)),
        data["cog T2"].to_numpy(float),
        data["ucla3_T1"].to_numpy(float),
        delta.to_numpy(float),
        inc.to_numpy(float),
        dec.to_numpy(float),
        inf.to_numpy(float),
        (delta * inc).to_numpy(float),
        (delta * dec).to_numpy(float),
        (delta * inf).to_numpy(float),
    ]
    names = [
        "Intercept",
        "cog_T2",
        "ucla3_T1",
        "delta_loneliness",
        "contact_increased",
        "contact_decreased",
        "contact_maintained_infrequent",
        "delta_x_increased",
        "delta_x_decreased",
        "delta_x_maintained_infrequent",
    ]
    if model_number >= 2:
        append_demographic_columns(data, columns, names, model_number >= 3, include_cohort)
    return np.column_stack(columns).astype(float), names


def build_overall_association_design(
    data: pd.DataFrame,
    model_number: int,
    include_cohort: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build the contact-independent Models 1-3 design matrix."""

    columns = [
        np.ones(len(data)),
        data["cog T2"].to_numpy(float),
        data["ucla3_T1"].to_numpy(float),
        data["delta_loneliness"].to_numpy(float),
    ]
    names = ["Intercept", "cog_T2", "ucla3_T1", "delta_loneliness"]
    if model_number >= 2:
        append_demographic_columns(
            data,
            columns,
            names,
            include_wealth_marital=model_number >= 3,
            include_cohort=include_cohort,
        )
    return np.column_stack(columns).astype(float), names


def build_overall_cohort_comparison_design(
    data: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Build fully adjusted Model 3 with loneliness-change-by-cohort interaction.

    Cohort A is the reference. The delta_loneliness coefficient is therefore
    the adjusted Cohort A slope, and delta_x_cohort_B is the Cohort B-minus-A
    slope difference.
    """

    X, names = build_overall_association_design(
        data,
        model_number=3,
        include_cohort=True,
    )
    cohort_b = data["cohort"].eq("B").to_numpy(float)
    interaction = data["delta_loneliness"].to_numpy(float) * cohort_b
    return np.column_stack([X, interaction]).astype(float), names + ["delta_x_cohort_B"]


def build_categorical_change_design(
    data: pd.DataFrame,
    transition_col: str,
    forced_group: str | None = None,
    forced_change_category: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    group = data[transition_col] if forced_group is None else pd.Series(forced_group, index=data.index)
    inc, dec, inf = contact_dummies(group)
    change_category = (
        data["change_category"]
        if forced_change_category is None
        else pd.Series(forced_change_category, index=data.index)
    )
    change_decreasing = change_category.eq("decreasing").astype(float)
    change_increasing = change_category.eq("increasing").astype(float)
    columns = [
        np.ones(len(data)),
        data["cog T2"].to_numpy(float),
        data["ucla3_T1"].to_numpy(float),
        change_decreasing.to_numpy(float),
        change_increasing.to_numpy(float),
        inc.to_numpy(float),
        dec.to_numpy(float),
        inf.to_numpy(float),
        (change_decreasing * inc).to_numpy(float),
        (change_decreasing * dec).to_numpy(float),
        (change_decreasing * inf).to_numpy(float),
        (change_increasing * inc).to_numpy(float),
        (change_increasing * dec).to_numpy(float),
        (change_increasing * inf).to_numpy(float),
    ]
    names = [
        "Intercept",
        "cog_T2",
        "ucla3_T1",
        "change_decreasing",
        "change_increasing",
        "contact_increased",
        "contact_decreased",
        "contact_maintained_infrequent",
        "decreasing_x_increased",
        "decreasing_x_decreased",
        "decreasing_x_maintained_infrequent",
        "increasing_x_increased",
        "increasing_x_decreased",
        "increasing_x_maintained_infrequent",
    ]
    append_demographic_columns(data, columns, names, include_wealth_marital=True, include_cohort=True)
    return np.column_stack(columns).astype(float), names


def fit_ols_hc3(X: np.ndarray, y: np.ndarray, names: list[str], model_label: str) -> dict:
    n, p = X.shape
    beta, _, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    if rank != p:
        raise ValueError(f"{model_label}: rank {rank} does not equal {p} parameters")
    xtx_inv = np.linalg.inv(X.T @ X)
    fitted = X @ beta
    residuals = y - fitted
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    if np.any(leverage >= 1):
        raise ValueError(f"{model_label}: leverage at or above one")
    adjusted_residuals = residuals / (1.0 - leverage)
    weighted_x = X * adjusted_residuals[:, None]
    covariance = xtx_inv @ (weighted_x.T @ weighted_x) @ xtx_inv
    covariance = (covariance + covariance.T) / 2.0
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    df_resid = n - p
    t_values = beta / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df_resid)
    t_critical = stats.t.ppf(0.975, df_resid)
    ci_low = beta - t_critical * standard_errors
    ci_high = beta + t_critical * standard_errors

    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / df_resid
    mse = rss / df_resid
    cooks_distance = (residuals**2 / (p * mse)) * (leverage / (1.0 - leverage) ** 2)

    coefficient_rows = []
    for index, name in enumerate(names):
        coefficient_rows.append(
            {
                "term": name,
                "term_label": TERM_LABELS[name],
                "estimate": beta[index],
                "hc3_standard_error": standard_errors[index],
                "ci_95_low": ci_low[index],
                "ci_95_high": ci_high[index],
                "t_value": t_values[index],
                "df": df_resid,
                "p_value": p_values[index],
            }
        )

    return {
        "model_label": model_label,
        "X": X,
        "y": y,
        "names": names,
        "beta": beta,
        "covariance": covariance,
        "standard_errors": standard_errors,
        "fitted": fitted,
        "residuals": residuals,
        "leverage": leverage,
        "cooks_distance": cooks_distance,
        "df_resid": df_resid,
        "n": n,
        "p": p,
        "rank": rank,
        "singular_values": singular_values,
        "coefficient_rows": coefficient_rows,
        "fit_row": {
            "N": n,
            "parameters": p,
            "df_residual": df_resid,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "rmse": math.sqrt(mse),
            "condition_number": float(np.linalg.cond(X)),
        },
    }


def wald_f(result: dict, term_names: list[str]) -> dict:
    indices = [result["names"].index(name) for name in term_names]
    R = np.zeros((len(indices), len(result["names"])))
    for row_index, column_index in enumerate(indices):
        R[row_index, column_index] = 1.0
    estimate = R @ result["beta"]
    covariance = R @ result["covariance"] @ R.T
    chi_square = float(estimate @ np.linalg.solve(covariance, estimate))
    numerator_df = len(indices)
    f_statistic = chi_square / numerator_df
    p_value = float(stats.f.sf(f_statistic, numerator_df, result["df_resid"]))
    return {
        "f_statistic": f_statistic,
        "df_numerator": numerator_df,
        "df_denominator": result["df_resid"],
        "p_value": p_value,
        "wald_chi_square": chi_square,
        "chi_square_df": numerator_df,
        "chi_square_p_value": float(stats.chi2.sf(chi_square, numerator_df)),
    }


def linear_contrast(result: dict, weights: np.ndarray) -> dict:
    estimate = float(weights @ result["beta"])
    variance = float(weights @ result["covariance"] @ weights)
    standard_error = math.sqrt(max(variance, 0.0))
    t_value = estimate / standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_value), result["df_resid"]))
    critical = float(stats.t.ppf(0.975, result["df_resid"]))
    return {
        "estimate": estimate,
        "hc3_standard_error": standard_error,
        "ci_95_low": estimate - critical * standard_error,
        "ci_95_high": estimate + critical * standard_error,
        "t_value": t_value,
        "df": result["df_resid"],
        "p_value": p_value,
    }


def continuous_group_slopes(result: dict, domain: str, cohort: str | None = None) -> list[dict]:
    base = result["names"].index("delta_loneliness")
    interaction_by_group = {
        "maintained_frequent": None,
        "increased_contact": "delta_x_increased",
        "decreased_contact": "delta_x_decreased",
        "maintained_infrequent": "delta_x_maintained_infrequent",
    }
    rows = []
    for group in CONTACT_ORDER:
        weights = np.zeros(len(result["names"]))
        weights[base] = 1.0
        interaction = interaction_by_group[group]
        if interaction is not None:
            weights[result["names"].index(interaction)] = 1.0
        row = linear_contrast(result, weights)
        row.update(
            {
                "domain": domain,
                "domain_label": DOMAIN_SPECS[domain]["label"],
                "cohort": cohort,
                "contact_group": group,
                "contact_group_label": CONTACT_LABELS[group],
                "effect_unit": "Per one-point increase in UCLA-3 loneliness change",
            }
        )
        rows.append(row)
    return rows


def categorical_change_simple_effects(result: dict, domain: str) -> list[dict]:
    interaction_by_change_group = {
        ("decreasing", "maintained_frequent"): None,
        ("decreasing", "increased_contact"): "decreasing_x_increased",
        ("decreasing", "decreased_contact"): "decreasing_x_decreased",
        ("decreasing", "maintained_infrequent"): "decreasing_x_maintained_infrequent",
        ("increasing", "maintained_frequent"): None,
        ("increasing", "increased_contact"): "increasing_x_increased",
        ("increasing", "decreased_contact"): "increasing_x_decreased",
        ("increasing", "maintained_infrequent"): "increasing_x_maintained_infrequent",
    }
    main_term = {"decreasing": "change_decreasing", "increasing": "change_increasing"}
    rows = []
    for group in CONTACT_ORDER:
        effects = {}
        for category in ["decreasing", "increasing"]:
            weights = np.zeros(len(result["names"]))
            weights[result["names"].index(main_term[category])] = 1.0
            interaction = interaction_by_change_group[(category, group)]
            if interaction is not None:
                weights[result["names"].index(interaction)] = 1.0
            effects[category] = weights
            row = linear_contrast(result, weights)
            row.update(
                {
                    "domain": domain,
                    "domain_label": DOMAIN_SPECS[domain]["label"],
                    "contact_group": group,
                    "contact_group_label": CONTACT_LABELS[group],
                    "contrast": f"{category.capitalize()} vs stable loneliness",
                    "first_change_category": category,
                    "second_change_category": "stable",
                }
            )
            rows.append(row)
        weights = effects["increasing"] - effects["decreasing"]
        row = linear_contrast(result, weights)
        row.update(
            {
                "domain": domain,
                "domain_label": DOMAIN_SPECS[domain]["label"],
                "contact_group": group,
                "contact_group_label": CONTACT_LABELS[group],
                "contrast": "Increasing vs decreasing loneliness",
                "first_change_category": "increasing",
                "second_change_category": "decreasing",
            }
        )
        rows.append(row)
    return rows


def create_categorical_predictions(
    sample: pd.DataFrame,
    spec: dict,
    result: dict,
    domain: str,
) -> pd.DataFrame:
    rows = []
    change_order = ["decreasing", "stable", "increasing"]
    for group in CONTACT_ORDER:
        for change_category in change_order:
            X, names = build_categorical_change_design(
                sample,
                spec["transition_col"],
                forced_group=group,
                forced_change_category=change_category,
            )
            if names != result["names"]:
                raise ValueError("Categorical prediction design terms do not match fitted Model 3")
            mean_design = X.mean(axis=0)
            estimate = float(mean_design @ result["beta"])
            standard_error = math.sqrt(max(float(mean_design @ result["covariance"] @ mean_design), 0.0))
            critical = float(stats.t.ppf(0.975, result["df_resid"]))
            observed_n = int(
                (sample[spec["transition_col"]].eq(group) & sample["change_category"].eq(change_category)).sum()
            )
            rows.append(
                {
                    "domain": domain,
                    "domain_label": spec["label"],
                    "contact_group": group,
                    "contact_group_label": CONTACT_LABELS[group],
                    "change_category": change_category,
                    "observed_cell_n": observed_n,
                    "adjusted_prediction": estimate,
                    "hc3_standard_error": standard_error,
                    "ci_95_low": estimate - critical * standard_error,
                    "ci_95_high": estimate + critical * standard_error,
                }
            )
    return pd.DataFrame(rows)


def create_predictions(sample: pd.DataFrame, spec: dict, result: dict, domain: str) -> pd.DataFrame:
    rows = []
    for group in CONTACT_ORDER:
        for delta in np.linspace(-1.0, 1.0, 41):
            X, names = build_continuous_design(
                sample,
                spec["transition_col"],
                model_number=3,
                include_cohort=True,
                forced_group=group,
                forced_delta=float(delta),
            )
            if names != result["names"]:
                raise ValueError("Prediction design terms do not match fitted Model 3")
            mean_design = X.mean(axis=0)
            estimate = float(mean_design @ result["beta"])
            standard_error = math.sqrt(max(float(mean_design @ result["covariance"] @ mean_design), 0.0))
            critical = float(stats.t.ppf(0.975, result["df_resid"]))
            rows.append(
                {
                    "domain": domain,
                    "domain_label": spec["label"],
                    "contact_group": group,
                    "contact_group_label": CONTACT_LABELS[group],
                    "group_n": int(sample[spec["transition_col"]].eq(group).sum()),
                    "delta_loneliness": float(delta),
                    "adjusted_prediction": estimate,
                    "hc3_standard_error": standard_error,
                    "ci_95_low": estimate - critical * standard_error,
                    "ci_95_high": estimate + critical * standard_error,
                }
            )
    return pd.DataFrame(rows)


def vif_rows(result: dict, domain: str) -> list[dict]:
    rows = []
    X = result["X"]
    for index, name in enumerate(result["names"]):
        if name == "Intercept":
            continue
        y = X[:, index]
        other = np.delete(X, index, axis=1)
        beta, _, _, _ = np.linalg.lstsq(other, y, rcond=None)
        residual = y - other @ beta
        tss = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - float(residual @ residual) / tss
        rows.append(
            {
                "domain": domain,
                "term": name,
                "term_label": TERM_LABELS[name],
                "r_squared_against_other_predictors": r_squared,
                "vif": 1.0 / (1.0 - r_squared),
            }
        )
    return rows


def diagnostic_row(result: dict, domain: str) -> dict:
    residuals = result["residuals"]
    X = result["X"]
    squared = residuals**2
    aux_beta, _, _, _ = np.linalg.lstsq(X, squared, rcond=None)
    aux_residuals = squared - X @ aux_beta
    tss = float(((squared - squared.mean()) ** 2).sum())
    aux_r_squared = 1.0 - float(aux_residuals @ aux_residuals) / tss
    bp_statistic = result["n"] * aux_r_squared
    bp_df = result["p"] - 1
    jb = stats.jarque_bera(residuals)
    leverage_threshold = 2.0 * result["p"] / result["n"]
    cooks_threshold = 4.0 / result["n"]
    return {
        "domain": domain,
        "domain_label": DOMAIN_SPECS[domain]["label"],
        "N": result["n"],
        "parameters": result["p"],
        "residual_mean": float(residuals.mean()),
        "residual_sd": float(residuals.std(ddof=1)),
        "residual_skewness": float(stats.skew(residuals, bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(residuals, fisher=True, bias=False)),
        "jarque_bera_statistic": float(jb.statistic),
        "jarque_bera_p_value": float(jb.pvalue),
        "breusch_pagan_lm_statistic": bp_statistic,
        "breusch_pagan_df": bp_df,
        "breusch_pagan_p_value": float(stats.chi2.sf(bp_statistic, bp_df)),
        "maximum_leverage": float(result["leverage"].max()),
        "leverage_threshold_2p_over_n": leverage_threshold,
        "count_leverage_above_2p_over_n": int((result["leverage"] > leverage_threshold).sum()),
        "maximum_cooks_distance": float(result["cooks_distance"].max()),
        "cooks_threshold_4_over_n": cooks_threshold,
        "count_cooks_distance_above_4_over_n": int((result["cooks_distance"] > cooks_threshold).sum()),
        "condition_number": float(np.linalg.cond(X)),
    }


def bh_adjust(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    m = len(values)
    adjusted_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    ranks = np.empty(m, dtype=int)
    adjusted[order] = adjusted_ranked
    ranks[order] = np.arange(1, m + 1)
    return adjusted, ranks


def plot_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    global_low = float(predictions["ci_95_low"].min())
    global_high = float(predictions["ci_95_high"].max())
    margin = 0.05 * (global_high - global_low)
    y_limits = (global_low - margin, global_high + margin)

    def draw_panel(ax: plt.Axes, domain: str, show_legend: bool, include_group_n: bool) -> None:
        domain_data = predictions.loc[predictions["domain"].eq(domain)]
        for group in CONTACT_ORDER:
            part = domain_data.loc[domain_data["contact_group"].eq(group)].sort_values("delta_loneliness")
            label = CONTACT_LABELS[group]
            if include_group_n:
                label = f"{label} (n={int(part['group_n'].iloc[0]):,})"
            ax.plot(
                part["delta_loneliness"],
                part["adjusted_prediction"],
                color=CONTACT_COLORS[group],
                linewidth=2.2,
                label=label,
            )
            ax.fill_between(
                part["delta_loneliness"].to_numpy(float),
                part["ci_95_low"].to_numpy(float),
                part["ci_95_high"].to_numpy(float),
                color=CONTACT_COLORS[group],
                alpha=0.14,
                linewidth=0,
            )
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2)
        ax.set_title(DOMAIN_SPECS[domain]["label"], fontsize=13, weight="bold")
        ax.set_xlim(-1, 1)
        ax.set_ylim(*y_limits)
        ax.grid(axis="both", color="#E6E6E6", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if show_legend:
            ax.legend(frameon=True, fontsize=8.5, loc="best")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharex=True, sharey=True)
    for index, domain in enumerate(DOMAIN_SPECS):
        draw_panel(axes[index], domain, show_legend=False, include_group_n=False)
    axes[0].set_ylabel("Adjusted predicted cognition at T3")
    fig.supxlabel("Change in UCLA-3 loneliness (T2 - T1)", y=0.07)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Contact transition and subsequent cognition", fontsize=16, weight="bold", y=0.99)
    fig.tight_layout(rect=[0.02, 0.11, 1.0, 0.95])
    fig.savefig(output_dir / "three_domain_interaction_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "three_domain_interaction_figure.pdf", bbox_inches="tight")
    plt.close(fig)

    for domain in DOMAIN_SPECS:
        fig, ax = plt.subplots(figsize=(8.7, 6.1))
        draw_panel(ax, domain, show_legend=True, include_group_n=True)
        ax.set_xlabel("Change in UCLA-3 loneliness (T2 - T1)")
        ax.set_ylabel("Adjusted predicted cognition at T3")
        ax.set_title(
            f"{DOMAIN_SPECS[domain]['label']} contact transition and subsequent cognition",
            fontsize=15,
            weight="bold",
            pad=12,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{domain}_interaction_figure_methods_aligned.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{domain}_interaction_figure_methods_aligned.pdf", bbox_inches="tight")
        plt.close(fig)


def p_display(value: float) -> str:
    return "< .001" if value < 0.001 else f"= {value:.3f}".replace("0.", ".")


def write_manuscript_text(
    output_dir: Path,
    primary_bh: pd.DataFrame,
    slopes: pd.DataFrame,
    categorical: pd.DataFrame,
    categorical_simple_effects: pd.DataFrame,
    cohort_sensitivity: pd.DataFrame,
    overall_effects: pd.DataFrame,
    overall_cohort_comparison: pd.DataFrame,
    transition_summary: pd.DataFrame,
    overall_magnitude: pd.DataFrame,
) -> None:
    primary_by_domain = primary_bh.set_index("domain")
    lines = [
        "# Manuscript-ready Methods additions and Results",
        "",
        "## Methods additions needed for exact alignment",
        "",
        "For each domain, the primary loneliness-change-by-contact-transition interaction was evaluated with a three-degree-of-freedom HC3 robust Wald F test. The three fully adjusted domain-specific omnibus interaction tests formed the primary multiplicity family; both unadjusted p values and Benjamini-Hochberg false-discovery-rate-adjusted p values were reported.",
        "",
        "Adjusted predicted-value figures were generated for all three domains from fully adjusted Model 3, regardless of statistical significance, by averaging predictions over the observed covariate distribution of each domain-specific analytical sample. Confidence bands reflect parameter uncertainty calculated from the HC3 covariance matrix.",
        "",
        "For the categorical-change sensitivity analysis, stable or unchanged loneliness was the reference category. The six interaction coefficients between the two nonreference loneliness-change categories and the three nonreference contact-transition categories were tested jointly with an HC3 robust Wald F test.",
        "",
        "As a secondary explanatory analysis, the overall association between continuous loneliness change and T3 cognition was estimated without contact-transition variables or interaction terms. Model 1 adjusted for T2 cognition and T1 loneliness; Model 2 additionally adjusted for age, sex, education, and race/ethnicity; and Model 3 additionally adjusted for IHS-transformed wealth, marital status, and cohort. All three pooled models used the same contact-independent complete-case sample and HC3 robust standard errors. The fully adjusted model was then repeated separately in Cohorts A and B with the cohort indicator omitted. To formally compare the cohort-specific loneliness-change coefficients, a pooled fully adjusted model added a loneliness-change-by-cohort interaction; its interaction coefficient tested the Cohort B-minus-A slope difference.",
        "",
        "## Results",
        "",
        "### Descriptive characteristics and contact-transition patterns",
        "",
        "Participant characteristics for each domain-specific complete-case sample are presented in Table 1. Contact-transition patterns differed across domains. Maintained frequent contact was the most common pattern for friends and children, whereas the other-relatives sample was more evenly distributed between maintained frequent and maintained infrequent contact.",
        "",
    ]

    for domain in DOMAIN_SPECS:
        rows = transition_summary.loc[transition_summary["domain"].eq(domain)].set_index("contact_group")
        lines.append(
            f"For {DOMAIN_SPECS[domain]['label'].lower()} (N = {int(rows['N'].iloc[0]):,}), "
            f"{int(rows.loc['maintained_frequent', 'n']):,} ({rows.loc['maintained_frequent', 'percent']:.1f}%) maintained frequent contact, "
            f"{int(rows.loc['increased_contact', 'n']):,} ({rows.loc['increased_contact', 'percent']:.1f}%) increased contact, "
            f"{int(rows.loc['decreased_contact', 'n']):,} ({rows.loc['decreased_contact', 'percent']:.1f}%) decreased contact, and "
            f"{int(rows.loc['maintained_infrequent', 'n']):,} ({rows.loc['maintained_infrequent', 'percent']:.1f}%) maintained infrequent contact."
        )

    lines.extend([
        "",
        "### Primary continuous loneliness-change interaction analyses",
        "",
        f"The fully adjusted interaction was not statistically significant for friends, F(3, {int(primary_by_domain.loc['friends', 'df_denominator'])}) = {primary_by_domain.loc['friends', 'f_statistic']:.2f}, unadjusted p {p_display(primary_by_domain.loc['friends', 'unadjusted_p_value'])}, BH-adjusted p {p_display(primary_by_domain.loc['friends', 'bh_adjusted_p_value'])}; other relatives, F(3, {int(primary_by_domain.loc['other_relatives', 'df_denominator'])}) = {primary_by_domain.loc['other_relatives', 'f_statistic']:.2f}, unadjusted p {p_display(primary_by_domain.loc['other_relatives', 'unadjusted_p_value'])}, BH-adjusted p {p_display(primary_by_domain.loc['other_relatives', 'bh_adjusted_p_value'])}; or children, F(3, {int(primary_by_domain.loc['children', 'df_denominator'])}) = {primary_by_domain.loc['children', 'f_statistic']:.2f}, unadjusted p {p_display(primary_by_domain.loc['children', 'unadjusted_p_value'])}, BH-adjusted p {p_display(primary_by_domain.loc['children', 'bh_adjusted_p_value'])}. Thus, there was insufficient evidence that contact-transition pattern modified the association between change in loneliness and subsequent cognition in any domain after accounting for the three primary tests.",
        "",
    ])

    children_slopes = slopes.loc[slopes["domain"].eq("children")].set_index("contact_group")
    lines.extend(
        [
            "In the children model, the loneliness-change slope was negative within the maintained-frequent group but positive within the decreased-contact group. Although their pairwise contrast was nominally significant, the prespecified four-group omnibus interaction was not significant and the domain-level result did not survive the multiplicity adjustment. The pairwise finding was therefore treated as secondary and exploratory rather than evidence of effect modification.",
            "",
            "### Sensitivity analyses",
            "",
        ]
    )

    cat_by_domain = categorical.set_index("domain")
    lines.append(
        f"When loneliness change was categorized as decreasing, stable, or increasing, the six-degree-of-freedom interaction was not statistically significant for friends, F(6, {int(cat_by_domain.loc['friends', 'df_denominator'])}) = {cat_by_domain.loc['friends', 'f_statistic']:.2f}, p {p_display(cat_by_domain.loc['friends', 'p_value'])}, or children, F(6, {int(cat_by_domain.loc['children', 'df_denominator'])}) = {cat_by_domain.loc['children', 'f_statistic']:.2f}, p {p_display(cat_by_domain.loc['children', 'p_value'])}. The interaction was nominally significant for other relatives, F(6, {int(cat_by_domain.loc['other_relatives', 'df_denominator'])}) = {cat_by_domain.loc['other_relatives', 'f_statistic']:.2f}, p {p_display(cat_by_domain.loc['other_relatives', 'p_value'])}. This was a secondary sensitivity result; no multiplicity correction was prespecified for the sensitivity family. For context, a descriptive BH adjustment across the three categorical-change sensitivity tests gave q {p_display(cat_by_domain.loc['other_relatives', 'descriptive_bh_adjusted_p'])} for other relatives."
    )
    other_simple = categorical_simple_effects.loc[
        categorical_simple_effects["domain"].eq("other_relatives")
        & categorical_simple_effects["contrast"].eq("Decreasing vs stable loneliness")
    ].set_index("contact_group")
    lines.append("")
    lines.append(
        f"Within the other-relatives sensitivity model, decreasing rather than stable loneliness was associated with higher adjusted T3 cognition in the increased-contact group (difference = {other_simple.loc['increased_contact', 'estimate']:.4f}, 95% CI {other_simple.loc['increased_contact', 'ci_95_low']:.4f} to {other_simple.loc['increased_contact', 'ci_95_high']:.4f}, p {p_display(other_simple.loc['increased_contact', 'p_value'])}) and in the maintained-infrequent group (difference = {other_simple.loc['maintained_infrequent', 'estimate']:.4f}, 95% CI {other_simple.loc['maintained_infrequent', 'ci_95_low']:.4f} to {other_simple.loc['maintained_infrequent', 'ci_95_high']:.4f}, p {p_display(other_simple.loc['maintained_infrequent', 'p_value'])}), but not in the maintained-frequent or decreased-contact groups. Because this pattern was not present in the continuous primary interaction and did not remain significant under the descriptive sensitivity-family adjustment, it was interpreted cautiously as evidence that the other-relatives result may be somewhat sensitive to how loneliness change is parameterized rather than as a confirmed interaction."
    )
    lines.append("")

    cohort_parts = []
    for domain in DOMAIN_SPECS:
        for cohort in ["A", "B"]:
            row = cohort_sensitivity.loc[
                cohort_sensitivity["domain"].eq(domain) & cohort_sensitivity["cohort"].eq(cohort)
            ].iloc[0]
            cohort_parts.append(
                f"{DOMAIN_SPECS[domain]['label'].lower()} Cohort {cohort}: F(3, {int(row['df_denominator'])}) = {row['f_statistic']:.2f}, p {p_display(row['p_value'])}"
            )
    lines.append(
        "Cohort-specific fully adjusted models produced the following omnibus interaction tests: "
        + "; ".join(cohort_parts)
        + ". These sensitivity estimates should be interpreted by direction, magnitude, and uncertainty rather than by isolated p values because the cohorts were analyzed separately and the study was not powered around six additional primary tests."
    )
    lines.append("")
    lines.append(
        f"For reference, the maintained-frequent children slope in the pooled model was {children_slopes.loc['maintained_frequent', 'estimate']:.4f} (95% CI {children_slopes.loc['maintained_frequent', 'ci_95_low']:.4f} to {children_slopes.loc['maintained_frequent', 'ci_95_high']:.4f}), whereas the decreased-contact slope was {children_slopes.loc['decreased_contact', 'estimate']:.4f} (95% CI {children_slopes.loc['decreased_contact', 'ci_95_low']:.4f} to {children_slopes.loc['decreased_contact', 'ci_95_high']:.4f})."
    )
    lines.extend(["", "### Secondary overall loneliness-change analysis", ""])

    pooled = overall_effects.loc[overall_effects["analysis_scope"].eq("Pooled")].set_index("model")
    lines.append(
        f"The contact-independent secondary analysis included {int(pooled.loc['Model 3', 'N']):,} respondents. "
        f"The coefficient for continuous loneliness change was {pooled.loc['Model 1', 'estimate']:.4f} "
        f"(95% CI {pooled.loc['Model 1', 'ci_95_low']:.4f} to {pooled.loc['Model 1', 'ci_95_high']:.4f}, "
        f"p {p_display(pooled.loc['Model 1', 'p_value'])}) in Model 1, "
        f"{pooled.loc['Model 2', 'estimate']:.4f} "
        f"(95% CI {pooled.loc['Model 2', 'ci_95_low']:.4f} to {pooled.loc['Model 2', 'ci_95_high']:.4f}, "
        f"p {p_display(pooled.loc['Model 2', 'p_value'])}) in Model 2, and "
        f"{pooled.loc['Model 3', 'estimate']:.4f} "
        f"(95% CI {pooled.loc['Model 3', 'ci_95_low']:.4f} to {pooled.loc['Model 3', 'ci_95_high']:.4f}, "
        f"p {p_display(pooled.loc['Model 3', 'p_value'])}) in the fully adjusted Model 3. "
        "Each coefficient represents the adjusted difference in latent cognition at T3 per one-point increase in the UCLA-3 loneliness-change score."
    )
    magnitude = overall_magnitude.iloc[0]
    lines.append("")
    lines.append(
        f"For magnitude context, the fully adjusted coefficient corresponded to approximately "
        f"{abs(magnitude['outcome_sd_per_one_point_loneliness_change']):.3f} observed standard deviations of T3 cognition "
        f"per one-point increase in loneliness, or a standardized effect of "
        f"{magnitude['standardized_effect_per_one_sd_loneliness_change']:.4f} per one-standard-deviation increase in loneliness change. "
        "This descriptive rescaling confirms that the association was very small."
    )

    cohorts = overall_effects.loc[
        overall_effects["analysis_scope"].eq("Cohort-specific sensitivity")
    ].set_index("cohort")
    lines.append("")
    lines.append(
        f"In the brief cohort-specific check, the fully adjusted coefficient was {cohorts.loc['A', 'estimate']:.4f} "
        f"(95% CI {cohorts.loc['A', 'ci_95_low']:.4f} to {cohorts.loc['A', 'ci_95_high']:.4f}, "
        f"p {p_display(cohorts.loc['A', 'p_value'])}) in Cohort A and {cohorts.loc['B', 'estimate']:.4f} "
        f"(95% CI {cohorts.loc['B', 'ci_95_low']:.4f} to {cohorts.loc['B', 'ci_95_high']:.4f}, "
        f"p {p_display(cohorts.loc['B', 'p_value'])}) in Cohort B."
    )
    cohort_comparison = overall_cohort_comparison.set_index("cohort_or_difference")
    lines.append("")
    lines.append(
        f"In the formal pooled coefficient-comparison model, the adjusted loneliness-change slope was "
        f"{cohort_comparison.loc['A', 'estimate']:.4f} (95% CI "
        f"{cohort_comparison.loc['A', 'ci_95_low']:.4f} to "
        f"{cohort_comparison.loc['A', 'ci_95_high']:.4f}) for Cohort A and "
        f"{cohort_comparison.loc['B', 'estimate']:.4f} (95% CI "
        f"{cohort_comparison.loc['B', 'ci_95_low']:.4f} to "
        f"{cohort_comparison.loc['B', 'ci_95_high']:.4f}) for Cohort B. The Cohort B-minus-A "
        f"difference was {cohort_comparison.loc['B minus A', 'estimate']:.4f} (95% CI "
        f"{cohort_comparison.loc['B minus A', 'ci_95_low']:.4f} to "
        f"{cohort_comparison.loc['B minus A', 'ci_95_high']:.4f}, "
        f"p {p_display(cohort_comparison.loc['B minus A', 'p_value'])}). Thus, there was no "
        "statistical evidence that the overall loneliness-change coefficient differed between cohorts. "
        "The cohort-stratified estimates are therefore retained as a sensitivity check rather than as "
        "evidence of cohort-specific effect heterogeneity."
    )

    model3 = pooled.loc["Model 3"]
    if model3["p_value"] < 0.05:
        direction = "lower" if model3["estimate"] < 0 else "higher"
        explanatory_sentence = (
            f"The fully adjusted pooled association was statistically significant but small in magnitude: "
            f"greater loneliness was associated with {direction} subsequent cognition. This supports the interpretation "
            "that an overall loneliness-cognition association was present, but its strength did not differ reliably "
            "across contact-transition patterns."
        )
    else:
        explanatory_sentence = (
            "The fully adjusted pooled association was not statistically significant, supporting the interpretation "
            "that little underlying loneliness-cognition association was detectable in this analytical sample for the "
            "contact-transition variables to modify."
        )
    lines.extend(["", explanatory_sentence, "", "## Discussion", ""])
    lines.append(
        "The primary hypothesis was not supported. Across friends, other relatives, and children, the prespecified "
        "omnibus interaction tests did not provide robust evidence that contact-transition pattern modified the "
        "association between loneliness change and subsequent cognition. This conclusion was internally consistent "
        "across the continuous and categorical loneliness-change specifications and the two HRS cohorts. The isolated "
        "children pairwise contrast and the unadjusted other-relatives categorical interaction were secondary findings "
        "and did not overturn the primary conclusion."
    )
    lines.extend(["", explanatory_sentence])
    lines.extend([
        "",
        "One interpretation is that frequency of contact and subjective loneliness represent related but distinct dimensions of social experience. Contact frequency may therefore have limited capacity to alter the longitudinal loneliness-cognition association. In addition, a simple weekly-contact measure does not capture relationship quality, emotional closeness, reciprocity, conflict, or the cognitive content of social interactions, which may be more relevant to cognition than frequency alone.",
        "",
        "The longitudinal timing, baseline cognition adjustment, prespecified three-domain testing strategy, and convergence across multiple sensitivity analyses strengthen the interpretation of the null moderation result. Important limitations remain. The analyses were unweighted and complete-case, the contact measures emphasized frequency rather than quality, residual confounding is possible, and the results describe associations rather than causal effects. The findings are therefore sample-conditional and should not be interpreted as showing that social relationships or loneliness are unimportant for cognitive aging.",
        "",
        "Taken together, the results indicate that changes in simple contact frequency with friends, other relatives, or children did not reliably distinguish the association between loneliness change and later cognition in this HRS sample. Future research may benefit from measures that characterize the quality and functional content of social relationships, although no additional social variables or alternative outcomes were introduced in the present analysis.",
    ])
    text = "\n".join(lines) + "\n"
    (output_dir / "final_results_discussion.md").write_text(text, encoding="utf-8")
    (output_dir / "three_domain_methods_results_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    input_hash = sha256_file(input_path)
    master = pd.read_csv(input_path, dtype={"hhidpn": "string"})
    add_validation(checks, "input SHA-256", input_hash == EXPECTED_INPUT_SHA256, input_hash, EXPECTED_INPUT_SHA256)
    add_validation(checks, "master row count", len(master) == EXPECTED_ROWS, len(master), EXPECTED_ROWS)
    add_validation(checks, "master column count", master.shape[1] == EXPECTED_COLUMNS, master.shape[1], EXPECTED_COLUMNS)
    add_validation(checks, "unique respondent IDs", master["hhidpn"].nunique() == EXPECTED_ROWS, master["hhidpn"].nunique(), EXPECTED_ROWS)
    observed_cohorts = master["cohort"].value_counts().to_dict()
    add_validation(checks, "master cohort counts", observed_cohorts == EXPECTED_COHORT_COUNTS, observed_cohorts, EXPECTED_COHORT_COUNTS)

    primary_omnibus_rows = []
    primary_coefficient_rows = []
    primary_fit_rows = []
    primary_slope_rows = []
    prediction_frames = []
    categorical_omnibus_rows = []
    categorical_coefficient_rows = []
    categorical_simple_effect_rows = []
    categorical_prediction_frames = []
    cohort_omnibus_rows = []
    cohort_coefficient_rows = []
    cohort_slope_rows = []
    diagnostic_rows = []
    all_vif_rows = []
    sample_counts = {}
    transition_summary_rows = []
    overall_coefficient_rows = []
    overall_fit_rows = []
    overall_effect_rows = []
    overall_cohort_comparison_rows = []

    for domain, spec in DOMAIN_SPECS.items():
        sample = prepare_domain_sample(master, domain, spec, checks)
        sample_counts[domain] = {
            "N": len(sample),
            "cohort_counts": sample["cohort"].value_counts().to_dict(),
            "contact_counts": sample[spec["transition_col"]].value_counts().to_dict(),
            "change_category_counts": sample["change_category"].value_counts().to_dict(),
        }
        for group in CONTACT_ORDER:
            count = int(sample[spec["transition_col"]].eq(group).sum())
            transition_summary_rows.append(
                {
                    "domain": domain,
                    "domain_label": spec["label"],
                    "contact_group": group,
                    "contact_group_label": CONTACT_LABELS[group],
                    "N": len(sample),
                    "n": count,
                    "percent": 100.0 * count / len(sample),
                }
            )
        y = sample["cog T3"].to_numpy(float)
        primary_results = {}
        for model_number in [1, 2, 3]:
            X, names = build_continuous_design(sample, spec["transition_col"], model_number=model_number)
            result = fit_ols_hc3(X, y, names, f"{domain} Model {model_number}")
            primary_results[model_number] = result
            interaction_terms = ["delta_x_increased", "delta_x_decreased", "delta_x_maintained_infrequent"]
            test = wald_f(result, interaction_terms)
            test.update(
                {
                    "domain": domain,
                    "domain_label": spec["label"],
                    "model": f"Model {model_number}",
                    "covariance": "HC3",
                }
            )
            primary_omnibus_rows.append(test)
            fit_row = result["fit_row"].copy()
            fit_row.update({"domain": domain, "domain_label": spec["label"], "model": f"Model {model_number}"})
            primary_fit_rows.append(fit_row)
            for row in result["coefficient_rows"]:
                row = row.copy()
                row.update({"domain": domain, "domain_label": spec["label"], "model": f"Model {model_number}"})
                primary_coefficient_rows.append(row)

        model3 = primary_results[3]
        model3_test = primary_omnibus_rows[-1]
        add_validation(
            checks,
            f"{domain}: Model 3 F reproduces locked result",
            math.isclose(model3_test["f_statistic"], spec["expected_model3_f"], rel_tol=0, abs_tol=5e-10),
            model3_test["f_statistic"],
            spec["expected_model3_f"],
        )
        add_validation(
            checks,
            f"{domain}: Model 3 p reproduces locked result",
            math.isclose(model3_test["p_value"], spec["expected_model3_p"], rel_tol=0, abs_tol=5e-10),
            model3_test["p_value"],
            spec["expected_model3_p"],
        )
        min_eigenvalue = float(np.linalg.eigvalsh(model3["covariance"]).min())
        add_validation(checks, f"{domain}: HC3 covariance positive semidefinite", min_eigenvalue >= -1e-10, min_eigenvalue, ">= -1e-10")
        primary_slope_rows.extend(continuous_group_slopes(model3, domain))
        prediction_frames.append(create_predictions(sample, spec, model3, domain))
        diagnostic_rows.append(diagnostic_row(model3, domain))
        all_vif_rows.extend(vif_rows(model3, domain))

        categorical_X, categorical_names = build_categorical_change_design(sample, spec["transition_col"])
        categorical_result = fit_ols_hc3(categorical_X, y, categorical_names, f"{domain} categorical-change Model 3")
        categorical_interaction_terms = [
            "decreasing_x_increased",
            "decreasing_x_decreased",
            "decreasing_x_maintained_infrequent",
            "increasing_x_increased",
            "increasing_x_decreased",
            "increasing_x_maintained_infrequent",
        ]
        categorical_test = wald_f(categorical_result, categorical_interaction_terms)
        categorical_test.update(
            {
                "domain": domain,
                "domain_label": spec["label"],
                "N": len(sample),
                "change_reference": "stable",
                "contact_reference": "maintained_frequent",
                "covariance": "HC3",
            }
        )
        categorical_omnibus_rows.append(categorical_test)
        for row in categorical_result["coefficient_rows"]:
            row = row.copy()
            row.update({"domain": domain, "domain_label": spec["label"], "model": "Categorical-change Model 3"})
            categorical_coefficient_rows.append(row)
        categorical_simple_effect_rows.extend(categorical_change_simple_effects(categorical_result, domain))
        categorical_prediction_frames.append(
            create_categorical_predictions(sample, spec, categorical_result, domain)
        )

        for cohort in ["A", "B"]:
            cohort_sample = sample.loc[sample["cohort"].eq(cohort)].copy()
            cohort_y = cohort_sample["cog T3"].to_numpy(float)
            cohort_X, cohort_names = build_continuous_design(
                cohort_sample,
                spec["transition_col"],
                model_number=3,
                include_cohort=False,
            )
            cohort_result = fit_ols_hc3(cohort_X, cohort_y, cohort_names, f"{domain} Cohort {cohort} Model 3")
            cohort_test = wald_f(
                cohort_result,
                ["delta_x_increased", "delta_x_decreased", "delta_x_maintained_infrequent"],
            )
            cohort_test.update(
                {
                    "domain": domain,
                    "domain_label": spec["label"],
                    "cohort": cohort,
                    "N": len(cohort_sample),
                    "covariance": "HC3",
                    "cohort_indicator_included": False,
                }
            )
            cohort_omnibus_rows.append(cohort_test)
            cohort_slope_rows.extend(continuous_group_slopes(cohort_result, domain, cohort=cohort))
            for row in cohort_result["coefficient_rows"]:
                row = row.copy()
                row.update(
                    {
                        "domain": domain,
                        "domain_label": spec["label"],
                        "cohort": cohort,
                        "model": "Cohort-specific Model 3",
                    }
                )
                cohort_coefficient_rows.append(row)

    # Professor-requested secondary explanatory analysis. The sample requires
    # complete longitudinal cognition, loneliness, and planned covariates, but
    # deliberately does not require any contact-domain information.
    overall_sample = prepare_overall_sample(master, checks)
    sample_counts["overall_contact_independent"] = {
        "N": len(overall_sample),
        "cohort_counts": overall_sample["cohort"].value_counts().to_dict(),
    }
    overall_y = overall_sample["cog T3"].to_numpy(float)
    overall_results = {}
    for model_number in [1, 2, 3]:
        X, names = build_overall_association_design(
            overall_sample,
            model_number=model_number,
            include_cohort=True,
        )
        result = fit_ols_hc3(X, overall_y, names, f"overall association Model {model_number}")
        overall_results[model_number] = result
        min_eigenvalue = float(np.linalg.eigvalsh(result["covariance"]).min())
        add_validation(
            checks,
            f"overall: Model {model_number} HC3 covariance positive semidefinite",
            min_eigenvalue >= -1e-10,
            min_eigenvalue,
            ">= -1e-10",
        )
        fit_row = result["fit_row"].copy()
        fit_row.update(
            {
                "analysis_scope": "Pooled",
                "cohort": "Combined",
                "model": f"Model {model_number}",
                "covariance": "HC3",
            }
        )
        overall_fit_rows.append(fit_row)
        for row in result["coefficient_rows"]:
            coefficient_row = row.copy()
            coefficient_row.update(
                {
                    "analysis_scope": "Pooled",
                    "cohort": "Combined",
                    "model": f"Model {model_number}",
                    "covariance": "HC3",
                }
            )
            overall_coefficient_rows.append(coefficient_row)
            if row["term"] == "delta_loneliness":
                effect_row = coefficient_row.copy()
                effect_row.update(
                    {
                        "N": result["n"],
                        "effect_unit": "Adjusted T3 cognition difference per one-point increase in UCLA-3 loneliness change",
                    }
                )
                overall_effect_rows.append(effect_row)

    add_validation(
        checks,
        "overall: Models 1-3 use the same analytical sample",
        {result["n"] for result in overall_results.values()} == {EXPECTED_OVERALL_N},
        sorted({result["n"] for result in overall_results.values()}),
        [EXPECTED_OVERALL_N],
    )

    # Formal comparison of the two cohort-specific loneliness-change slopes.
    # This pooled interaction model is the valid test of coefficient equality;
    # comparing whether two separate p values cross .05 is not such a test.
    comparison_X, comparison_names = build_overall_cohort_comparison_design(overall_sample)
    comparison_result = fit_ols_hc3(
        comparison_X,
        overall_y,
        comparison_names,
        "overall association formal cohort comparison",
    )
    comparison_min_eigenvalue = float(
        np.linalg.eigvalsh(comparison_result["covariance"]).min()
    )
    add_validation(
        checks,
        "overall: formal cohort-comparison N",
        comparison_result["n"] == EXPECTED_OVERALL_N,
        comparison_result["n"],
        EXPECTED_OVERALL_N,
    )
    add_validation(
        checks,
        "overall: formal cohort-comparison HC3 covariance positive semidefinite",
        comparison_min_eigenvalue >= -1e-10,
        comparison_min_eigenvalue,
        ">= -1e-10",
    )

    delta_index = comparison_names.index("delta_loneliness")
    interaction_index = comparison_names.index("delta_x_cohort_B")
    comparison_specs = [
        ("Cohort A loneliness-change slope", "A", [(delta_index, 1.0)], False),
        (
            "Cohort B loneliness-change slope",
            "B",
            [(delta_index, 1.0), (interaction_index, 1.0)],
            False,
        ),
        (
            "Cohort B minus Cohort A slope difference",
            "B minus A",
            [(interaction_index, 1.0)],
            True,
        ),
    ]
    for comparison_label, cohort_label, nonzero_weights, formal_test in comparison_specs:
        weights = np.zeros(len(comparison_names))
        for index, value in nonzero_weights:
            weights[index] = value
        row = linear_contrast(comparison_result, weights)
        row.update(
            {
                "comparison": comparison_label,
                "cohort_or_difference": cohort_label,
                "formal_equality_test": formal_test,
                "N": comparison_result["n"],
                "model": "Pooled Model 3 plus loneliness-change-by-cohort interaction",
                "covariance": "HC3",
                "reference_cohort": "Cohort A",
                "effect_unit": "Adjusted T3 cognition difference per one-point UCLA-3 loneliness change",
            }
        )
        overall_cohort_comparison_rows.append(row)

    comparison_fit_row = comparison_result["fit_row"].copy()
    comparison_fit_row.update(
        {
            "analysis_scope": "Formal cohort comparison",
            "cohort": "Combined",
            "model": "Model 3 plus loneliness-change-by-cohort interaction",
            "covariance": "HC3",
        }
    )
    overall_fit_rows.append(comparison_fit_row)
    for row in comparison_result["coefficient_rows"]:
        coefficient_row = row.copy()
        coefficient_row.update(
            {
                "analysis_scope": "Formal cohort comparison",
                "cohort": "Combined",
                "model": "Model 3 plus loneliness-change-by-cohort interaction",
                "covariance": "HC3",
            }
        )
        overall_coefficient_rows.append(coefficient_row)

    for cohort in ["A", "B"]:
        cohort_sample = overall_sample.loc[overall_sample["cohort"].eq(cohort)].copy()
        cohort_y = cohort_sample["cog T3"].to_numpy(float)
        cohort_X, cohort_names = build_overall_association_design(
            cohort_sample,
            model_number=3,
            include_cohort=False,
        )
        cohort_result = fit_ols_hc3(
            cohort_X,
            cohort_y,
            cohort_names,
            f"overall association Cohort {cohort} Model 3",
        )
        min_eigenvalue = float(np.linalg.eigvalsh(cohort_result["covariance"]).min())
        add_validation(
            checks,
            f"overall: Cohort {cohort} Model 3 HC3 covariance positive semidefinite",
            min_eigenvalue >= -1e-10,
            min_eigenvalue,
            ">= -1e-10",
        )
        fit_row = cohort_result["fit_row"].copy()
        fit_row.update(
            {
                "analysis_scope": "Cohort-specific sensitivity",
                "cohort": cohort,
                "model": "Model 3",
                "covariance": "HC3",
            }
        )
        overall_fit_rows.append(fit_row)
        for row in cohort_result["coefficient_rows"]:
            coefficient_row = row.copy()
            coefficient_row.update(
                {
                    "analysis_scope": "Cohort-specific sensitivity",
                    "cohort": cohort,
                    "model": "Model 3",
                    "covariance": "HC3",
                }
            )
            overall_coefficient_rows.append(coefficient_row)
            if row["term"] == "delta_loneliness":
                effect_row = coefficient_row.copy()
                effect_row.update(
                    {
                        "N": cohort_result["n"],
                        "effect_unit": "Adjusted T3 cognition difference per one-point increase in UCLA-3 loneliness change",
                    }
                )
                overall_effect_rows.append(effect_row)

    primary_omnibus = pd.DataFrame(primary_omnibus_rows)
    primary_coefficients = pd.DataFrame(primary_coefficient_rows)
    primary_fit = pd.DataFrame(primary_fit_rows)
    primary_slopes = pd.DataFrame(primary_slope_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    categorical_omnibus = pd.DataFrame(categorical_omnibus_rows)
    categorical_coefficients = pd.DataFrame(categorical_coefficient_rows)
    categorical_simple_effects = pd.DataFrame(categorical_simple_effect_rows)
    categorical_predictions = pd.concat(categorical_prediction_frames, ignore_index=True)
    cohort_omnibus = pd.DataFrame(cohort_omnibus_rows)
    cohort_coefficients = pd.DataFrame(cohort_coefficient_rows)
    cohort_slopes = pd.DataFrame(cohort_slope_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    vif = pd.DataFrame(all_vif_rows)
    overall_coefficients = pd.DataFrame(overall_coefficient_rows)
    overall_fit = pd.DataFrame(overall_fit_rows)
    overall_effects = pd.DataFrame(overall_effect_rows)
    overall_cohort_comparison = pd.DataFrame(overall_cohort_comparison_rows)
    transition_summary = pd.DataFrame(transition_summary_rows)
    pooled_model3_effect = overall_effects.loc[
        overall_effects["analysis_scope"].eq("Pooled")
        & overall_effects["model"].eq("Model 3")
    ].iloc[0]
    outcome_sd = float(overall_sample["cog T3"].std(ddof=1))
    loneliness_change_sd = float(overall_sample["delta_loneliness"].std(ddof=1))
    overall_magnitude = pd.DataFrame(
        [
            {
                "N": len(overall_sample),
                "model": "Pooled Model 3",
                "coefficient_per_one_point_loneliness_change": pooled_model3_effect["estimate"],
                "observed_t3_cognition_sd": outcome_sd,
                "observed_loneliness_change_sd": loneliness_change_sd,
                "outcome_sd_per_one_point_loneliness_change": pooled_model3_effect["estimate"] / outcome_sd,
                "cognition_units_per_one_sd_loneliness_change": pooled_model3_effect["estimate"] * loneliness_change_sd,
                "standardized_effect_per_one_sd_loneliness_change": (
                    pooled_model3_effect["estimate"] * loneliness_change_sd / outcome_sd
                ),
                "note": "Descriptive rescaling of the fully adjusted coefficient; not an additional regression model.",
            }
        ]
    )

    categorical_adjusted, categorical_ranks = bh_adjust(categorical_omnibus["p_value"].to_numpy(float))
    categorical_omnibus["descriptive_bh_rank"] = categorical_ranks
    categorical_omnibus["descriptive_bh_adjusted_p"] = categorical_adjusted
    categorical_omnibus["multiplicity_note"] = (
        "Descriptive BH context across the three categorical-change sensitivity tests; "
        "not the prespecified primary testing family"
    )

    primary_bh = primary_omnibus.loc[primary_omnibus["model"].eq("Model 3")].copy()
    adjusted, ranks = bh_adjust(primary_bh["p_value"].to_numpy(float))
    primary_bh["unadjusted_p_value"] = primary_bh["p_value"]
    primary_bh["bh_rank"] = ranks
    primary_bh["bh_adjusted_p_value"] = adjusted
    primary_bh["bh_reject_at_fdr_0_05"] = adjusted <= 0.05
    primary_bh["primary_conclusion"] = np.where(
        primary_bh["bh_reject_at_fdr_0_05"],
        "Reject domain omnibus null at BH FDR 0.05",
        "Do not reject domain omnibus null at BH FDR 0.05",
    )
    primary_bh = primary_bh.drop(columns=["p_value"])

    primary_omnibus.to_csv(output_dir / "three_domain_primary_omnibus_all_models.csv", index=False, float_format="%.12g")
    primary_bh.to_csv(output_dir / "three_domain_model3_omnibus_bh.csv", index=False, float_format="%.12g")
    primary_coefficients.to_csv(output_dir / "three_domain_primary_coefficients.csv", index=False, float_format="%.12g")
    primary_fit.to_csv(output_dir / "three_domain_primary_model_fit.csv", index=False, float_format="%.12g")
    primary_slopes.to_csv(output_dir / "three_domain_model3_group_slopes.csv", index=False, float_format="%.12g")
    predictions.to_csv(output_dir / "three_domain_adjusted_predictions.csv", index=False, float_format="%.12g")
    categorical_omnibus.to_csv(output_dir / "categorical_change_sensitivity_omnibus.csv", index=False, float_format="%.12g")
    categorical_coefficients.to_csv(output_dir / "categorical_change_sensitivity_coefficients.csv", index=False, float_format="%.12g")
    categorical_simple_effects.to_csv(output_dir / "categorical_change_sensitivity_simple_effects.csv", index=False, float_format="%.12g")
    categorical_predictions.to_csv(output_dir / "categorical_change_sensitivity_adjusted_predictions.csv", index=False, float_format="%.12g")
    cohort_omnibus.to_csv(output_dir / "cohort_specific_sensitivity_omnibus.csv", index=False, float_format="%.12g")
    cohort_coefficients.to_csv(output_dir / "cohort_specific_sensitivity_coefficients.csv", index=False, float_format="%.12g")
    cohort_slopes.to_csv(output_dir / "cohort_specific_sensitivity_group_slopes.csv", index=False, float_format="%.12g")
    diagnostics.to_csv(output_dir / "three_domain_model3_diagnostic_summary.csv", index=False, float_format="%.12g")
    vif.to_csv(output_dir / "three_domain_model3_vif.csv", index=False, float_format="%.12g")
    overall_effects.to_csv(
        output_dir / "overall_loneliness_change_key_effects.csv",
        index=False,
        float_format="%.12g",
    )
    overall_coefficients.to_csv(
        output_dir / "overall_loneliness_change_coefficients.csv",
        index=False,
        float_format="%.12g",
    )
    overall_fit.to_csv(
        output_dir / "overall_loneliness_change_model_fit.csv",
        index=False,
        float_format="%.12g",
    )
    transition_summary.to_csv(
        output_dir / "contact_transition_summary.csv",
        index=False,
        float_format="%.12g",
    )
    overall_magnitude.to_csv(
        output_dir / "overall_loneliness_change_magnitude_context.csv",
        index=False,
        float_format="%.12g",
    )
    overall_cohort_comparison.to_csv(
        output_dir / "overall_loneliness_change_cohort_comparison.csv",
        index=False,
        float_format="%.12g",
    )

    assumptions = pd.DataFrame(
        [
            ["A01", "Frozen input", "Use the 8,041-row, 79-column master with the recorded SHA-256 hash.", "Fixed"],
            ["A02", "Primary inference", "Unweighted OLS with HC3 covariance and residual-df t/F reference distributions.", "Fixed"],
            ["A03", "Primary multiplicity", "BH adjustment applies only to the three pooled Model 3 omnibus interaction tests.", "Fixed"],
            ["A04", "Categorical-change reference", "Stable or unchanged loneliness is the reference category; maintained frequent remains the contact reference.", "Methods-required implementation assumption"],
            ["A05", "Categorical sensitivity test", "Joint HC3 robust Wald F test of six change-category-by-contact interaction terms.", "Methods-required implementation assumption"],
            ["A06", "Cohort sensitivity", "Fit fully adjusted continuous-change Model 3 separately by cohort and omit cohort indicator.", "Methods-specified"],
            ["A07", "Prediction range", "Display pooled Model 3 marginal predictions from -1 to +1 loneliness change for comparability and support.", "Fixed runbook default"],
            ["A08", "Stochastic steps", "No imputation, resampling, random initialization, or stochastic optimization.", "None"],
            ["A09", "Interpretation", "Results are sample-conditional associations, not nationally representative or causal effects.", "Fixed"],
            ["A10", "Overall-association sample", "Use the 5,243-person fully adjusted longitudinal sample without requiring contact data.", "Professor-specified"],
            ["A11", "Overall-association sequence", "Model 1 adjusts for T2 cognition and T1 loneliness; Model 2 adds age, sex, education, and race/ethnicity; Model 3 adds IHS wealth, marital status, and cohort.", "Professor-specified"],
            ["A12", "Overall cohort check", "Repeat fully adjusted Model 3 separately in Cohorts A and B and omit the cohort indicator; do not add further exploratory models.", "Professor-specified"],
            ["A13", "Formal cohort comparison", "Add a loneliness-change-by-cohort interaction to pooled fully adjusted Model 3; use the HC3 interaction test to evaluate equality of the Cohort A and B slopes.", "User-requested follow-up"],
        ],
        columns=["assumption_id", "topic", "decision_or_assumption", "status"],
    )
    assumptions.to_csv(output_dir / "three_domain_assumptions_log.csv", index=False)
    pd.DataFrame(checks).to_csv(output_dir / "three_domain_validation_checks.csv", index=False)

    plot_predictions(predictions, output_dir)
    write_manuscript_text(
        output_dir,
        primary_bh,
        primary_slopes,
        categorical_omnibus,
        categorical_simple_effects,
        cohort_omnibus,
        overall_effects,
        overall_cohort_comparison,
        transition_summary,
        overall_magnitude,
    )

    step_log = pd.DataFrame(
        [
            [1, "Validate Methods and frozen input", "PASS", "Primary formula matched; sensitivity and reporting gaps identified; input hash and dimensions passed."],
            [2, "Reconstruct domain samples", "PASS", "Stored eligibility flags, sample counts, cohort counts, transition counts, loneliness scores, and transition coding passed."],
            [3, "Fit pooled Models 1-3", "PASS", "All nine primary models were full rank and used HC3 covariance on fixed domain-specific complete-case samples."],
            [4, "Test primary interactions", "PASS", "Three-df HC3 robust Wald F tests were computed for each model and domain."],
            [5, "Apply primary multiplicity adjustment", "PASS", "BH FDR adjustment was applied across exactly the three pooled Model 3 domain omnibus tests."],
            [6, "Run categorical-change sensitivity", "PASS", "Three fully adjusted Model 3 sensitivity models and six-df interaction tests were completed."],
            [7, "Run cohort-specific sensitivity", "PASS", "Six cohort-specific fully adjusted Model 3 analyses were completed without cohort indicators."],
            [8, "Run pooled overall loneliness-change models", "PASS", "Contact-independent Models 1-3 were completed on the same 5,243-person fully adjusted sample with HC3 inference."],
            [9, "Run overall cohort checks", "PASS", "The fully adjusted contact-independent model was repeated in Cohorts A and B with the cohort indicator omitted."],
            [10, "Compare overall cohort coefficients", "PASS", "A pooled fully adjusted loneliness-change-by-cohort interaction formally tested the Cohort B-minus-A slope difference using HC3 inference."],
            [11, "Generate figures and diagnostics", "PASS", "Common-scale interaction figures, HC3 confidence bands, residual diagnostics, influence summaries, and VIFs were saved."],
            [12, "Write paper-ready text", "PASS", "Results and Discussion were generated in the professor-specified hierarchy directly from numeric outputs."],
        ],
        columns=["step", "action", "status", "outcome"],
    )
    step_log.to_csv(output_dir / "three_domain_analysis_step_log.csv", index=False)

    summary = {
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "rows": len(master),
            "columns": master.shape[1],
        },
        "sample_counts": sample_counts,
        "primary_model3_omnibus": primary_bh.to_dict(orient="records"),
        "categorical_change_sensitivity_omnibus": categorical_omnibus.to_dict(orient="records"),
        "cohort_specific_sensitivity_omnibus": cohort_omnibus.to_dict(orient="records"),
        "overall_loneliness_change_effects": overall_effects.to_dict(orient="records"),
        "overall_loneliness_change_cohort_comparison": overall_cohort_comparison.to_dict(orient="records"),
        "overall_loneliness_change_magnitude_context": overall_magnitude.to_dict(orient="records"),
        "validation": {
            "checks": len(checks),
            "passed": int(sum(row["status"] == "PASS" for row in checks)),
            "failed": int(sum(row["status"] == "FAIL" for row in checks)),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "analysis_scope": {
            "primary_models": 9,
            "categorical_change_sensitivity_models": 3,
            "cohort_specific_sensitivity_models": 6,
            "overall_loneliness_change_models": 5,
            "formal_cohort_comparison_models": 1,
            "total_regression_models": 24,
            "survey_weighted": False,
            "causal_interpretation": False,
        },
    }
    write_json(output_dir / "three_domain_key_results.json", summary)

    manifest_files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "analysis_manifest.json":
            manifest_files.append(
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "input": summary["input"],
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "outputs": manifest_files,
            "deterministic": True,
        },
    )

    print(f"Methods-aligned three-domain analysis completed: {output_dir}")
    print(primary_bh[["domain_label", "f_statistic", "df_denominator", "unadjusted_p_value", "bh_adjusted_p_value"]].to_string(index=False))
    print("Categorical-change sensitivity:")
    print(categorical_omnibus[["domain_label", "f_statistic", "df_denominator", "p_value"]].to_string(index=False))
    print("Cohort-specific sensitivity:")
    print(cohort_omnibus[["domain_label", "cohort", "f_statistic", "df_denominator", "p_value"]].to_string(index=False))
    print("Overall loneliness-change association:")
    print(
        overall_effects[
            ["analysis_scope", "cohort", "model", "N", "estimate", "ci_95_low", "ci_95_high", "p_value"]
        ].to_string(index=False)
    )
    print("Formal cohort coefficient comparison:")
    print(
        overall_cohort_comparison[
            ["comparison", "estimate", "ci_95_low", "ci_95_high", "p_value"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
