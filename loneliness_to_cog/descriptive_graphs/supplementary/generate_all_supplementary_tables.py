#!/usr/bin/env python3
"""Generate Supplementary Tables S1A-S1C and S2-S4 for the HRS paper.

Inputs
------
1. The frozen 8,041-row participant-level HRS analysis file.
2. The final-results directory produced by run_final_phase_analysis.py.

Outputs
-------
Six manuscript-formatted CSV files, one for each supplementary table.

 python run_final_phase_analysis.py \
    --input "HRS_participant_level_cohorts_current.csv" \
    --output-dir "final_phase_results"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DOMAIN_SPECS = {
    "friends": {
        "domain_label": "Friends",
        "table": "S1A",
        "file_stub": "friends_characteristics",
        "eligibility_col": "eligible_friends_contact_fully_adjusted_model",
        "contact_t1_col": "weekly_contact_friends_T1",
        "contact_t2_col": "weekly_contact_friends_T2",
        "transition_col": "friends_contact_transition",
        "contact_variable_label": "Friends contact transition",
        "expected_n": 5068,
        "expected_cohorts": {"A": 2640, "B": 2428},
        "expected_contact": {
            "maintained_frequent": 2785,
            "increased_contact": 622,
            "decreased_contact": 691,
            "maintained_infrequent": 970,
        },
    },
    "other_relatives": {
        "domain_label": "Other relatives",
        "table": "S1B",
        "file_stub": "other_relative_characteristics",
        "eligibility_col": "eligible_other_relatives_contact_fully_adjusted_model",
        "contact_t1_col": "weekly_contact_other_relatives_T1",
        "contact_t2_col": "weekly_contact_other_relatives_T2",
        "transition_col": "other_relatives_contact_transition",
        "contact_variable_label": "Other relatives contact transition",
        "expected_n": 5080,
        "expected_cohorts": {"A": 2638, "B": 2442},
        "expected_contact": {
            "maintained_frequent": 1873,
            "increased_contact": 794,
            "decreased_contact": 743,
            "maintained_infrequent": 1670,
        },
    },
    "children": {
        "domain_label": "Children",
        "table": "S1C",
        "file_stub": "children_characteristics",
        "eligibility_col": "eligible_children_contact_fully_adjusted_model",
        "contact_t1_col": "weekly_contact_children_T1",
        "contact_t2_col": "weekly_contact_children_T2",
        "transition_col": "children_contact_transition",
        "contact_variable_label": "Children contact transition",
        "expected_n": 4717,
        "expected_cohorts": {"A": 2465, "B": 2252},
        "expected_contact": {
            "maintained_frequent": 3536,
            "increased_contact": 342,
            "decreased_contact": 399,
            "maintained_infrequent": 440,
        },
    },
}

DOMAIN_ORDER = ["Friends", "Other relatives", "Children"]
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
CONTACT_LABEL_ORDER = [CONTACT_LABELS[x] for x in CONTACT_ORDER]
RACE_ORDER = [
    "White, non-Hispanic",
    "Black/African American, non-Hispanic",
    "Hispanic",
    "Other, non-Hispanic",
]
MARITAL_ORDER = [
    "Married/partnered",
    "Separated/divorced",
    "Widowed",
    "Never married",
]


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
    recoded = series.map(mapping)
    unmapped = series.notna() & recoded.isna()
    if unmapped.any():
        raise ValueError(f"Unmapped marital labels: {sorted(series[unmapped].unique())}")
    return recoded


def prepare_master(input_path: Path) -> pd.DataFrame:
    master = pd.read_csv(input_path, dtype={"hhidpn": "string"})
    t1_items = [
        "lack_companionship_T1",
        "left_out_T1",
        "feels_isolated_T1",
    ]
    t2_items = [
        "lack_companionship_T2",
        "left_out_T2",
        "feels_isolated_T2",
    ]
    master["ucla3_T1"] = master[t1_items].sum(axis=1, min_count=3) / 3.0
    master["ucla3_T2"] = master[t2_items].sum(axis=1, min_count=3) / 3.0
    master["delta_loneliness"] = master["ucla3_T2"] - master["ucla3_T1"]
    return master


def prepare_domain_sample(master: pd.DataFrame, spec: dict) -> pd.DataFrame:
    covariates = [
        "age_T2",
        "sex",
        "education",
        "race_ethnicity",
        "wealth_T2",
        "marital_status_T2",
    ]
    reconstructed = (
        master["self_completed_loneliness_T1"].eq(1)
        & master["self_completed_loneliness_T2"].eq(1)
        & master["cog T2"].notna()
        & master["cog T3"].notna()
        & master[spec["contact_t1_col"]].isin([0, 1])
        & master[spec["contact_t2_col"]].isin([0, 1])
        & master[covariates].notna().all(axis=1)
    )

    stored = master[spec["eligibility_col"]].eq(1)
    if not stored.equals(reconstructed):
        n_mismatch = int((stored != reconstructed).sum())
        raise ValueError(
            f"{spec['domain_label']}: stored eligibility differs from the "
            f"reconstructed rule for {n_mismatch} rows"
        )

    sample = master.loc[reconstructed].copy()
    sample["marital_group"] = marital_recode(sample["marital_status_T2"])

    if len(sample) != spec["expected_n"]:
        raise ValueError(
            f"{spec['domain_label']}: expected N={spec['expected_n']}; "
            f"observed N={len(sample)}"
        )
    if sample["cohort"].value_counts().to_dict() != spec["expected_cohorts"]:
        raise ValueError(f"{spec['domain_label']}: unexpected cohort counts")
    observed_contact = sample[spec["transition_col"]].value_counts().to_dict()
    if observed_contact != spec["expected_contact"]:
        raise ValueError(f"{spec['domain_label']}: unexpected contact counts")

    return sample


def characteristics_table(sample: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Create S1A, S1B, or S1C using the original display rules."""
    sample_sets = {
        "Combined": sample,
        "Cohort A": sample.loc[sample["cohort"].eq("A")],
        "Cohort B": sample.loc[sample["cohort"].eq("B")],
    }
    rows: list[dict[str, str]] = []

    def add(section: str, characteristic: str, values: dict[str, str]) -> None:
        rows.append({"section": section, "characteristic": characteristic, **values})

    add("Sample", "N", {name: f"{len(x):,}" for name, x in sample_sets.items()})

    continuous = [
        ("Age at T2, years", "age_T2"),
        ("Education, years", "education"),
        ("Loneliness at T1", "ucla3_T1"),
        ("Loneliness at T2", "ucla3_T2"),
        ("Change in loneliness (T2 − T1)", "delta_loneliness"),
        ("Cognition at T2", "cog T2"),
        ("Cognition at T3", "cog T3"),
    ]
    for label, column in continuous:
        display = {}
        for sample_name, frame in sample_sets.items():
            values = frame[column].astype(float)
            display[sample_name] = f"{values.mean():.2f} ({values.std(ddof=1):.2f})"
        add("Continuous: mean (SD)", label, display)

    wealth_display = {}
    for sample_name, frame in sample_sets.items():
        q1, median, q3 = frame["wealth_T2"].astype(float).quantile([0.25, 0.50, 0.75])
        wealth_display[sample_name] = f"${median:,.0f} [${q1:,.0f}, ${q3:,.0f}]"
    add("Continuous: median [IQR]", "Wealth at T2, dollars", wealth_display)

    categorical = [
        ("Sex", "sex", ["Female", "Male"], lambda x: x),
        ("Race/ethnicity", "race_ethnicity", RACE_ORDER, lambda x: x),
        ("Marital status at T2", "marital_group", MARITAL_ORDER, lambda x: x),
        (
            spec["contact_variable_label"],
            spec["transition_col"],
            CONTACT_ORDER,
            lambda x: CONTACT_LABELS[x],
        ),
    ]
    for variable_label, column, levels, label_function in categorical:
        for level in levels:
            display = {}
            for sample_name, frame in sample_sets.items():
                denominator = int(frame[column].notna().sum())
                count = int(frame[column].eq(level).sum())
                percent = 100.0 * count / denominator
                display[sample_name] = f"{count:,} ({percent:.1f}%)"
            add(
                "Categorical: n (%)",
                f"{variable_label}: {label_function(level)}",
                display,
            )

    return pd.DataFrame(rows)


def p_fmt(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"{value:.3f}".lstrip("0")


def f_fmt(value: float, df1: int, df2: int) -> str:
    return f"F({df1}, {df2}) = {value:.2f}"


def supplementary_table_s2(results_dir: Path) -> pd.DataFrame:
    slopes = pd.read_csv(results_dir / "three_domain_model3_group_slopes.csv")
    rows = []
    for domain in DOMAIN_ORDER:
        for group in CONTACT_LABEL_ORDER:
            row = slopes.loc[
                slopes["domain_label"].eq(domain)
                & slopes["contact_group_label"].eq(group)
            ].iloc[0]
            rows.append(
                {
                    "Domain": domain,
                    "Contact transition": group,
                    "Slope": f"{row['estimate']:.4f}",
                    "95% CI": f"[{row['ci_95_low']:.4f}, {row['ci_95_high']:.4f}]",
                    "p": p_fmt(float(row["p_value"])),
                }
            )
    return pd.DataFrame(rows)


def supplementary_table_s3(results_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(
        results_dir / "categorical_change_sensitivity_omnibus.csv"
    )
    rows = []
    for domain in DOMAIN_ORDER:
        row = results.loc[results["domain_label"].eq(domain)].iloc[0]
        rows.append(
            {
                "Domain": domain,
                "Omnibus interaction": f_fmt(
                    float(row["f_statistic"]),
                    int(row["df_numerator"]),
                    int(row["df_denominator"]),
                ),
                "p": p_fmt(float(row["p_value"])),
                "Descriptive BH-adjusted p": p_fmt(
                    float(row["descriptive_bh_adjusted_p"])
                ),
            }
        )
    return pd.DataFrame(rows)


def supplementary_table_s4(results_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(results_dir / "cohort_specific_sensitivity_omnibus.csv")
    rows = []
    for domain in DOMAIN_ORDER:
        for cohort in ["A", "B"]:
            row = results.loc[
                results["domain_label"].eq(domain) & results["cohort"].eq(cohort)
            ].iloc[0]
            rows.append(
                {
                    "Domain": domain,
                    "Cohort": cohort,
                    "N": f"{int(row['N']):,}",
                    "Omnibus interaction": f_fmt(
                        float(row["f_statistic"]),
                        int(row["df_numerator"]),
                        int(row["df_denominator"]),
                    ),
                    "p": p_fmt(float(row["p_value"])),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master = prepare_master(args.input.resolve())

    for spec in DOMAIN_SPECS.values():
        sample = prepare_domain_sample(master, spec)
        table = characteristics_table(sample, spec)
        filename = (
            f"Supplementary_Table_{spec['table']}_{spec['file_stub']}.csv"
        )
        table.to_csv(output_dir / filename, index=False)

    supplementary_table_s2(args.results_dir.resolve()).to_csv(
        output_dir / "Supplementary_Table_S2_group_specific_slopes.csv",
        index=False,
    )
    supplementary_table_s3(args.results_dir.resolve()).to_csv(
        output_dir / "Supplementary_Table_S3_categorical_sensitivity.csv",
        index=False,
    )
    supplementary_table_s4(args.results_dir.resolve()).to_csv(
        output_dir / "Supplementary_Table_S4_cohort_sensitivity.csv",
        index=False,
    )

    for path in sorted(output_dir.glob("Supplementary_Table_*.csv")):
        print(path)


if __name__ == "__main__":
    main()
