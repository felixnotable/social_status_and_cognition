#!/usr/bin/env python3
"""
Primary pooled latent-cognition analysis for equal-weight contact change.

For each relationship domain separately:
1. Require all six T1/T2 modality scores to be nonmissing.
2. Compute equal-weight T1 and T2 contact composites.
3. Compute change = T2 - T1.
4. Require observed T2/T3 latent cognition and complete core covariates.
5. Fit pooled A+B OLS with HC3 robust standard errors.

Model:
Cog_T3 ~ Cog_T2 + contact_T1 + contact_change
         + age_T2_centered
         + sex_T2
         + education_T2
         + race_ethnicity_T2
         + asinh(wealth_T2 / 100000)
         + marital_status_4_T2
         + cohort

Reference categories:
- sex: Female
- race/ethnicity: White, non-Hispanic
- marital status: Married/partnered
- cohort: A
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

RELATIONSHIPS = ["children", "other_relatives", "friends"]
MODALITIES = ["inperson", "phone", "written_email"]
LABELS = {
    "children": "Children",
    "other_relatives": "Other relatives",
    "friends": "Friends",
}
MARITAL_MAP = {
    "Married": "Married/partnered",
    "Married, spouse absent": "Married/partnered",
    "Partnered": "Married/partnered",
    "Separated": "Separated/divorced",
    "Divorced": "Separated/divorced",
    "Separated/Divorced": "Separated/divorced",
    "Widowed": "Widowed",
    "Never married": "Never married",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("composite_change_cognition_latent"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False).copy()
    df["age_T2_centered"] = pd.to_numeric(df["age_T2"], errors="coerce") - 75.0
    df["wealth_ihs_T2"] = np.arcsinh(
        pd.to_numeric(df["wealth_T2"], errors="coerce") / 100000.0
    )
    df["marital_status_4_T2"] = df["marital_status_T2"].map(MARITAL_MAP)

    results, coefs, flows, analysis_parts = [], [], [], []

    for rel in RELATIONSHIPS:
        six = [
            f"{rel}_{m}_freqscore_{tp}"
            for tp in ["T1", "T2"]
            for m in MODALITIES
        ]
        mask = df[six].notna().all(axis=1)
        d = df.loc[mask].copy()

        d["contact_T1"] = d[
            [f"{rel}_{m}_freqscore_T1" for m in MODALITIES]
        ].mean(axis=1)
        d["contact_T2"] = d[
            [f"{rel}_{m}_freqscore_T2" for m in MODALITIES]
        ].mean(axis=1)
        d["contact_change"] = d["contact_T2"] - d["contact_T1"]

        flows.append({"relationship": rel, "stage": "Complete six contact measures", "n": len(d)})
        d = d.dropna(subset=["Cog_T2"])
        flows.append({"relationship": rel, "stage": "Plus observed latent cognition at T2", "n": len(d)})
        d = d.dropna(subset=["Cog_T3"])
        flows.append({"relationship": rel, "stage": "Plus observed latent cognition at T3", "n": len(d)})

        req = [
            "age_T2_centered", "sex_T2", "education_T2",
            "race_ethnicity_T2", "wealth_ihs_T2",
            "marital_status_4_T2", "cohort",
        ]
        d = d.dropna(subset=req).copy()
        flows.append({"relationship": rel, "stage": "Plus complete core covariates", "n": len(d)})

        formula = (
            'Cog_T3 ~ Cog_T2 + contact_T1 + contact_change'
            ' + age_T2_centered'
            ' + C(sex_T2, Treatment(reference="Female"))'
            ' + education_T2'
            ' + C(race_ethnicity_T2, Treatment(reference="White, non-Hispanic"))'
            ' + wealth_ihs_T2'
            ' + C(marital_status_4_T2, Treatment(reference="Married/partnered"))'
            ' + C(cohort, Treatment(reference="A"))'
        )
        m = smf.ols(formula, data=d).fit(cov_type="HC3")
        ci = m.conf_int()

        results.append({
            "relationship": rel,
            "complete6_n": int(mask.sum()),
            "regression_n": int(m.nobs),
            "contact_change_beta": m.params["contact_change"],
            "contact_change_se_HC3": m.bse["contact_change"],
            "contact_change_ci_low": ci.loc["contact_change", 0],
            "contact_change_ci_high": ci.loc["contact_change", 1],
            "contact_change_p_value": m.pvalues["contact_change"],
            "baseline_contact_T1_beta": m.params["contact_T1"],
            "baseline_contact_T1_p_value": m.pvalues["contact_T1"],
            "r_squared": m.rsquared,
            "adjusted_r_squared": m.rsquared_adj,
        })

        for term in m.params.index:
            coefs.append({
                "relationship": rel,
                "term": term,
                "estimate": m.params[term],
                "se_HC3": m.bse[term],
                "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1],
                "p_value": m.pvalues[term],
            })

    results_df = pd.DataFrame(results)
    pd.DataFrame(coefs).to_csv(
        args.output_dir/"composite_change_cognition_full_coefficients.csv", index=False
    )
    pd.DataFrame(flows).to_csv(
        args.output_dir/"composite_change_cognition_sample_flow.csv", index=False
    )
    results_df.to_csv(
        args.output_dir/"composite_change_cognition_models.csv", index=False
    )

    x = np.arange(len(results_df))
    y = results_df["contact_change_beta"].to_numpy()
    lo = y - results_df["contact_change_ci_low"].to_numpy()
    hi = results_df["contact_change_ci_high"].to_numpy() - y
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o", capsize=5)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in results_df["relationship"]])
    ax.set_ylabel("Change in T3 latent cognition per 1-point contact-change score")
    ax.set_title("Adjusted association of contact-frequency change with T3 cognition")
    ax.grid(True, axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(
        args.output_dir/"composite_change_cognition_coefficients.png",
        dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

if __name__ == "__main__":
    main()
