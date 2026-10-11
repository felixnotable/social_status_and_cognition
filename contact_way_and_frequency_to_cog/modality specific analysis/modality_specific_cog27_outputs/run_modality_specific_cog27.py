#!/usr/bin/env python3
"""
Modality-specific cog27 analysis for HRS contact data.

Input
-----
contact_cohorts_A_B_no_loneliness_filter.csv

Eligibility within each relationship domain
-------------------------------------------
1. All three contact modalities observed at T1:
   in-person, phone, written/email
2. All three modalities observed at T2
3. cog27_T2 observed
4. cog27_T3 observed
5. Complete T2 core covariates

This ensures all three modalities are compared on the same people.

Models
------
Separate modality models:
    cog27_T3 ~ cog27_T2 + modality_T1 + modality_change + core covariates

Joint modality model:
    cog27_T3 ~ cog27_T2
               + inperson_T1 + phone_T1 + written_email_T1
               + inperson_change + phone_change + written_email_change
               + core covariates

Analyses
--------
- Pooled Cohort A+B, with Cohort A as reference
- Cohort A separately
- Cohort B separately
- HC3 robust standard errors
- BH-FDR across 3 modality change tests within each relationship/stratum
- Pairwise coefficient-difference tests from the joint model
- Omnibus equality test: Δ in-person = Δ phone = Δ written/email
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

RELATIONSHIPS = ["children", "other_relatives", "friends"]
MODALITIES = ["inperson", "phone", "written_email"]

MARITAL_MAP = {
    "Married":"Married/partnered",
    "Married, spouse absent":"Married/partnered",
    "Partnered":"Married/partnered",
    "Separated":"Separated/divorced",
    "Divorced":"Separated/divorced",
    "Separated/Divorced":"Separated/divorced",
    "Widowed":"Widowed",
    "Never married":"Never married",
}

def formula(extra_terms, pooled):
    f = (
        "cog27_T3 ~ cog27_T2"
        + extra_terms
        + " + age_T2_centered"
        + ' + C(sex_T2, Treatment(reference="Female"))'
        + " + education_T2"
        + ' + C(race_ethnicity_T2, Treatment(reference="White, non-Hispanic"))'
        + " + wealth_ihs_T2"
        + ' + C(marital_status_4_T2, Treatment(reference="Married/partnered"))'
    )
    if pooled:
        f += ' + C(cohort, Treatment(reference="A"))'
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("modality_specific_cog27"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    df["age_T2_centered"] = pd.to_numeric(df["age_T2"], errors="coerce") - 75
    df["wealth_ihs_T2"] = np.arcsinh(pd.to_numeric(df["wealth_T2"], errors="coerce") / 100000.0)
    df["marital_status_4_T2"] = df["marital_status_T2"].map(MARITAL_MAP)

    separate_rows, joint_rows, test_rows, flow_rows = [], [], [], []

    for rel in RELATIONSHIPS:
        for stratum in ["Pooled","A","B"]:
            base = df.copy() if stratum == "Pooled" else df[df["cohort"].eq(stratum)].copy()
            flow_rows.append({"analysis":stratum,"relationship":rel,
                              "stage":"Master cohort rows in stratum","n":len(base)})

            six = [f"{rel}_{m}_freqscore_{tp}" for tp in ["T1","T2"] for m in MODALITIES]
            d = base[base[six].notna().all(axis=1)].copy()
            flow_rows.append({"analysis":stratum,"relationship":rel,
                              "stage":"Complete six contact measures","n":len(d)})

            d = d.dropna(subset=["cog27_T2"]).copy()
            flow_rows.append({"analysis":stratum,"relationship":rel,
                              "stage":"Plus observed cog27 at T2","n":len(d)})
            d = d.dropna(subset=["cog27_T3"]).copy()
            flow_rows.append({"analysis":stratum,"relationship":rel,
                              "stage":"Plus observed cog27 at T3","n":len(d)})

            req = ["age_T2_centered","sex_T2","education_T2","race_ethnicity_T2",
                   "wealth_ihs_T2","marital_status_4_T2"]
            pooled = stratum == "Pooled"
            if pooled:
                req.append("cohort")
            d = d.dropna(subset=req).copy()
            flow_rows.append({"analysis":stratum,"relationship":rel,
                              "stage":"Plus complete core covariates","n":len(d)})

            for m in MODALITIES:
                d[f"{m}_T1"] = d[f"{rel}_{m}_freqscore_T1"]
                d[f"{m}_T2"] = d[f"{rel}_{m}_freqscore_T2"]
                d[f"{m}_change"] = d[f"{m}_T2"] - d[f"{m}_T1"]

            # Separate models
            for m in MODALITIES:
                fit = smf.ols(
                    formula(f" + {m}_T1 + {m}_change", pooled), data=d
                ).fit(cov_type="HC3")
                ci = fit.conf_int()
                separate_rows.append({
                    "analysis":stratum,"relationship":rel,"modality":m,
                    "regression_n":int(fit.nobs),
                    "change_beta":fit.params[f"{m}_change"],
                    "change_se_HC3":fit.bse[f"{m}_change"],
                    "change_ci_low":ci.loc[f"{m}_change",0],
                    "change_ci_high":ci.loc[f"{m}_change",1],
                    "change_p_value":fit.pvalues[f"{m}_change"],
                    "baseline_T1_beta":fit.params[f"{m}_T1"],
                    "baseline_T1_p_value":fit.pvalues[f"{m}_T1"],
                    "r_squared":fit.rsquared,
                    "adjusted_r_squared":fit.rsquared_adj,
                })

            # Joint model
            extra = "".join(
                [f" + {m}_T1" for m in MODALITIES]
                + [f" + {m}_change" for m in MODALITIES]
            )
            jfit = smf.ols(formula(extra, pooled), data=d).fit(cov_type="HC3")
            ci = jfit.conf_int()

            for m in MODALITIES:
                joint_rows.append({
                    "analysis":stratum,"relationship":rel,"modality":m,
                    "regression_n":int(jfit.nobs),
                    "change_beta":jfit.params[f"{m}_change"],
                    "change_se_HC3":jfit.bse[f"{m}_change"],
                    "change_ci_low":ci.loc[f"{m}_change",0],
                    "change_ci_high":ci.loc[f"{m}_change",1],
                    "change_p_value":jfit.pvalues[f"{m}_change"],
                    "baseline_T1_beta":jfit.params[f"{m}_T1"],
                    "baseline_T1_p_value":jfit.pvalues[f"{m}_T1"],
                    "r_squared":jfit.rsquared,
                    "adjusted_r_squared":jfit.rsquared_adj,
                })

            # Pairwise coefficient differences
            pairs = [("inperson","phone"),("inperson","written_email"),("phone","written_email")]
            pair_rows, pair_ps = [], []
            for m1,m2 in pairs:
                t = jfit.t_test(f"{m1}_change = {m2}_change")
                ci2 = np.asarray(t.conf_int()).squeeze()
                row = {
                    "analysis":stratum,"relationship":rel,
                    "test_type":"pairwise_difference",
                    "comparison":f"{m1}_minus_{m2}",
                    "estimate_difference":float(np.asarray(t.effect).squeeze()),
                    "se_HC3":float(np.asarray(t.sd).squeeze()),
                    "ci_low":float(ci2[0]),"ci_high":float(ci2[1]),
                    "statistic":float(np.asarray(t.tvalue).squeeze()),
                    "df_num":1,
                    "p_value":float(np.asarray(t.pvalue).squeeze()),
                }
                pair_rows.append(row); pair_ps.append(row["p_value"])
            q = multipletests(pair_ps, method="fdr_bh")[1]
            for row,qq in zip(pair_rows,q):
                row["p_BH_within_relationship"] = qq
                test_rows.append(row)

            # Omnibus equality test
            w = jfit.wald_test(
                ["inperson_change = phone_change",
                 "inperson_change = written_email_change"],
                use_f=True, scalar=True
            )
            test_rows.append({
                "analysis":stratum,"relationship":rel,
                "test_type":"omnibus_equal_changes",
                "comparison":"inperson = phone = written_email",
                "estimate_difference":np.nan,"se_HC3":np.nan,
                "ci_low":np.nan,"ci_high":np.nan,
                "statistic":float(w.statistic),"df_num":int(w.df_num),
                "p_value":float(w.pvalue),
                "p_BH_within_relationship":np.nan,
            })

    separate = pd.DataFrame(separate_rows)
    joint = pd.DataFrame(joint_rows)
    tests = pd.DataFrame(test_rows)

    for tab in [separate,joint]:
        tab["change_p_BH_within_relationship"] = np.nan
        for _, idx in tab.groupby(["analysis","relationship"]).groups.items():
            tab.loc[idx,"change_p_BH_within_relationship"] = multipletests(
                tab.loc[idx,"change_p_value"], method="fdr_bh"
            )[1]

    separate.to_csv(args.output_dir/"modality_separate_models.csv", index=False)
    joint.to_csv(args.output_dir/"modality_joint_model.csv", index=False)
    tests.to_csv(args.output_dir/"modality_difference_tests.csv", index=False)
    pd.DataFrame(flow_rows).to_csv(args.output_dir/"modality_sample_flow.csv", index=False)

if __name__ == "__main__":
    main()
