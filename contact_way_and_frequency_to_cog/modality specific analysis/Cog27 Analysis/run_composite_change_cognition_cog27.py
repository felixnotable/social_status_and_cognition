#!/usr/bin/env python3
# Reproduce the cog27 equal-weight contact-change models.
# Model: cog27_T3 ~ cog27_T2 + T1 contact + T1->T2 contact change + core T2 covariates.
# Eligibility: complete six T1/T2 contact modalities within relationship, cog27_T2/T3,
# and complete T2 core covariates. Pooled A+B and separate Cohort A/B models are fit.
# HC3 robust standard errors; BH-FDR across the three relationship domains per stratum.

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

RELATIONSHIPS=["children","other_relatives","friends"]
MODALITIES=["inperson","phone","written_email"]
MARITAL_MAP={
    "Married":"Married/partnered","Married, spouse absent":"Married/partnered",
    "Partnered":"Married/partnered","Separated":"Separated/divorced",
    "Divorced":"Separated/divorced","Separated/Divorced":"Separated/divorced",
    "Widowed":"Widowed","Never married":"Never married"
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input_csv",type=Path)
    ap.add_argument("--output-dir",type=Path,default=Path("composite_change_cognition_cog27"))
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.input_csv,low_memory=False)
    df["age_T2_centered"]=pd.to_numeric(df["age_T2"],errors="coerce")-75
    df["wealth_ihs_T2"]=np.arcsinh(pd.to_numeric(df["wealth_T2"],errors="coerce")/100000.0)
    df["marital_status_4_T2"]=df["marital_status_T2"].map(MARITAL_MAP)

    rows=[]; flows=[]
    for rel in RELATIONSHIPS:
        for stratum in ["Pooled","A","B"]:
            b=df.copy() if stratum=="Pooled" else df[df.cohort.eq(stratum)].copy()
            six=[f"{rel}_{m}_freqscore_{tp}" for tp in ["T1","T2"] for m in MODALITIES]
            d=b[b[six].notna().all(axis=1)].copy()
            flows.append({"analysis":stratum,"relationship":rel,"stage":"Complete six contact measures","n":len(d)})
            d["contact_T1"]=d[[f"{rel}_{m}_freqscore_T1" for m in MODALITIES]].mean(axis=1)
            d["contact_T2"]=d[[f"{rel}_{m}_freqscore_T2" for m in MODALITIES]].mean(axis=1)
            d["contact_change"]=d.contact_T2-d.contact_T1
            d=d.dropna(subset=["cog27_T2"]); flows.append({"analysis":stratum,"relationship":rel,"stage":"Plus observed cog27 at T2","n":len(d)})
            d=d.dropna(subset=["cog27_T3"]); flows.append({"analysis":stratum,"relationship":rel,"stage":"Plus observed cog27 at T3","n":len(d)})
            req=["age_T2_centered","sex_T2","education_T2","race_ethnicity_T2","wealth_ihs_T2","marital_status_4_T2"]
            if stratum=="Pooled": req.append("cohort")
            d=d.dropna(subset=req); flows.append({"analysis":stratum,"relationship":rel,"stage":"Plus complete core covariates","n":len(d)})
            formula=('cog27_T3 ~ cog27_T2 + contact_T1 + contact_change + age_T2_centered'
                     ' + C(sex_T2, Treatment(reference="Female")) + education_T2'
                     ' + C(race_ethnicity_T2, Treatment(reference="White, non-Hispanic"))'
                     ' + wealth_ihs_T2'
                     ' + C(marital_status_4_T2, Treatment(reference="Married/partnered"))')
            if stratum=="Pooled": formula+=' + C(cohort, Treatment(reference="A"))'
            m=smf.ols(formula,data=d).fit(cov_type="HC3")
            ci=m.conf_int().loc["contact_change"]
            rows.append({"analysis":stratum,"relationship":rel,"regression_n":int(m.nobs),
                         "contact_change_beta":m.params["contact_change"],
                         "contact_change_se_HC3":m.bse["contact_change"],
                         "contact_change_ci_low":ci.iloc[0],"contact_change_ci_high":ci.iloc[1],
                         "contact_change_p_value":m.pvalues["contact_change"],
                         "baseline_contact_T1_beta":m.params["contact_T1"],
                         "baseline_contact_T1_p_value":m.pvalues["contact_T1"],
                         "r_squared":m.rsquared})
    out=pd.DataFrame(rows); out["contact_change_p_BH"]=np.nan
    for s in ["Pooled","A","B"]:
        idx=out.analysis.eq(s)
        out.loc[idx,"contact_change_p_BH"]=multipletests(out.loc[idx,"contact_change_p_value"],method="fdr_bh")[1]
    out.to_csv(args.output_dir/"cog27_composite_change_models.csv",index=False)
    pd.DataFrame(flows).to_csv(args.output_dir/"cog27_composite_change_sample_flow.csv",index=False)

if __name__=="__main__":
    main()
