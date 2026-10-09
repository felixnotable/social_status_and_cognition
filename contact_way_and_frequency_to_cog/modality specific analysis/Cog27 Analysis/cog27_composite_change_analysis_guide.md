# Guide to the cog27 Equal-Weight Contact-Change Analysis

## 1. Purpose

This analysis tests whether change in overall contact frequency from T1 to T2 is associated with later cognition at T3, using the Langa-Weir 27-point cognition score (`cog27`) as the outcome.

The analysis is run separately for:

- Children
- Other relatives
- Friends

The main exposure is an equal-weight overall contact score constructed from in-person, phone, and written/email contact.

---

## 2. Cohort timing

### Cohort A
- T1 contact: 2006
- T2 contact: 2010
- T2 cognition adjustment: 2010
- T3 cognition outcome: 2012

### Cohort B
- T1 contact: 2008
- T2 contact: 2012
- T2 cognition adjustment: 2012
- T3 cognition outcome: 2014

The pooled model combines Cohort A and Cohort B and includes cohort as a covariate.

---

## 3. Contact score construction

Each modality uses a 0-5 frequency score, where higher values mean more frequent contact.

For each relationship domain:

`OverallContact_T1 = (InPerson_T1 + Phone_T1 + Written_T1) / 3`

`OverallContact_T2 = (InPerson_T2 + Phone_T2 + Written_T2) / 3`

`ContactChange = OverallContact_T2 - OverallContact_T1`

Interpretation:

- Positive change = overall contact increased
- Zero = no net change
- Negative change = overall contact decreased

---

## 4. Complete-six contact requirement

A participant must have all six contact modality scores available within the relationship domain:

### T1
- in-person
- phone
- written/email

### T2
- in-person
- phone
- written/email

This is the **complete-six** contact sample.

| Relationship | Complete-six N |
|---|---:|
| Children | 7,944 |
| Other relatives | 9,266 |
| Friends | 9,150 |

Eligibility is relationship-specific. A participant can qualify for one domain but not another.

---

## 5. Additional cog27 regression requirements

The regression sample additionally requires:

- `cog27_T2`
- `cog27_T3`
- age at T2
- sex at T2
- education at T2
- race/ethnicity at T2
- wealth at T2
- marital status at T2
- cohort membership for the pooled model

The analysis does **not** require:

- latent `Cog_T1`, `Cog_T2`, or `Cog_T3`
- `cog27_T1`
- T3 contact
- T3 demographic covariates
- loneliness
- loneliness change
- loneliness-contact interactions

Therefore, missing latent cognition does not exclude a participant from this cog27 analysis.

---

## 6. Covariate coding

### Age
Age is centered at 75:

`AgeCentered = Age_T2 - 75`

This only changes the regression reference point. It does not restrict the sample to age 75.

### Sex
Female is the reference group.

### Education
Education is modeled continuously in years.

### Race/ethnicity
White, non-Hispanic is the reference group.

### Wealth
Wealth is transformed as:

`asinh(wealth_T2 / 100000)`

This reduces the effect of extreme wealth values while retaining zero and negative wealth.

### Marital status
Four groups are used:

- Married/partnered — reference
- Separated/divorced
- Widowed
- Never married

### Cohort
In the pooled model:

- Cohort A = reference
- Cohort B is estimated relative to Cohort A

Cohort is not included in the separate A-only or B-only models.

---

## 7. Regression model

For each relationship domain:

`cog27_T3 ~ cog27_T2 + contact_T1 + contact_change + covariates`

The model therefore includes:

- T3 cog27 as the outcome
- T2 cog27 as baseline cognition
- T1 overall contact level
- T1-to-T2 overall contact change
- T2 covariates

The main coefficient of interest is the coefficient for `contact_change`.

OLS regression is used with **HC3 robust standard errors**.

---

## 8. Pooled A+B results

| Relationship | Regression N | Contact-change beta | 95% CI | p-value | BH-adjusted p |
|---|---:|---:|---:|---:|---:|
| Children | 7,083 | +0.182 | +0.082 to +0.282 | 0.000345 | 0.000517 |
| Other relatives | 8,265 | +0.104 | +0.033 to +0.174 | 0.003866 | 0.003866 |
| Friends | 8,164 | +0.145 | +0.072 to +0.219 | 0.000101 | 0.000303 |

All three pooled relationship-domain results are statistically significant.

All three remain significant after BH-FDR correction across the three relationship-domain tests.

---

## 9. Interpretation of the pooled coefficients

The contact-change coefficient represents the expected difference in T3 cog27 associated with a one-point greater T1-to-T2 increase in overall contact, holding the other model variables constant.

### Children
Beta = +0.182

A one-point greater increase in the children contact composite is associated with about 0.18 points higher T3 cog27.

### Other relatives
Beta = +0.104

A one-point greater increase in the other-relative contact composite is associated with about 0.10 points higher T3 cog27.

### Friends
Beta = +0.145

A one-point greater increase in the friends contact composite is associated with about 0.15 points higher T3 cog27.

All three coefficients are positive.

These results are **associational, not causal**.

---

## 10. Cohort-specific results

| Relationship | Cohort | N | Beta | 95% CI | p-value | BH-adjusted p |
|---|---|---:|---:|---:|---:|---:|
| Children | A | 3,771 | +0.187 | +0.051 to +0.323 | 0.0072 | 0.0108 |
| Children | B | 3,312 | +0.181 | +0.033 to +0.328 | 0.0164 | 0.0259 |
| Other relatives | A | 4,411 | +0.121 | +0.022 to +0.220 | 0.0170 | 0.0170 |
| Other relatives | B | 3,854 | +0.085 | -0.015 to +0.185 | 0.0969 | 0.0969 |
| Friends | A | 4,368 | +0.158 | +0.060 to +0.256 | 0.00166 | 0.00498 |
| Friends | B | 3,796 | +0.134 | +0.024 to +0.244 | 0.0173 | 0.0259 |

### Children
The association is positive and significant in both cohorts, with very similar estimates.

### Friends
The association is also positive and significant in both cohorts.

### Other relatives
The estimate is positive in both cohorts, but only Cohort A is statistically significant.

Therefore, children and friends show the clearest cross-cohort replication.

---

## 11. Baseline T1 contact

The T1 contact composite is also positively associated with T3 cog27 in the pooled models:

| Relationship | T1 contact beta | p-value |
|---|---:|---:|
| Children | +0.263 | <0.001 |
| Other relatives | +0.113 | 0.0017 |
| Friends | +0.238 | <0.001 |

This means the model distinguishes:

1. starting contact level at T1
2. change in contact from T1 to T2

The contact-change coefficient therefore reflects change after accounting for baseline contact level.

---

## 12. Sample flow

### Children

| Stage | N |
|---|---:|
| Master cohort | 14,744 |
| Complete six contact measures | 7,944 |
| + cog27 T2 | 7,767 |
| + cog27 T3 | 7,092 |
| + complete T2 covariates | 7,083 |

### Other relatives

| Stage | N |
|---|---:|
| Master cohort | 14,744 |
| Complete six contact measures | 9,266 |
| + cog27 T2 | 9,061 |
| + cog27 T3 | 8,280 |
| + complete T2 covariates | 8,265 |

### Friends

| Stage | N |
|---|---:|
| Master cohort | 14,744 |
| Complete six contact measures | 9,150 |
| + cog27 T2 | 8,952 |
| + cog27 T3 | 8,178 |
| + complete T2 covariates | 8,164 |

Most attrition after complete-six occurs when requiring T3 cog27.

Very few additional participants are lost because of the T2 covariates.

---

## 13. Comparison with latent cognition

| Relationship | cog27 N | cog27 p | Latent Cog N | Latent Cog p |
|---|---:|---:|---:|---:|
| Children | 7,083 | 0.000345 | 5,332 | 0.124 |
| Other relatives | 8,265 | 0.003866 | 5,973 | 0.972 |
| Friends | 8,164 | 0.000101 | 5,883 | 0.0116 |

The cog27 analysis:

- retains substantially more participants
- shows significant pooled associations for all three relationship domains

The latent-Cog analysis:

- has a much smaller sample
- shows a significant association only for friends

Do not directly compare the raw beta values across cog27 and latent Cog because they use different outcome scales.

---

## 14. Main interpretation

The main result is:

> Greater T1-to-T2 increases in overall contact frequency are associated with higher T3 cog27 scores after adjustment for T2 cognition, T1 contact level, demographics, wealth, marital status, and cohort.

The strongest replication is seen for:

1. Children
2. Friends

because both are positive and statistically significant in Cohort A and Cohort B.

Other-relative contact is positive overall, but the cohort-specific result is less consistent.

---

# 15. Explanation of each output file

## `cog27_composite_change_models.csv`

This is the **main results file**.

It contains one row for each combination of:

- Pooled
- Cohort A
- Cohort B

and:

- Children
- Other relatives
- Friends

Important columns:

- `analysis`: Pooled, A, or B
- `relationship`: relationship domain
- `complete6_n`: number meeting the complete-six contact requirement
- `regression_n`: final regression N
- `contact_change_beta`: main contact-change coefficient
- `contact_change_se_HC3`: robust standard error
- `contact_change_ci_low`: lower 95% CI
- `contact_change_ci_high`: upper 95% CI
- `contact_change_p_value`: unadjusted p-value
- `contact_change_p_BH`: BH-FDR adjusted p-value
- `baseline_contact_T1_beta`: coefficient for T1 contact
- `baseline_contact_T1_p_value`: p-value for T1 contact
- `r_squared`
- `adjusted_r_squared`

Use this file for the main scientific interpretation.

---

## `cog27_composite_change_full_coefficients.csv`

This contains every regression coefficient from every model.

It includes coefficients for:

- T2 cog27
- T1 contact
- contact change
- age
- sex
- education
- race/ethnicity
- wealth
- marital status
- cohort where applicable

Important columns:

- `analysis`
- `relationship`
- `term`
- `estimate`
- `se_HC3`
- `ci_low`
- `ci_high`
- `p_value`

Use this file when you want to inspect the complete adjusted model.

---

## `cog27_composite_change_sample_flow.csv`

This documents exactly where participants are lost.

For each pooled/cohort analysis and each relationship domain, it shows N at stages such as:

- master cohort
- complete-six contact
- T2 cog27
- T3 cog27
- complete covariates

Use this file to understand missing-data attrition and final regression eligibility.

---

## `cog27_composite_change_analysis_data.csv`

This is the participant-level dataset after applying the analysis eligibility rules.

Important variables include:

- `hhidpn_key`
- `cohort`
- `analysis`
- `relationship`
- `cog27_T2`
- `cog27_T3`
- `contact_T1`
- `contact_T2`
- `contact_change`
- age
- sex
- education
- race/ethnicity
- wealth
- transformed wealth
- marital status

Because pooled and cohort-specific datasets are both stored, the same participant may appear more than once.

Use this file for auditing observations or reproducing descriptive checks.

---

## `cog27_vs_latent_pooled_comparison.csv`

This directly compares the pooled cog27 and latent-Cog results.

For each relationship it contains:

- cog27 regression N
- cog27 beta
- cog27 CI
- cog27 p
- cog27 BH-adjusted p
- latent-Cog regression N
- latent-Cog beta
- latent-Cog CI
- latent-Cog p

Use this file to compare how cognition measurement affects the findings.

Do not compare the absolute beta magnitude across the two cognition outcomes because their scales differ.

---

## `cog27_composite_change_coefficients_pooled.png`

This plot shows the pooled contact-change estimates and 95% confidence intervals for:

- Children
- Other relatives
- Friends

How to read it:

- dot = estimated beta
- vertical error bar = 95% CI
- horizontal zero line = no association

If the confidence interval does not cross zero, the result is statistically significant at approximately the 0.05 level.

All three pooled confidence intervals are above zero.

---

## `cog27_composite_change_coefficients_by_cohort.png`

This plot shows separate Cohort A and Cohort B estimates for each relationship.

Use it to evaluate replication.

Look for:

- whether A and B have the same direction
- whether the estimates are similar
- whether their confidence intervals overlap
- whether each confidence interval excludes zero

Children and friends show the clearest consistency across cohorts.

---

## `run_composite_change_cognition_cog27.py`

This is the reproducible Python script.

It:

1. reads `contact_cohorts_A_B_no_loneliness_filter.csv`
2. identifies complete-six relationship-specific samples
3. calculates T1 and T2 equal-weight contact composites
4. calculates T1-to-T2 contact change
5. applies cog27 and covariate eligibility
6. runs pooled and Cohort A/B HC3 regressions
7. applies BH-FDR correction
8. exports the model and sample-flow results

Example run:

```bash
python run_composite_change_cognition_cog27.py     contact_cohorts_A_B_no_loneliness_filter.csv     --output-dir composite_change_cognition_cog27
```

---

## `MODEL_SPECIFICATION_COG27.txt`

This is a short plain-text reference describing:

- outcome
- baseline cognition adjustment
- exposure
- covariates
- eligibility
- estimator
- multiple-testing correction

Use it when you need a quick summary of the model specification.

---


## 16. Column-by-column definitions

This section defines the columns in each CSV output.

### A. `cog27_composite_change_models.csv`

| Column | Meaning |
|---|---|
| `analysis` | Analysis stratum. `Pooled` combines Cohort A and Cohort B; `A` and `B` are cohort-specific models. |
| `relationship` | Contact domain: `children`, `other_relatives`, or `friends`. |
| `complete6_n` | Number of participants in that stratum with all six required modality scores: three modalities at T1 and the same three at T2. |
| `regression_n` | Final number of participants used in the regression after additionally requiring cog27_T2, cog27_T3, and all required T2 covariates. |
| `contact_change_beta` | Main regression coefficient for the equal-weight contact-change score, defined as T2 overall contact minus T1 overall contact. Positive values mean greater increases in contact are associated with higher T3 cog27. |
| `contact_change_se_HC3` | HC3 robust standard error for `contact_change_beta`. |
| `contact_change_ci_low` | Lower bound of the 95% confidence interval for the contact-change coefficient. |
| `contact_change_ci_high` | Upper bound of the 95% confidence interval for the contact-change coefficient. |
| `contact_change_p_value` | Unadjusted p-value testing whether the contact-change coefficient equals zero. |
| `baseline_contact_T1_beta` | Regression coefficient for the participant's T1 equal-weight contact level. This adjusts the change effect for where the participant started. |
| `baseline_contact_T1_p_value` | P-value for the T1 contact coefficient. |
| `r_squared` | Proportion of variation in T3 cog27 explained by the complete fitted model. |
| `adjusted_r_squared` | R-squared adjusted for the number of predictors in the model. |
| `contact_change_p_BH` | Benjamini-Hochberg FDR-adjusted p-value for contact change, correcting across the three relationship-domain tests within the same analysis stratum. |

#### How to use this file

For the main research question, focus primarily on:

- `contact_change_beta`
- `contact_change_ci_low`
- `contact_change_ci_high`
- `contact_change_p_value`
- `contact_change_p_BH`
- `regression_n`

Use `analysis = Pooled` for the primary pooled result and `analysis = A` or `B` to examine replication across cohorts.

---

### B. `cog27_composite_change_full_coefficients.csv`

| Column | Meaning |
|---|---|
| `analysis` | `Pooled`, `A`, or `B`. |
| `relationship` | `children`, `other_relatives`, or `friends`. |
| `term` | Exact regression term being estimated. Examples include `cog27_T2`, `contact_T1`, `contact_change`, age, education, wealth, and categorical contrast terms. |
| `estimate` | Regression coefficient for that term. |
| `se_HC3` | HC3 robust standard error for the coefficient. |
| `ci_low` | Lower 95% confidence-interval bound. |
| `ci_high` | Upper 95% confidence-interval bound. |
| `p_value` | P-value testing whether that coefficient equals zero. |

#### Understanding categorical `term` values

Categorical predictors appear as contrasts against the reference category.

For example, a term involving:

`sex_T2 ... [T.Male]`

represents the estimated difference for Male relative to the Female reference group.

Likewise:

- race/ethnicity terms are relative to White, non-Hispanic
- marital-status terms are relative to Married/partnered
- Cohort B is relative to Cohort A in pooled models

This file is useful when the goal is to inspect the entire adjusted regression rather than only the contact-change coefficient.

---

### C. `cog27_composite_change_sample_flow.csv`

| Column | Meaning |
|---|---|
| `analysis` | `Pooled`, `A`, or `B`. |
| `relationship` | Contact domain. |
| `stage` | Eligibility step being applied. |
| `n` | Number of participants remaining after that stage. |

#### Meaning of the `stage` values

| Stage | Meaning |
|---|---|
| `Master cohort rows in stratum` | All participants in the relevant master cohort before relationship-specific complete-case requirements are applied. |
| `Complete six contact measures` | Participants with all three modalities observed at both T1 and T2 for that relationship. |
| `Plus observed cog27 at T2` | Complete-six participants who also have T2 cog27. |
| `Plus observed cog27 at T3` | Participants who additionally have the T3 cog27 outcome. |
| `Plus complete core covariates` | Final regression sample after requiring all T2 adjustment variables. |

This file should be used whenever reporting or auditing sample attrition.

---

### D. `cog27_composite_change_analysis_data.csv`

| Column | Meaning |
|---|---|
| `hhidpn_key` | Unique HRS participant identifier used for merging records across source files. |
| `cohort` | Original cohort membership: `A` or `B`. |
| `analysis` | Indicates whether the row belongs to the `Pooled`, `A`, or `B` analysis dataset. Because pooled and cohort-specific datasets are stacked together, a participant may appear more than once. |
| `relationship` | Relationship domain for this analysis row: children, other relatives, or friends. |
| `cog27_T2` | Langa-Weir 27-point cognition score at T2; used as the baseline cognition adjustment. Higher values indicate better cognition. |
| `cog27_T3` | Langa-Weir 27-point cognition score at T3; this is the regression outcome. Higher values indicate better cognition. |
| `contact_T1` | Equal-weight T1 contact composite: mean of in-person, phone, and written/email frequency scores. Range 0-5. |
| `contact_T2` | Equal-weight T2 contact composite: mean of the same three modality scores. Range 0-5. |
| `contact_change` | `contact_T2 - contact_T1`. Positive values mean contact increased; negative values mean it decreased. |
| `age_T2_centered` | Age at T2 minus 75. For example, age 80 becomes +5 and age 70 becomes -5. |
| `sex_T2` | Sex category at T2. Female is the model reference category. |
| `education_T2` | Completed years of education used as a continuous covariate. |
| `race_ethnicity_T2` | Four-category race/ethnicity variable. White, non-Hispanic is the model reference. |
| `wealth_T2` | Original T2 household wealth value before transformation. |
| `wealth_ihs_T2` | Wealth transformed as `asinh(wealth_T2 / 100000)` for use in the regression. |
| `marital_status_T2` | Original T2 marital-status category from the constructed master dataset. |
| `marital_status_4_T2` | Four-category analysis version of marital status: Married/partnered, Separated/divorced, Widowed, or Never married. Married/partnered is the reference. |

#### Important note about duplicate participants

This file contains stacked analysis datasets.

A participant can appear:

- once in the pooled children model
- once in the Cohort A or B children model
- again for friends
- again for other relatives

Therefore, do not count raw rows in this file as unique participants without filtering by `analysis` and `relationship`.

---

### E. `cog27_vs_latent_pooled_comparison.csv`

| Column | Meaning |
|---|---|
| `relationship` | Children, other relatives, or friends. |
| `cog27_regression_n` | Final sample size in the pooled cog27 model. |
| `cog27_beta` | Contact-change coefficient from the pooled cog27 model. |
| `cog27_ci_low` | Lower 95% CI for the cog27 contact-change coefficient. |
| `cog27_ci_high` | Upper 95% CI for the cog27 contact-change coefficient. |
| `cog27_p` | Unadjusted p-value for the pooled cog27 contact-change coefficient. |
| `cog27_p_BH` | BH-FDR-adjusted cog27 p-value across the three pooled relationship-domain tests. |
| `latent_regression_n` | Final sample size in the pooled latent-Cog model. |
| `latent_beta` | Contact-change coefficient from the latent-Cog model. |
| `latent_ci_low` | Lower 95% CI for the latent-Cog coefficient. |
| `latent_ci_high` | Upper 95% CI for the latent-Cog coefficient. |
| `latent_p` | Unadjusted p-value for the latent-Cog contact-change coefficient. |

#### Important interpretation rule

Do **not** directly compare `cog27_beta` with `latent_beta` as though one were larger or smaller in effect size.

The outcomes use different scales:

- cog27 is a 27-point observed cognition score
- latent Cog is a different continuous latent measure

Use this comparison file mainly to compare:

- sample size
- direction of association
- statistical significance
- confidence-interval evidence
- consistency across relationship domains

---

### F. PNG files

The PNG files do not contain data columns, but their graphical elements correspond to the model-output columns.

#### `cog27_composite_change_coefficients_pooled.png`

- x-axis: relationship domain
- point: `contact_change_beta`
- vertical error bar: `contact_change_ci_low` to `contact_change_ci_high`
- horizontal zero line: no contact-change association

#### `cog27_composite_change_coefficients_by_cohort.png`

- x-axis: relationship domain
- separate estimates: Cohort A and Cohort B
- point: cohort-specific `contact_change_beta`
- error bar: cohort-specific 95% CI
- zero line: no association

Use the cohort plot primarily to assess whether the direction and magnitude replicate across A and B.

---

### G. Python and specification files

#### `run_composite_change_cognition_cog27.py`

This is executable analysis code rather than tabular data. It reproduces the contact-composite construction, sample restrictions, regression models, and BH-FDR correction.

#### `MODEL_SPECIFICATION_COG27.txt`

This is a short human-readable summary of the model conditions. It contains no participant-level data.

#### `composite_change_cognition_cog27_outputs.zip`

This is an archive containing the analysis outputs.

---

## 17. Recommended next step

The next substantive analysis is the modality-specific analysis.

Instead of averaging the three contact modalities together, examine changes in:

- in-person contact
- phone contact
- written/email contact

The main question becomes:

> Which modality or modalities account for the association between overall contact change and later cognition?

Because children and friends replicate across Cohort A and Cohort B in the cog27 analysis, these two relationship domains are especially important to examine.

The modality-specific analysis should distinguish between:

- one modality being individually statistically significant
- actual statistical evidence that modality coefficients differ from one another
