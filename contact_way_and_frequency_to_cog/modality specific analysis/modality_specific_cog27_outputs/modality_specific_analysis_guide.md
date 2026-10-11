# Modality-Specific cog27 Analysis Guide
## Purpose
This analysis decomposes the previously significant equal-weight contact composite into **in-person, phone, and written/email contact**. It asks whether change in each modality from T1 to T2 is associated with T3 cognition, and whether the modality coefficients are statistically different from one another.
## Analysis sample and conditions
- Relationship-specific complete-six sample: all three modalities observed at both T1 and T2.
- `cog27_T2` and `cog27_T3` required.
- Complete T2 age, sex, education, race/ethnicity, wealth, and marital-status covariates required.
- Pooled A+B model includes cohort; A-only and B-only models are also run.
- HC3 robust standard errors.
- Every modality within a given relationship/stratum uses the **same participants**, preventing sample differences from driving modality comparisons.
### Separate modality model
`cog27_T3 ~ cog27_T2 + modality_T1 + modality_change + covariates`
### Joint modality model
`cog27_T3 ~ cog27_T2 + IP_T1 + Phone_T1 + Written_T1 + ΔIP + ΔPhone + ΔWritten + covariates`
The joint model estimates each modality-change association while adjusting for the other two modality changes and all three baseline modality levels.
## Pooled A+B: separate modality models
| Relationship | Modality | N | Beta | 95% CI | p | BH p |
|---|---|---:|---:|---:|---:|---:|
| Children | In-person | 7,083 | +0.084 | +0.017 to +0.150 | 0.01343 | 0.01343 |
| Children | Phone | 7,083 | +0.119 | +0.030 to +0.208 | 0.008912 | 0.01343 |
| Children | Written/email | 7,083 | +0.072 | +0.015 to +0.129 | 0.01268 | 0.01343 |
| Other relatives | In-person | 8,265 | +0.047 | -0.005 to +0.100 | 0.07624 | 0.1144 |
| Other relatives | Phone | 8,265 | +0.037 | -0.017 to +0.091 | 0.1788 | 0.1788 |
| Other relatives | Written/email | 8,265 | +0.092 | +0.038 to +0.147 | 0.0009088 | 0.002726 |
| Friends | In-person | 8,164 | +0.053 | -0.002 to +0.107 | 0.05982 | 0.05982 |
| Friends | Phone | 8,164 | +0.115 | +0.058 to +0.172 | 7.034e-05 | 0.000211 |
| Friends | Written/email | 8,164 | +0.089 | +0.036 to +0.141 | 0.001059 | 0.001588 |

## Pooled A+B: joint modality model
| Relationship | Modality | N | Joint beta | 95% CI | p | BH p |
|---|---|---:|---:|---:|---:|---:|
| Children | In-person | 7,083 | +0.063 | -0.008 to +0.135 | 0.08347 | 0.1252 |
| Children | Phone | 7,083 | +0.062 | -0.036 to +0.160 | 0.2128 | 0.2128 |
| Children | Written/email | 7,083 | +0.061 | +0.004 to +0.119 | 0.0361 | 0.1083 |
| Other relatives | In-person | 8,265 | +0.044 | -0.022 to +0.110 | 0.1948 | 0.2922 |
| Other relatives | Phone | 8,265 | -0.018 | -0.089 to +0.052 | 0.6048 | 0.6048 |
| Other relatives | Written/email | 8,265 | +0.090 | +0.033 to +0.147 | 0.002108 | 0.006324 |
| Friends | In-person | 8,164 | -0.031 | -0.099 to +0.037 | 0.3665 | 0.3665 |
| Friends | Phone | 8,164 | +0.105 | +0.034 to +0.175 | 0.003776 | 0.01133 |
| Friends | Written/email | 8,164 | +0.069 | +0.014 to +0.123 | 0.01401 | 0.02102 |

## Formal modality-difference tests: pooled A+B
The omnibus test asks whether all three change coefficients are equal. Pairwise tests compare two change coefficients directly. A significant individual modality and a nonsignificant modality do **not** by themselves prove that the two coefficients differ; these tests answer that question.

### Children
- Omnibus equality test: p = **0.9991**.
- in-person vs phone: difference = +0.001, 95% CI -0.140 to +0.142, p=0.9872, BH p=0.989.
- in-person vs written/email: difference = +0.002, 95% CI -0.092 to +0.096, p=0.9668, BH p=0.989.
- phone vs written/email: difference = +0.001, 95% CI -0.119 to +0.121, p=0.989, BH p=0.989.

### Other relatives
- Omnibus equality test: p = **0.08535**.
- in-person vs phone: difference = +0.062, 95% CI -0.059 to +0.183, p=0.3145, BH p=0.3145.
- in-person vs written/email: difference = -0.046, 95% CI -0.135 to +0.042, p=0.3061, BH p=0.3145.
- phone vs written/email: difference = -0.108, 95% CI -0.209 to -0.008, p=0.03385, BH p=0.1016.

### Friends
- Omnibus equality test: p = **0.05746**.
- in-person vs phone: difference = -0.136, 95% CI -0.259 to -0.013, p=0.03001, BH p=0.0467.
- in-person vs written/email: difference = -0.100, 95% CI -0.191 to -0.009, p=0.03114, BH p=0.0467.
- phone vs written/email: difference = +0.036, 95% CI -0.060 to +0.132, p=0.4597, BH p=0.4597.

## Cohort A/B replication: separate modality models
| Relationship | Modality | Cohort A beta (p) | Cohort B beta (p) |
|---|---|---:|---:|
| Children | In-person | +0.043 (0.3629) | +0.130 (0.008114) |
| Children | Phone | +0.103 (0.09842) | +0.136 (0.04542) |
| Children | Written/email | +0.115 (0.003227) | +0.028 (0.5175) |
| Other relatives | In-person | +0.064 (0.08423) | +0.026 (0.4975) |
| Other relatives | Phone | +0.002 (0.9568) | +0.071 (0.07378) |
| Other relatives | Written/email | +0.132 (0.0004226) | +0.050 (0.2336) |
| Friends | In-person | +0.031 (0.413) | +0.076 (0.06312) |
| Friends | Phone | +0.106 (0.005989) | +0.122 (0.005313) |
| Friends | Written/email | +0.122 (0.0006536) | +0.053 (0.198) |

## Output files
- `modality_separate_models.csv`: change and baseline coefficients from the nine separate modality models in Pooled/A/B strata.
- `modality_joint_model.csv`: coefficients from joint models containing all three modality baselines and changes simultaneously.
- `modality_difference_tests.csv`: omnibus equality tests and pairwise differences among the three change coefficients.
- `modality_results_summary.csv`: side-by-side separate vs joint change results for quick comparison.
- `modality_sample_flow.csv`: N retained at each eligibility stage.
- `modality_analysis_data.csv`: participant-level final analysis data, stacked by relationship and analysis stratum.
- `modality_separate_models_pooled.png`: pooled separate-model coefficient plot.
- `modality_joint_model_pooled.png`: pooled joint-model coefficient plot.
- `modality_separate_models_by_cohort.png`: Cohort A/B replication plot.
- `run_modality_specific_cog27.py`: analysis script reference.
- `modality_specific_analysis_guide.md`: this guide.



## Interpretation of the main findings

### Children

The pooled **separate** models are positive for all three modalities:

- in-person: beta = +0.084, p = .013
- phone: beta = +0.119, p = .0089
- written/email: beta = +0.072, p = .0127

All three survive BH correction within the children domain.

However, the cohort-specific pattern is not the same for each modality. Written/email is strongest in Cohort A, whereas in-person is strongest in Cohort B. In the **joint model**, the three change coefficients become almost identical (about +0.06 each), and the omnibus equality test is p = .999.

**Interpretation:** the children composite association appears to be broadly distributed across contact modes rather than driven by one clearly superior modality. There is no evidence that the three children modality-change coefficients differ.

### Other relatives

In the pooled separate models, written/email is the clear signal:

- in-person: p = .076
- phone: p = .179
- written/email: beta = +0.092, p = .00091, BH p = .00273

Written/email remains significant in the pooled joint model:

- beta = +0.090
- p = .00211
- BH p = .00632

However, this is driven mainly by Cohort A. Cohort B does not independently replicate the written/email association. The omnibus equality test is p = .085, and the phone-versus-written pairwise difference is p = .034 before correction but BH p = .102.

**Interpretation:** written/email is the most promising other-relative modality, but the cross-cohort evidence is not strong enough to conclude that it is definitively different from the other modes.

### Friends

This is the clearest modality-specific result.

Pooled separate models:

- in-person: beta = +0.053, p = .060
- phone: beta = +0.115, p = .000070, BH p = .000211
- written/email: beta = +0.089, p = .00106, BH p = .00159

Pooled joint model:

- in-person: beta = -0.031, p = .366
- phone: beta = +0.105, p = .00378, BH p = .0113
- written/email: beta = +0.069, p = .0140, BH p = .0210

Phone change also replicates in the **separate models** in both cohorts:

- Cohort A: beta = +0.106, p = .00599, BH p = .00898
- Cohort B: beta = +0.122, p = .00531, BH p = .0159

Written/email is significant in Cohort A but not Cohort B. In-person is not significant after BH correction in either cohort.

In the pooled joint model, formal pairwise comparisons show:

- in-person vs phone: BH p = .0467
- in-person vs written/email: BH p = .0467
- phone vs written/email: BH p = .460

The overall three-way equality test is borderline rather than conventionally significant:

- omnibus p = .0575

**Interpretation:** friend phone contact is the strongest and most consistently replicated modality-specific signal. The pooled joint model also suggests that phone and written/email changes are more positively associated with later cognition than in-person change. However, because the overall omnibus test is p = .0575, the evidence for a general difference among all three modalities should be described as suggestive rather than definitive.

## Important statistical interpretation

A modality being significant while another is nonsignificant does **not** prove that the two modality effects differ.

The formal joint-model coefficient-difference tests are the appropriate evidence for claims such as "phone is stronger than in-person."

This distinction is especially important for the manuscript.

## Column definitions

### `modality_separate_models.csv`

| Column | Meaning |
|---|---|
| `analysis` | `Pooled`, `A`, or `B`. |
| `relationship` | `children`, `other_relatives`, or `friends`. |
| `modality` | `inperson`, `phone`, or `written_email`. |
| `regression_n` | Number of participants in the model. Within a relationship/stratum this is identical across all three modalities. |
| `change_beta` | Coefficient for T1-to-T2 change in that modality. Positive values mean increasing/more-maintained contact is associated with higher T3 cog27. |
| `change_se_HC3` | HC3 robust standard error of the change coefficient. |
| `change_ci_low` | Lower 95% CI for the change coefficient. |
| `change_ci_high` | Upper 95% CI for the change coefficient. |
| `change_p_value` | Unadjusted p-value for the change coefficient. |
| `baseline_T1_beta` | Coefficient for that modality's T1 contact-frequency score. |
| `baseline_T1_p_value` | P-value for the T1 modality coefficient. |
| `r_squared` | R-squared for the full regression model. |
| `adjusted_r_squared` | R-squared adjusted for the number of model parameters. |
| `change_p_BH_within_relationship` | BH-FDR p-value correcting the three modality-change tests within that relationship and analysis stratum. |

### `modality_joint_model.csv`

The columns have the same meanings as `modality_separate_models.csv`, but the coefficients come from a **single model containing all three baseline modality scores and all three change scores simultaneously**.

Therefore, `change_beta` in this file represents the association for one modality change **holding changes in the other two modalities constant**.

### `modality_difference_tests.csv`

| Column | Meaning |
|---|---|
| `analysis` | `Pooled`, `A`, or `B`. |
| `relationship` | Relationship domain. |
| `test_type` | `pairwise_difference` or `omnibus_equal_changes`. |
| `comparison` | Which change coefficients are being compared. |
| `estimate_difference` | Difference between two change coefficients for pairwise tests. Blank for the omnibus test. |
| `se_HC3` | Robust SE of the coefficient difference. |
| `ci_low` | Lower 95% CI for a pairwise coefficient difference. |
| `ci_high` | Upper 95% CI for a pairwise coefficient difference. |
| `statistic` | Test statistic. Pairwise rows use a t-type statistic; omnibus rows use an F/Wald statistic. |
| `df_num` | Numerator degrees of freedom for the tested restriction. |
| `p_value` | Unadjusted p-value. |
| `p_BH_within_relationship` | BH-FDR p-value across the three pairwise comparisons within the relationship. Blank for the omnibus test. |

For a pairwise row such as `inperson_minus_phone`, a negative estimate means the in-person change coefficient is smaller than the phone change coefficient.

### `modality_results_summary.csv`

This file places the separate-model and joint-model results side by side.

| Column group | Meaning |
|---|---|
| `analysis`, `relationship`, `modality`, `n` | Model identification and sample size. |
| `separate_beta`, `separate_ci_low`, `separate_ci_high`, `separate_p`, `separate_p_BH` | Results when that modality is modeled separately. |
| `joint_beta`, `joint_ci_low`, `joint_ci_high`, `joint_p`, `joint_p_BH` | Results when all three modalities are modeled together. |

This is the easiest CSV for quickly seeing how adjustment for the other modalities changes a result.

### `modality_sample_flow.csv`

| Column | Meaning |
|---|---|
| `analysis` | `Pooled`, `A`, or `B`. |
| `relationship` | Relationship domain. |
| `stage` | Eligibility step. |
| `n` | Number remaining after that step. |

The final N is intentionally identical to the preceding cog27 composite-change analysis for the same relationship and stratum.

### `modality_analysis_data.csv`

| Column | Meaning |
|---|---|
| `hhidpn_key` | Unique HRS participant identifier. |
| `cohort` | Cohort A or B. |
| `analysis` | Pooled, A, or B analysis dataset. |
| `relationship` | Relationship domain. |
| `cog27_T2` | Baseline cognition adjustment. |
| `cog27_T3` | Cognitive outcome. |
| `age_T2_centered` | T2 age minus 75. |
| `sex_T2` | Sex covariate; Female is reference. |
| `education_T2` | Completed years of education. |
| `race_ethnicity_T2` | Race/ethnicity covariate; White non-Hispanic is reference. |
| `wealth_T2` | Original T2 wealth. |
| `wealth_ihs_T2` | `asinh(wealth_T2 / 100000)`. |
| `marital_status_T2` | Original marital-status category. |
| `marital_status_4_T2` | Four-category model version of marital status. |
| `inperson_T1`, `phone_T1`, `written_email_T1` | T1 frequency scores, 0-5. |
| `inperson_T2`, `phone_T2`, `written_email_T2` | T2 frequency scores, 0-5. |
| `inperson_change`, `phone_change`, `written_email_change` | T2 minus T1 modality-specific change scores. |

Because pooled and cohort-specific data are stacked, one participant may occur more than once in this file.

## Recommended manuscript-level conclusion from this stage

The data do **not** support a simple claim that one modality is universally superior.

Instead:

- **Children:** overall contact-change association is spread across modalities; no evidence of modality differences.
- **Friends:** phone is the most robust replicated modality, with written/email also contributing in pooled models; in-person is weakest.
- **Other relatives:** written/email is the strongest pooled modality, but replication across cohorts is limited.

The friend result is the strongest basis for a modality-specific scientific claim.
