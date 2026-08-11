"""
Step 3 — Measurement feasibility for the proposed longitudinal analysis

Purpose
-------
Starting from participants with valid/self-completed loneliness at both T1
and T2, quantify:

1. Availability of:
   - weekly relatives/friends contact
   - weekly child contact
   - living alone
2. T1->T2 transition distributions.
3. Remaining sample after requiring T2 and T3 cognition.
4. Loneliness change distribution:
   - decreased
   - stable
   - increased
5. Cross-tabs of the three loneliness directions with the four contact
   trajectories.

IMPORTANT:
This script is a feasibility/sample-size audit only. It does NOT test whether
any social variable predicts cognition and does NOT choose a moderator based
on outcome significance.

Required inputs
---------------
- H_HRS_d_stata.zip
- HRS_participant_level_cohorts_updated.csv
- HRS_child_contact_correct_T1_T2.csv

Writes:
- step3_measurement_feasibility_summary.csv
- step3_loneliness_by_contact_trajectory.csv
"""

from pathlib import Path
import zipfile
import tempfile
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------
H_HRS_ZIP = Path("H_HRS_d_stata.zip")
MASTER_CSV = Path("HRS_participant_level_cohorts_updated.csv")
CHILD_CSV = Path("HRS_child_contact_correct_T1_T2.csv")

OUT_SUMMARY = Path("step3_measurement_feasibility_summary.csv")
OUT_CROSSTAB = Path("step3_loneliness_by_contact_trajectory.csv")

COHORT_MAP = {
    "A": {"T1_wave": 8, "T2_wave": 10},
    "B": {"T1_wave": 9, "T2_wave": 11},
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_hhidpn(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def binary_value(x):
    sx = str(x)
    if sx == "0":
        return 0.0
    if sx == "1":
        return 1.0
    try:
        v = float(x)
        if v in (0.0, 1.0):
            return v
    except Exception:
        pass
    return np.nan


def contact_transition(v1, v2):
    if pd.isna(v1) or pd.isna(v2):
        return ""
    v1, v2 = int(v1), int(v2)
    return {
        (1, 1): "weekly_to_weekly",
        (0, 1): "below_to_weekly",
        (1, 0): "weekly_to_below",
        (0, 0): "below_to_below",
    }[(v1, v2)]


def loneliness_direction(delta, tol=1e-8):
    if pd.isna(delta):
        return ""
    if delta > tol:
        return "increased"
    if delta < -tol:
        return "decreased"
    return "stable"


def add_row(rows, section, measure, cohort, category, count, denominator, notes=""):
    rows.append({
        "section": section,
        "measure": measure,
        "cohort": cohort,
        "category": category,
        "count": int(count),
        "denominator": int(denominator) if denominator is not None else "",
        "percent": (
            round(100.0 * count / denominator, 1)
            if denominator not in (None, 0)
            else ""
        ),
        "notes": notes,
    })


# ---------------------------------------------------------------------
# LOAD MASTER + DEFINE STARTING SAMPLE
# ---------------------------------------------------------------------
master = pd.read_csv(MASTER_CSV, low_memory=False)
master["hhidpn"] = normalize_hhidpn(master["hhidpn"])
master["cohort"] = master["cohort"].astype(str)

for c in [
    "T1 loneliness", "T2 loneliness", "Loneliness T2-T1",
    "self_completed_loneliness_T1", "self_completed_loneliness_T2",
    "Cog_T2", "Cog_T3", "lives_alone_T1", "lives_alone_T2",
]:
    master[c] = pd.to_numeric(master[c], errors="coerce")

# Require actual T1/T2 loneliness values AND self-completion at both waves.
base = master[
    master["T1 loneliness"].notna()
    & master["T2 loneliness"].notna()
    & (master["self_completed_loneliness_T1"] == 1)
    & (master["self_completed_loneliness_T2"] == 1)
].copy()

# Recompute change directly rather than relying only on a previously derived column.
base["delta_loneliness"] = base["T2 loneliness"] - base["T1 loneliness"]
base["loneliness_direction"] = base["delta_loneliness"].map(loneliness_direction)

# ---------------------------------------------------------------------
# LOAD CORRECTED CHILD CONTACT
# ---------------------------------------------------------------------
child = pd.read_csv(CHILD_CSV, low_memory=False)
child["hhidpn"] = normalize_hhidpn(child["hhidpn"])
for c in ["weekly_contact_children_T1", "weekly_contact_children_T2"]:
    child[c] = pd.to_numeric(child[c], errors="coerce")

child["child_transition"] = [
    contact_transition(a, b)
    for a, b in zip(
        child["weekly_contact_children_T1"],
        child["weekly_contact_children_T2"],
    )
]

# ---------------------------------------------------------------------
# DIRECTLY EXTRACT RELATIVES/FRIENDS CONTACT FROM H_HRS_d.dta
# ---------------------------------------------------------------------
needed = ["hhidpn", "r8rfcnt", "r9rfcnt", "r10rfcnt", "r11rfcnt"]

with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(H_HRS_ZIP) as z:
        z.extract("H_HRS_d.dta", td)
    hrs = pd.read_stata(
        Path(td) / "H_HRS_d.dta",
        columns=needed,
        convert_categoricals=False,
        convert_missing=True,
    )

hrs["hhidpn"] = normalize_hhidpn(hrs["hhidpn"])

rf_parts = []
for cohort, timing in COHORT_MAP.items():
    w1 = timing["T1_wave"]
    w2 = timing["T2_wave"]

    ids = base.loc[base["cohort"] == cohort, ["cohort", "hhidpn"]]
    x = ids.merge(
        hrs[["hhidpn", f"r{w1}rfcnt", f"r{w2}rfcnt"]],
        on="hhidpn",
        how="left",
    )
    x["weekly_rf_T1"] = x[f"r{w1}rfcnt"].map(binary_value)
    x["weekly_rf_T2"] = x[f"r{w2}rfcnt"].map(binary_value)
    x["rf_transition"] = [
        contact_transition(a, b)
        for a, b in zip(x["weekly_rf_T1"], x["weekly_rf_T2"])
    ]
    rf_parts.append(
        x[["cohort", "hhidpn", "weekly_rf_T1", "weekly_rf_T2", "rf_transition"]]
    )

rf = pd.concat(rf_parts, ignore_index=True)

# ---------------------------------------------------------------------
# MERGE FEASIBILITY VARIABLES INTO ONE PARTICIPANT TABLE
# ---------------------------------------------------------------------
x = base.merge(
    rf,
    on=["cohort", "hhidpn"],
    how="left",
).merge(
    child[
        [
            "cohort", "hhidpn",
            "weekly_contact_children_T1",
            "weekly_contact_children_T2",
            "child_transition",
            "child_contact_status_T1",
            "child_contact_status_T2",
        ]
    ],
    on=["cohort", "hhidpn"],
    how="left",
)

x["rf_valid_both"] = x["weekly_rf_T1"].notna() & x["weekly_rf_T2"].notna()
x["child_valid_both"] = (
    x["weekly_contact_children_T1"].notna()
    & x["weekly_contact_children_T2"].notna()
)
x["living_valid_both"] = x["lives_alone_T1"].notna() & x["lives_alone_T2"].notna()

# Cognition is used only for sample retention, not outcome association.
x["cog_T2_T3_valid"] = x["Cog_T2"].notna() & x["Cog_T3"].notna()

# ---------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------
rows = []

for cohort in ["A", "B", "Combined"]:
    z = x if cohort == "Combined" else x[x["cohort"] == cohort]
    n = len(z)

    add_row(
        rows, "Base sample", "Self-completed loneliness T1 and T2",
        cohort, "valid", n, n
    )

    # Measurement availability
    for measure, flag in [
        ("Weekly relatives/friends contact", "rf_valid_both"),
        ("Weekly child contact", "child_valid_both"),
        ("Living alone", "living_valid_both"),
    ]:
        count = int(z[flag].sum())
        note = ""
        if measure == "Weekly child contact":
            note = (
                "No-children (.k) is not recoded as 0; it is outside the "
                "binary child-contact-frequency analysis."
            )
        add_row(
            rows, "Measurement availability", measure, cohort,
            "valid at T1 and T2", count, n, note
        )

    # Cognition-ready sample for each measure
    for measure, flag in [
        ("Weekly relatives/friends contact", "rf_valid_both"),
        ("Weekly child contact", "child_valid_both"),
        ("Living alone", "living_valid_both"),
    ]:
        count = int((z[flag] & z["cog_T2_T3_valid"]).sum())
        add_row(
            rows, "Cognition-ready sample", measure, cohort,
            "valid measure + T2/T3 cognition", count, n,
            "Cognition used only to determine retained sample size."
        )

    # Loneliness direction
    for cat in ["decreased", "stable", "increased"]:
        count = int((z["loneliness_direction"] == cat).sum())
        add_row(
            rows, "Loneliness change", "Loneliness T1->T2",
            cohort, cat, count, n
        )

    # Relative/friend transitions among people valid at both waves
    zr = z[z["rf_valid_both"]]
    for cat in [
        "weekly_to_weekly",
        "below_to_weekly",
        "weekly_to_below",
        "below_to_below",
    ]:
        count = int((zr["rf_transition"] == cat).sum())
        add_row(
            rows, "Contact trajectories",
            "Weekly relatives/friends contact",
            cohort, cat, count, len(zr)
        )

    # Child transitions among people valid at both waves
    zc = z[z["child_valid_both"]]
    for cat in [
        "weekly_to_weekly",
        "below_to_weekly",
        "weekly_to_below",
        "below_to_below",
    ]:
        count = int((zc["child_transition"] == cat).sum())
        add_row(
            rows, "Contact trajectories",
            "Weekly child contact",
            cohort, cat, count, len(zc)
        )

    # Living-alone transitions
    zl = z[z["living_valid_both"]].copy()
    living_map = {
        (0, 0): "not_alone_to_not_alone",
        (0, 1): "not_alone_to_living_alone",
        (1, 0): "living_alone_to_not_alone",
        (1, 1): "living_alone_to_living_alone",
    }
    living_transition = []
    for a, b in zip(zl["lives_alone_T1"], zl["lives_alone_T2"]):
        if pd.isna(a) or pd.isna(b):
            living_transition.append("")
        else:
            living_transition.append(living_map[(int(a), int(b))])
    zl["living_transition"] = living_transition

    for cat in living_map.values():
        count = int((zl["living_transition"] == cat).sum())
        add_row(
            rows, "Living arrangement trajectories",
            "Living alone", cohort, cat, count, len(zl)
        )

summary = pd.DataFrame(rows)
summary.to_csv(OUT_SUMMARY, index=False)

# ---------------------------------------------------------------------
# LONELINESS DIRECTION x CONTACT TRAJECTORY
#
# Require the social measure and T2/T3 cognition so these are close to
# the cells that could actually contribute to the eventual analysis.
# ---------------------------------------------------------------------
cross_rows = []

def make_cross(measure, valid_flag, transition_col):
    z = x[x[valid_flag] & x["cog_T2_T3_valid"]].copy()
    for cohort in ["A", "B", "Combined"]:
        zz = z if cohort == "Combined" else z[z["cohort"] == cohort]
        for loneliness_cat in ["decreased", "stable", "increased"]:
            for contact_cat in [
                "weekly_to_weekly",
                "below_to_weekly",
                "weekly_to_below",
                "below_to_below",
            ]:
                count = int(
                    (
                        (zz["loneliness_direction"] == loneliness_cat)
                        & (zz[transition_col] == contact_cat)
                    ).sum()
                )
                cross_rows.append({
                    "measure": measure,
                    "cohort": cohort,
                    "loneliness_direction": loneliness_cat,
                    "contact_trajectory": contact_cat,
                    "count": count,
                    "analysis_sample_n": len(zz),
                })

make_cross(
    "Weekly relatives/friends contact",
    "rf_valid_both",
    "rf_transition",
)
make_cross(
    "Weekly child contact",
    "child_valid_both",
    "child_transition",
)

cross = pd.DataFrame(cross_rows)
cross.to_csv(OUT_CROSSTAB, index=False)

print(f"Wrote: {OUT_SUMMARY}")
print(f"Wrote: {OUT_CROSSTAB}")

print("\nStarting sample:")
print(x.groupby("cohort").size())

print("\nCombined cognition-ready sample sizes:")
for label, flag in [
    ("Relatives/friends", "rf_valid_both"),
    ("Children", "child_valid_both"),
    ("Living alone", "living_valid_both"),
]:
    n = int((x[flag] & x["cog_T2_T3_valid"]).sum())
    print(f"{label}: {n}")
