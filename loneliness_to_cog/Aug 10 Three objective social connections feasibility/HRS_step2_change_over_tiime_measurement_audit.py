"""
Step 2 — HRS objective-social-variable measurement audit

Purpose
-------
1. Verify the wave-specific Harmonized HRS variables used for:
   - weekly contact with children (RwKCNT)
   - weekly contact with relatives/friends (RwRFCNT)
   - weekly/monthly social activity (RwSOCWK / RwSOCMN)
2. Re-extract the correct respondent-level child-contact variables directly
   from H_HRS_d.dta.
3. Check T1/T2 availability and transition counts separately for Cohort A
   (W8 -> W10) and Cohort B (W9 -> W11).
4. Do NOT examine associations with cognition.

Required inputs
---------------
- H_HRS_d_stata.zip   (contains H_HRS_d.dta)
- HRS_participant_level_cohorts_updated.csv

The script writes:
- HRS_child_contact_correct_T1_T2.csv
- step2_measurement_audit_summary.csv
"""

from pathlib import Path
import zipfile
import tempfile
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# INPUTS — change these paths if necessary
# ---------------------------------------------------------------------
H_HRS_ZIP = Path("H_HRS_d_stata.zip")
MASTER_CSV = Path("HRS_participant_level_cohorts_updated.csv")

OUT_CHILD = Path("HRS_child_contact_correct_T1_T2.csv")
OUT_AUDIT = Path("step2_measurement_audit_summary.csv")

# Cohort timing used in the project
COHORT_MAP = {
    "A": {"T1_wave": 8, "T1_year": 2006, "T2_wave": 10, "T2_year": 2010},
    "B": {"T1_wave": 9, "T1_year": 2008, "T2_wave": 11, "T2_year": 2012},
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_hhidpn(s):
    """Normalize HHIDPN so merges are reliable."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def stata_raw_string(x):
    """
    pandas.read_stata(convert_missing=True) preserves Stata tagged missings.
    Converting to str gives values such as '.k', '.l', '.m', or '.'.
    """
    if pd.isna(x):
        return "."
    return str(x)


def binary_value(x):
    """Return 0/1 for valid binary values; otherwise NaN."""
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


def contact_status(raw):
    sx = stata_raw_string(raw)
    if sx == "1":
        return "weekly"
    if sx == "0":
        return "less_than_weekly"
    if sx == ".k":
        return "no_children"
    if sx == ".l":
        return "leave_behind_not_completed_or_ineligible"
    if sx == ".d":
        return "dont_know"
    if sx == ".r":
        return "refused"
    if sx == ".m":
        return "other_missing"
    if sx == ".":
        return "not_interviewed_or_plain_missing"
    return f"other:{sx}"


def transition(v1, v2):
    if pd.isna(v1) or pd.isna(v2):
        return ""
    v1, v2 = int(v1), int(v2)
    return {
        (1, 1): "maintained_weekly",
        (0, 1): "increased_to_weekly",
        (1, 0): "decreased_below_weekly",
        (0, 0): "persistently_below_weekly",
    }[(v1, v2)]


def pct(n, d):
    return 100.0 * n / d if d else np.nan


# ---------------------------------------------------------------------
# LOAD MASTER COHORT FILE
# ---------------------------------------------------------------------
master = pd.read_csv(MASTER_CSV, low_memory=False)
master["hhidpn"] = normalize_hhidpn(master["hhidpn"])
master["cohort"] = master["cohort"].astype(str)

# The feasibility subset used during Step 2:
# loneliness must have been self-completed at both T1 and T2.
self_lonely = master[
    (pd.to_numeric(master["self_completed_loneliness_T1"], errors="coerce") == 1)
    & (pd.to_numeric(master["self_completed_loneliness_T2"], errors="coerce") == 1)
].copy()

# ---------------------------------------------------------------------
# EXTRACT ONLY REQUIRED H_HRS VARIABLES
# ---------------------------------------------------------------------
needed = ["hhidpn"]

for w in [8, 9, 10, 11]:
    needed += [
        f"r{w}kcnt",     # any weekly child contact
        f"r{w}kcntf",    # weekly child contact in person
        f"r{w}kcntpm",   # weekly child contact phone/mail/email
        f"r{w}rfcnt",    # weekly relatives/friends contact
    ]

# Social activity begins at W9 in Harmonized HRS.
for w in [9, 10, 11]:
    needed += [f"r{w}socwk", f"r{w}socmn"]

needed = list(dict.fromkeys(needed))

with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(H_HRS_ZIP) as z:
        z.extract("H_HRS_d.dta", td)
    dta_path = Path(td) / "H_HRS_d.dta"

    hrs = pd.read_stata(
        dta_path,
        columns=needed,
        convert_categoricals=False,
        convert_missing=True,
    )

hrs["hhidpn"] = normalize_hhidpn(hrs["hhidpn"])

# ---------------------------------------------------------------------
# 1) RE-EXTRACT CORRECT RESPONDENT-LEVEL CHILD CONTACT
# ---------------------------------------------------------------------
child_rows = []

for cohort, timing in COHORT_MAP.items():
    ids = master.loc[master["cohort"] == cohort, ["cohort", "hhidpn"]].copy()

    w1 = timing["T1_wave"]
    w2 = timing["T2_wave"]

    cols = [
        "hhidpn",
        f"r{w1}kcnt", f"r{w2}kcnt",
        f"r{w1}kcntf", f"r{w2}kcntf",
        f"r{w1}kcntpm", f"r{w2}kcntpm",
    ]
    x = ids.merge(hrs[cols], on="hhidpn", how="left")

    raw1 = x[f"r{w1}kcnt"]
    raw2 = x[f"r{w2}kcnt"]

    x["T1_wave"] = w1
    x["T1_year"] = timing["T1_year"]
    x["T2_wave"] = w2
    x["T2_year"] = timing["T2_year"]
    x["child_contact_source_T1"] = f"R{w1}KCNT"
    x["child_contact_source_T2"] = f"R{w2}KCNT"

    x["weekly_contact_children_T1"] = raw1.map(binary_value)
    x["weekly_contact_children_T2"] = raw2.map(binary_value)

    x["child_contact_status_T1"] = raw1.map(contact_status)
    x["child_contact_status_T2"] = raw2.map(contact_status)

    x["child_contact_change_T2_minus_T1"] = (
        x["weekly_contact_children_T2"] - x["weekly_contact_children_T1"]
    )
    x["child_contact_transition"] = [
        transition(a, b)
        for a, b in zip(
            x["weekly_contact_children_T1"],
            x["weekly_contact_children_T2"],
        )
    ]

    x["weekly_contact_children_in_person_T1"] = x[f"r{w1}kcntf"].map(binary_value)
    x["weekly_contact_children_in_person_T2"] = x[f"r{w2}kcntf"].map(binary_value)
    x["weekly_contact_children_phone_mail_email_T1"] = x[f"r{w1}kcntpm"].map(binary_value)
    x["weekly_contact_children_phone_mail_email_T2"] = x[f"r{w2}kcntpm"].map(binary_value)

    # Preserve Stata source values so ".k" is not silently turned into 0.
    x["child_contact_raw_T1"] = raw1.map(stata_raw_string)
    x["child_contact_raw_T2"] = raw2.map(stata_raw_string)

    keep = [
        "cohort", "hhidpn",
        "T1_wave", "T1_year", "T2_wave", "T2_year",
        "child_contact_source_T1", "child_contact_source_T2",
        "weekly_contact_children_T1", "weekly_contact_children_T2",
        "child_contact_status_T1", "child_contact_status_T2",
        "child_contact_change_T2_minus_T1", "child_contact_transition",
        "weekly_contact_children_in_person_T1",
        "weekly_contact_children_in_person_T2",
        "weekly_contact_children_phone_mail_email_T1",
        "weekly_contact_children_phone_mail_email_T2",
        "child_contact_raw_T1", "child_contact_raw_T2",
    ]
    child_rows.append(x[keep])

child = pd.concat(child_rows, ignore_index=True)
child.to_csv(OUT_CHILD, index=False)

# ---------------------------------------------------------------------
# 2) AUDIT AVAILABILITY + TRANSITIONS IN SELF-COMPLETED LONELINESS SAMPLE
# ---------------------------------------------------------------------
audit_rows = []

def add_audit(measure, cohort, statistic, count, denominator=None, note=""):
    audit_rows.append({
        "measure": measure,
        "cohort": cohort,
        "statistic": statistic,
        "count": int(count),
        "denominator": "" if denominator is None else int(denominator),
        "percent": "" if denominator in (None, 0) else round(pct(count, denominator), 1),
        "note": note,
    })


def audit_binary_measure(cohort, label, var1, var2):
    base = self_lonely[self_lonely["cohort"] == cohort][["hhidpn"]]
    x = base.merge(hrs[["hhidpn", var1, var2]], on="hhidpn", how="left")

    x["v1"] = x[var1].map(binary_value)
    x["v2"] = x[var2].map(binary_value)

    n = len(x)
    valid1 = x["v1"].notna().sum()
    valid2 = x["v2"].notna().sum()
    both = (x["v1"].notna() & x["v2"].notna()).sum()

    add_audit(label, cohort, "starting_self_completed_loneliness_sample", n, n)
    add_audit(label, cohort, "valid_T1", valid1, n)
    add_audit(label, cohort, "valid_T2", valid2, n)
    add_audit(label, cohort, "valid_both_T1_T2", both, n)

    y = x[x["v1"].notna() & x["v2"].notna()].copy()
    transition_counts = {
        "0->0": ((y["v1"] == 0) & (y["v2"] == 0)).sum(),
        "0->1": ((y["v1"] == 0) & (y["v2"] == 1)).sum(),
        "1->0": ((y["v1"] == 1) & (y["v2"] == 0)).sum(),
        "1->1": ((y["v1"] == 1) & (y["v2"] == 1)).sum(),
    }
    for k, v in transition_counts.items():
        add_audit(label, cohort, f"transition_{k}", v, both)


# Relatives/friends contact: cleanly available in both cohorts.
audit_binary_measure("A", "weekly_relatives_friends_contact", "r8rfcnt", "r10rfcnt")
audit_binary_measure("B", "weekly_relatives_friends_contact", "r9rfcnt", "r11rfcnt")

# Correct child contact.
audit_binary_measure("A", "weekly_child_contact_RwKCNT", "r8kcnt", "r10kcnt")
audit_binary_measure("B", "weekly_child_contact_RwKCNT", "r9kcnt", "r11kcnt")

# Social activity:
# Cohort A cannot support T1->T2 change because RwSOCWK/RwSOCMN begin at W9
# and Cohort A T1 is Wave 8.
nA = len(self_lonely[self_lonely["cohort"] == "A"])
add_audit(
    "weekly_social_activity",
    "A",
    "not_longitudinally_usable",
    0,
    nA,
    "Cohort A requires W8->W10; RwSOCWK begins at W9."
)
add_audit(
    "monthly_social_activity",
    "A",
    "not_longitudinally_usable",
    0,
    nA,
    "Cohort A requires W8->W10; RwSOCMN begins at W9."
)

# Cohort B can be inspected, but W9->W11 comparability should be treated cautiously.
audit_binary_measure("B", "weekly_social_activity", "r9socwk", "r11socwk")
audit_binary_measure("B", "monthly_social_activity", "r9socmn", "r11socmn")

audit = pd.DataFrame(audit_rows)
audit.to_csv(OUT_AUDIT, index=False)

print(f"Wrote: {OUT_CHILD}")
print(f"Wrote: {OUT_AUDIT}")
print("\nKey self-completed-loneliness sample sizes:")
print(self_lonely.groupby("cohort").size())
