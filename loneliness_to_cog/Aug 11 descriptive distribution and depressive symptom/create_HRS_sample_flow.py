"""
Create a progressive sample-flow summary for each social-contact domain
and for Cohort A, Cohort B, and Combined.

Input:
    HRS_participant_level_cohorts_with_contact_transitions.csv

Flow:
    1. All participants in master
    2. Valid/self-completed T1 and T2 loneliness
    3. Plus valid T1/T2 contact for the domain
    4. Plus cognition T2
    5. Plus cognition T3
    6. Plus complete core covariates

Core covariates:
    age_T2, sex, education, race_ethnicity, wealth_T2, marital_status_T2

No participant is deleted from the master. This script only summarizes
progressive analytic eligibility.
"""

import csv
from pathlib import Path
from collections import OrderedDict

INPUT = Path("HRS_participant_level_cohorts_with_contact_transitions.csv")
OUTPUT = Path("HRS_sample_flow_by_contact_domain.csv")

CORE_COVARIATES = [
    "age_T2",
    "sex",
    "education",
    "race_ethnicity",
    "wealth_T2",
    "marital_status_T2",
]

DOMAINS = OrderedDict([
    ("Friends", ("weekly_contact_friends_T1", "weekly_contact_friends_T2")),
    ("Other relatives", ("weekly_contact_other_relatives_T1", "weekly_contact_other_relatives_T2")),
    ("Children", ("weekly_contact_children_T1", "weekly_contact_children_T2")),
])

def present(v):
    return str(v).strip() != ""

def one(v):
    try:
        return float(str(v).strip()) == 1
    except Exception:
        return False

def binary(v):
    try:
        return float(str(v).strip()) in (0.0, 1.0)
    except Exception:
        return False

def valid_loneliness(r):
    return (
        present(r["T1 loneliness"])
        and present(r["T2 loneliness"])
        and one(r["self_completed_loneliness_T1"])
        and one(r["self_completed_loneliness_T2"])
    )

def has_cog(r, tp):
    return one(r[f"has_Cog_{tp}"])

def complete_core_covariates(r):
    return all(present(r[c]) for c in CORE_COVARIATES)

with INPUT.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

out = []

for domain, (c1, c2) in DOMAINS.items():
    for cohort in ("A", "B", "Combined"):
        z = rows if cohort == "Combined" else [
            r for r in rows if r["cohort"] == cohort
        ]
        master_n = len(z)

        current = z
        counts = [
            ("01_master", "All participants in master dataset", len(current))
        ]

        current = [r for r in current if valid_loneliness(r)]
        counts.append(
            ("02_loneliness",
             "Valid/self-completed loneliness at T1 and T2",
             len(current))
        )

        current = [
            r for r in current
            if binary(r[c1]) and binary(r[c2])
        ]
        counts.append(
            ("03_contact",
             f"Plus valid {domain.lower()} contact at T1 and T2",
             len(current))
        )

        current = [r for r in current if has_cog(r, "T2")]
        counts.append(("04_cog_T2", "Plus cognition at T2", len(current)))

        current = [r for r in current if has_cog(r, "T3")]
        counts.append(
            ("05_cog_T3",
             "Plus cognition at T3 (contact-model eligible)",
             len(current))
        )

        current = [r for r in current if complete_core_covariates(r)]
        counts.append(
            ("06_core_covariates",
             "Plus complete core covariates (fully adjusted sample)",
             len(current))
        )

        previous = None
        for code, label, n in counts:
            out.append({
                "contact_domain": domain,
                "cohort": cohort,
                "stage_code": code,
                "stage": label,
                "remaining_n": n,
                "excluded_since_previous_stage":
                    "" if previous is None else previous - n,
                "percent_of_master":
                    round(100 * n / master_n, 1) if master_n else "",
                "percent_retained_from_previous_stage":
                    "" if previous in (None, 0)
                    else round(100 * n / previous, 1),
            })
            previous = n

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(out[0].keys())
    )
    writer.writeheader()
    writer.writerows(out)

print(f"Wrote {OUTPUT}")
