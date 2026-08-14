"""
Add only the final contact-transition and contact-model eligibility fields
to the adjusted one-row-per-participant HRS master dataset.

Input
-----
HRS_participant_level_cohorts_updated_with_social_connections(1).csv

New columns (6 total)
---------------------
friends_contact_transition
eligible_friends_contact_model
other_relatives_contact_transition
eligible_other_relatives_contact_model
children_contact_transition
eligible_children_contact_model

Transition coding
-----------------
1 -> 1 = maintained_frequent
0 -> 1 = increased_contact
1 -> 0 = decreased_contact
0 -> 0 = maintained_infrequent

Eligibility definition
----------------------
eligible_x_contact_model = 1 if:
  1. eligible_basic_analysis == 1
     (already requires valid/self-completed T1/T2 loneliness
      and valid Cog at T2 and T3)
  2. x contact is valid 0/1 at both T1 and T2.

No core-covariate requirement is included in this flag.

Childless respondents
---------------------
No-child respondents remain in the master dataset. Their child-contact value
remains null, so children_contact_transition is blank and
eligible_children_contact_model = 0.

Outputs
-------
HRS_participant_level_cohorts_with_contact_transitions.csv
HRS_contact_transition_eligibility_audit.csv
"""

import csv
from pathlib import Path
from collections import Counter

INPUT = Path("HRS_participant_level_cohorts_updated_with_social_connections(1).csv")
OUTPUT = Path("HRS_participant_level_cohorts_with_contact_transitions.csv")
AUDIT = Path("HRS_contact_transition_eligibility_audit.csv")

DOMAINS = {
    "friends": (
        "weekly_contact_friends_T1",
        "weekly_contact_friends_T2",
    ),
    "other_relatives": (
        "weekly_contact_other_relatives_T1",
        "weekly_contact_other_relatives_T2",
    ),
    "children": (
        "weekly_contact_children_T1",
        "weekly_contact_children_T2",
    ),
}

NEW_COLUMNS = [
    "friends_contact_transition",
    "eligible_friends_contact_model",
    "other_relatives_contact_transition",
    "eligible_other_relatives_contact_model",
    "children_contact_transition",
    "eligible_children_contact_model",
]

TRANSITION_MAP = {
    ("1", "1"): "maintained_frequent",
    ("0", "1"): "increased_contact",
    ("1", "0"): "decreased_contact",
    ("0", "0"): "maintained_infrequent",
}

def clean_binary(value):
    s = str(value).strip()
    if not s:
        return None

    try:
        x = float(s)
        if x == 0:
            return "0"
        if x == 1:
            return "1"
    except Exception:
        pass

    return None

def is_one(value):
    try:
        return float(str(value).strip()) == 1
    except Exception:
        return False

with INPUT.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    original_columns = list(reader.fieldnames)
    rows = list(reader)

required = ["cohort", "hhidpn", "eligible_basic_analysis"]
for pair in DOMAINS.values():
    required.extend(pair)

missing = [c for c in required if c not in original_columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if any(c in original_columns for c in NEW_COLUMNS):
    raise ValueError("One or more output columns already exist.")

out_rows = []

for row in rows:
    out = dict(row)
    basic_ok = is_one(row["eligible_basic_analysis"])

    for domain, (t1_col, t2_col) in DOMAINS.items():
        t1 = clean_binary(row.get(t1_col, ""))
        t2 = clean_binary(row.get(t2_col, ""))

        valid_contact = (
            t1 is not None
            and t2 is not None
        )

        out[f"{domain}_contact_transition"] = (
            TRANSITION_MAP[(t1, t2)]
            if valid_contact
            else ""
        )

        out[f"eligible_{domain}_contact_model"] = int(
            basic_ok and valid_contact
        )

    out_rows.append(out)

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=original_columns + NEW_COLUMNS
    )
    writer.writeheader()
    writer.writerows(out_rows)

audit = []

def add(section, domain, cohort, statistic, count, denominator=None, notes=""):
    pct = ""
    if denominator not in (None, 0):
        pct = round(100 * count / denominator, 1)

    audit.append({
        "section": section,
        "domain": domain,
        "cohort": cohort,
        "statistic": statistic,
        "count": count,
        "denominator": "" if denominator is None else denominator,
        "percent": pct,
        "notes": notes,
    })

keys = [(r["cohort"], r["hhidpn"]) for r in out_rows]

unchanged = all(
    old.get(c, "") == new.get(c, "")
    for old, new in zip(rows, out_rows)
    for c in original_columns
)

add("structural_validation", "all", "Combined",
    "input_rows", len(rows), len(rows))
add("structural_validation", "all", "Combined",
    "output_rows", len(out_rows), len(rows))
add("structural_validation", "all", "Combined",
    "unique_cohort_hhidpn", len(set(keys)), len(out_rows))
add("structural_validation", "all", "Combined",
    "original_columns_retained", len(original_columns), len(original_columns))
add("structural_validation", "all", "Combined",
    "new_columns_added", len(NEW_COLUMNS), len(NEW_COLUMNS))
add("structural_validation", "all", "Combined",
    "all_original_values_unchanged", int(unchanged), 1)

for cohort in ("A", "B", "Combined"):
    z = out_rows if cohort == "Combined" else [
        r for r in out_rows
        if r["cohort"] == cohort
    ]

    add(
        "eligibility", "all", cohort,
        "eligible_basic_analysis",
        sum(is_one(r["eligible_basic_analysis"]) for r in z),
        len(z),
    )

for domain, (t1_col, t2_col) in DOMAINS.items():
    trans_col = f"{domain}_contact_transition"
    elig_col = f"eligible_{domain}_contact_model"

    for cohort in ("A", "B", "Combined"):
        z = out_rows if cohort == "Combined" else [
            r for r in out_rows
            if r["cohort"] == cohort
        ]

        valid = [
            r for r in z
            if clean_binary(r[t1_col]) is not None
            and clean_binary(r[t2_col]) is not None
        ]

        add(
            "contact_availability",
            domain,
            cohort,
            "valid_contact_T1_T2",
            len(valid),
            len(z),
        )

        for tr in [
            "maintained_frequent",
            "increased_contact",
            "decreased_contact",
            "maintained_infrequent",
        ]:
            add(
                "transition_distribution",
                domain,
                cohort,
                tr,
                sum(r[trans_col] == tr for r in valid),
                len(valid),
            )

        add(
            "eligibility",
            domain,
            cohort,
            "eligible_contact_model",
            sum(is_one(r[elig_col]) for r in z),
            len(z),
            "Requires eligible_basic_analysis=1 plus valid T1/T2 contact."
        )

bad_child_transition = sum(
    (
        clean_binary(r["weekly_contact_children_T1"]) is None
        or clean_binary(r["weekly_contact_children_T2"]) is None
    )
    and r["children_contact_transition"] != ""
    for r in out_rows
)

bad_child_eligible = sum(
    (
        clean_binary(r["weekly_contact_children_T1"]) is None
        or clean_binary(r["weekly_contact_children_T2"]) is None
    )
    and is_one(r["eligible_children_contact_model"])
    for r in out_rows
)

add(
    "child_null_validation",
    "children",
    "Combined",
    "invalid_or_null_child_contact_with_transition",
    bad_child_transition,
    len(out_rows),
    "Must be 0."
)
add(
    "child_null_validation",
    "children",
    "Combined",
    "invalid_or_null_child_contact_marked_eligible",
    bad_child_eligible,
    len(out_rows),
    "Must be 0."
)

with AUDIT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "section", "domain", "cohort", "statistic",
            "count", "denominator", "percent", "notes",
        ]
    )
    writer.writeheader()
    writer.writerows(audit)

print(f"Wrote {OUTPUT}")
print(f"Wrote {AUDIT}")
