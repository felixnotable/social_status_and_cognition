"""
HRS contact-transition + loneliness sample-size audit

Professor-requested descriptive audit:
1. Produce four T1->T2 contact-transition counts separately for each
   available relationship-specific contact domain and separately for Cohort A/B.
2. Cross-tabulate each contact transition with simple loneliness-change
   categories (decreased / stable / increased) to inspect sample sizes.
3. Do NOT run the final interaction regressions.

Primary statistical model planned later:
T3 cognition =
    T2 cognition
    + T1 loneliness
    + continuous loneliness change
    + contact-transition group
    + loneliness-change x contact-transition group
    + covariates

Inputs:
- HRS_participant_level_cohorts_updated.csv
- HRS_separate_friend_relative_contact_T1_T2.csv
- HRS_child_contact_correct_T1_T2.csv

Output:
- HRS_contact_transition_loneliness_audit.csv
"""

import csv
from pathlib import Path
from collections import Counter

FR_FILE = Path("HRS_separate_friend_relative_contact_T1_T2.csv")
CHILD_FILE = Path("HRS_child_contact_correct_T1_T2.csv")
MASTER_FILE = Path("HRS_participant_level_cohorts_updated.csv")
OUT_AUDIT = Path("HRS_contact_transition_loneliness_audit.csv")

def norm_id(x):
    try:
        return str(int(float(str(x).strip())))
    except Exception:
        return ""

def as_num(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None

def loneliness_dir(delta):
    if delta is None:
        return ""
    if delta < -1e-8:
        return "decreased"
    if delta > 1e-8:
        return "increased"
    return "stable"

def contact_transition(a, b):
    if a is None or b is None:
        return ""
    a, b = int(a), int(b)
    return {
        (1,1): "maintained_frequent",
        (0,1): "increased_contact",
        (1,0): "decreased_contact",
        (0,0): "maintained_infrequent",
    }[(a,b)]

# Base sample: valid T1/T2 loneliness + self-completed loneliness at both waves.
master = {}
with MASTER_FILE.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        hid = norm_id(r["hhidpn"])
        cohort = r["cohort"]
        t1 = as_num(r["T1 loneliness"])
        t2 = as_num(r["T2 loneliness"])
        s1 = as_num(r["self_completed_loneliness_T1"])
        s2 = as_num(r["self_completed_loneliness_T2"])
        if cohort not in ("A","B") or t1 is None or t2 is None or s1 != 1 or s2 != 1:
            continue
        master[(cohort,hid)] = {
            "loneliness_direction": loneliness_dir(t2-t1),
            "loneliness_change": t2-t1,
        }

domains = {
    "friends": {},
    "other_relatives": {},
    "children": {},
}

# Friend and other-relative contact reconstructed from raw HRS Leave-Behind items.
with FR_FILE.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        key = (r["cohort"], norm_id(r["hhidpn"]))
        if key not in master:
            continue

        f1 = as_num(r["weekly_friends_T1"])
        f2 = as_num(r["weekly_friends_T2"])
        o1 = as_num(r["weekly_other_relatives_T1"])
        o2 = as_num(r["weekly_other_relatives_T2"])

        domains["friends"][key] = contact_transition(f1, f2)
        domains["other_relatives"][key] = contact_transition(o1, o2)

# Correct respondent-level child-contact variables.
# Participants with no children are retained in the master dataset but have
# no valid 0/1 child-contact transition and therefore do not enter this
# child-contact-specific transition audit.
with CHILD_FILE.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        key = (r["cohort"], norm_id(r["hhidpn"]))
        if key not in master:
            continue
        c1 = as_num(r["weekly_contact_children_T1"])
        c2 = as_num(r["weekly_contact_children_T2"])
        domains["children"][key] = contact_transition(c1, c2)

transition_order = [
    "maintained_frequent",
    "increased_contact",
    "decreased_contact",
    "maintained_infrequent",
]
loneliness_order = ["decreased", "stable", "increased"]

pretty = {
    "friends": "Friends",
    "other_relatives": "Other relatives",
    "children": "Children",
    "maintained_frequent": "Maintained frequent contact (1→1)",
    "increased_contact": "Increased contact (0→1)",
    "decreased_contact": "Decreased contact (1→0)",
    "maintained_infrequent": "Maintained infrequent contact (0→0)",
    "decreased": "Loneliness decreased",
    "stable": "Loneliness stable",
    "increased": "Loneliness increased",
}

rows = []

for domain, data in domains.items():
    for cohort in ("A","B"):
        valid = [(k,tr) for k,tr in data.items() if k[0] == cohort and tr]
        denom = len(valid)

        cnt = Counter(tr for _,tr in valid)
        for tr in transition_order:
            n = cnt[tr]
            rows.append({
                "section": "contact_transition_distribution",
                "contact_domain": pretty[domain],
                "cohort": cohort,
                "contact_transition": pretty[tr],
                "loneliness_category": "",
                "count": n,
                "denominator": denom,
                "percent": round(100*n/denom, 1) if denom else "",
                "notes": (
                    "Denominator = participants with valid T1 and T2 contact "
                    "in this domain and valid self-completed T1/T2 loneliness."
                )
            })

        cross = Counter()
        for key, tr in valid:
            lcat = master[key]["loneliness_direction"]
            cross[(tr,lcat)] += 1

        for tr in transition_order:
            for lcat in loneliness_order:
                n = cross[(tr,lcat)]
                rows.append({
                    "section": "contact_transition_x_loneliness_category",
                    "contact_domain": pretty[domain],
                    "cohort": cohort,
                    "contact_transition": pretty[tr],
                    "loneliness_category": pretty[lcat],
                    "count": n,
                    "denominator": denom,
                    "percent": round(100*n/denom, 1) if denom else "",
                    "notes": (
                        "Loneliness categories are descriptive only; the "
                        "proposed primary model retains loneliness change "
                        "as a continuous variable."
                    )
                })

with OUT_AUDIT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "section","contact_domain","cohort","contact_transition",
            "loneliness_category","count","denominator","percent","notes"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT_AUDIT}")
