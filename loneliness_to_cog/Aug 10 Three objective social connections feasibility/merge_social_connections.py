"""
Add three objective social-connection domains to the latest one-row-per-participant
HRS master dataset.

New columns:
  weekly_contact_friends_T1
  weekly_contact_friends_T2
  weekly_contact_other_relatives_T1
  weekly_contact_other_relatives_T2
  weekly_contact_children_T1
  weekly_contact_children_T2

Coding:
  1 = at least weekly
  0 = less than weekly
  blank = unavailable/not valid

IMPORTANT:
Respondents with no children remain in the master dataset, but their
weekly_contact_children_T1/T2 value is blank/null rather than 0.

Cohort timing:
  Cohort A: T1=W8 (2006),  T2=W10 (2010)
  Cohort B: T1=W9 (2008),  T2=W11 (2012)

Inputs:
  HRS_participant_level_cohorts_updated(3)(1).csv
  HRS_child_contact_correct_T1_T2.csv
  raw_lb/H06LB_R.da, H06LB_R.dct
  raw_lb/H08LB_R.da, H08LB_R.dct
  raw_lb/H10LB_R.da, H10LB_R.dct
  raw_lb/H12LB_R.da, H12LB_R.dct

Outputs:
  HRS_participant_level_cohorts_updated_with_social_connections.csv
  HRS_social_connections_merge_validation.csv
"""

import csv
import re
from pathlib import Path
from collections import Counter

MASTER = Path("HRS_participant_level_cohorts_updated(3)(1).csv")
CHILD_SOURCE = Path("HRS_child_contact_correct_T1_T2.csv")
RAW = Path("raw_lb")

OUT_MERGED = Path("HRS_participant_level_cohorts_updated_with_social_connections.csv")
OUT_VALIDATION = Path("HRS_social_connections_merge_validation.csv")

COHORT_WAVES = {"A": (8, 10), "B": (9, 11)}
WAVE_SPECS = {
    8: ("H06LB_R.da", "H06LB_R.dct", "K"),
    9: ("H08LB_R.da", "H08LB_R.dct", "L"),
    10: ("H10LB_R.da", "H10LB_R.dct", "M"),
    11: ("H12LB_R.da", "H12LB_R.dct", "N"),
}

NEW_COLUMNS = [
    "weekly_contact_friends_T1",
    "weekly_contact_friends_T2",
    "weekly_contact_other_relatives_T1",
    "weekly_contact_other_relatives_T2",
    "weekly_contact_children_T1",
    "weekly_contact_children_T2",
]

def norm_id(x):
    try:
        return str(int(float(str(x).strip())))
    except Exception:
        return ""

def raw_int(x):
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None

def binary_output(x):
    s = str(x).strip()
    if not s:
        return ""
    try:
        v = float(s)
        if v in (0.0, 1.0):
            return str(int(v))
    except Exception:
        pass
    return ""

def dct_positions(dct_path, variable_names):
    specs = {}
    for line in Path(dct_path).read_text(errors="ignore").splitlines():
        m = re.search(r"_column\((\d+)\)\s+\w+\s+(\w+)\s+%(\d+)", line)
        if m and m.group(2) in variable_names:
            start = int(m.group(1)) - 1
            width = int(m.group(3))
            specs[m.group(2)] = (start, start + width)

    missing = set(variable_names) - set(specs)
    if missing:
        raise ValueError(f"Variables missing from {dct_path}: {sorted(missing)}")
    return specs

def derive_weekly(has_group, mode_values):
    has_group = raw_int(has_group)
    vals = [raw_int(v) for v in mode_values]

    if any(v in (1, 2) for v in vals):
        return 1

    valid = [v for v in vals if v in (1, 2, 3, 4, 5, 6)]

    if has_group == 5:
        return 0

    if has_group == 1 and valid and all(v in (3, 4, 5, 6) for v in valid):
        return 0

    return None

def as_output(v):
    return "" if v is None else str(int(v))

# Reconstruct friends and other relatives from raw Leave-Behind data.
wave_contact = {}

for wave, (da_name, dct_name, prefix) in WAVE_SPECS.items():
    names = [
        "HHID", "PN",
        f"{prefix}LB011",
        f"{prefix}LB013A", f"{prefix}LB013B", f"{prefix}LB013C",
        f"{prefix}LB015",
        f"{prefix}LB017A", f"{prefix}LB017B", f"{prefix}LB017C",
    ]

    pos = dct_positions(RAW / dct_name, names)
    records = {}

    with (RAW / da_name).open("r", encoding="latin1", errors="ignore") as fh:
        for line in fh:
            vals = {v: line[a:b].strip() for v, (a, b) in pos.items()}
            hhidpn = str(int(vals["HHID"].zfill(6) + vals["PN"].zfill(3)))

            records[hhidpn] = {
                "other_relatives": derive_weekly(
                    vals[f"{prefix}LB011"],
                    [vals[f"{prefix}LB013A"],
                     vals[f"{prefix}LB013B"],
                     vals[f"{prefix}LB013C"]],
                ),
                "friends": derive_weekly(
                    vals[f"{prefix}LB015"],
                    [vals[f"{prefix}LB017A"],
                     vals[f"{prefix}LB017B"],
                     vals[f"{prefix}LB017C"]],
                ),
            }

    wave_contact[wave] = records

# Correct child-contact source.
child_map = {}
with CHILD_SOURCE.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        key = (r["cohort"], norm_id(r["hhidpn"]))
        child_map[key] = {
            "T1": binary_output(r["weekly_contact_children_T1"]),
            "T2": binary_output(r["weekly_contact_children_T2"]),
            "status_T1": r.get("child_contact_status_T1", "").strip(),
            "status_T2": r.get("child_contact_status_T2", "").strip(),
        }

# Master.
with MASTER.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    original_columns = list(reader.fieldnames)
    master_rows = list(reader)

if any(c in original_columns for c in NEW_COLUMNS):
    raise ValueError("A target social-connection column already exists.")

merged_rows = []

for r in master_rows:
    cohort = r["cohort"]
    hhidpn = norm_id(r["hhidpn"])
    t1_wave, t2_wave = COHORT_WAVES[cohort]

    source_t1 = wave_contact[t1_wave].get(hhidpn, {})
    source_t2 = wave_contact[t2_wave].get(hhidpn, {})
    child = child_map.get((cohort, hhidpn), {})

    out = dict(r)
    out["weekly_contact_friends_T1"] = as_output(source_t1.get("friends"))
    out["weekly_contact_friends_T2"] = as_output(source_t2.get("friends"))
    out["weekly_contact_other_relatives_T1"] = as_output(source_t1.get("other_relatives"))
    out["weekly_contact_other_relatives_T2"] = as_output(source_t2.get("other_relatives"))
    out["weekly_contact_children_T1"] = child.get("T1", "")
    out["weekly_contact_children_T2"] = child.get("T2", "")
    merged_rows.append(out)

with OUT_MERGED.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=original_columns + NEW_COLUMNS)
    writer.writeheader()
    writer.writerows(merged_rows)

# Validation.
validation = []

def add(section, measure, cohort, statistic, count, denominator=None, notes=""):
    percent = ""
    if denominator not in (None, 0):
        percent = round(100 * count / denominator, 1)
    validation.append({
        "section": section,
        "measure": measure,
        "cohort": cohort,
        "statistic": statistic,
        "count": count,
        "denominator": "" if denominator is None else denominator,
        "percent": percent,
        "notes": notes,
    })

merged_ids = [(r["cohort"], norm_id(r["hhidpn"])) for r in merged_rows]

unchanged = all(
    old.get(c, "") == new.get(c, "")
    for old, new in zip(master_rows, merged_rows)
    for c in original_columns
)

add("structural_validation", "Master dataset", "Combined",
    "original_row_count", len(master_rows), len(master_rows))
add("structural_validation", "Merged dataset", "Combined",
    "merged_row_count", len(merged_rows), len(master_rows))
add("structural_validation", "Merged dataset", "Combined",
    "unique_cohort_hhidpn", len(set(merged_ids)), len(merged_rows))
add("structural_validation", "Merged dataset", "Combined",
    "original_columns_retained", len(original_columns), len(original_columns))
add("structural_validation", "Merged dataset", "Combined",
    "new_columns_added", len(NEW_COLUMNS), len(NEW_COLUMNS))
add("structural_validation", "Merged dataset", "Combined",
    "all_original_cell_values_unchanged", int(unchanged), 1)

domain_cols = {
    "Friends": ("weekly_contact_friends_T1", "weekly_contact_friends_T2"),
    "Other relatives": ("weekly_contact_other_relatives_T1", "weekly_contact_other_relatives_T2"),
    "Children": ("weekly_contact_children_T1", "weekly_contact_children_T2"),
}

for measure, (c1, c2) in domain_cols.items():
    for cohort in ("A", "B", "Combined"):
        z = merged_rows if cohort == "Combined" else [
            r for r in merged_rows if r["cohort"] == cohort
        ]
        n = len(z)

        valid_t1 = sum(r[c1] in ("0", "1") for r in z)
        valid_t2 = sum(r[c2] in ("0", "1") for r in z)
        valid_both = sum(
            r[c1] in ("0", "1") and r[c2] in ("0", "1")
            for r in z
        )

        add("measurement_availability", measure, cohort,
            "valid_T1", valid_t1, n)
        add("measurement_availability", measure, cohort,
            "valid_T2", valid_t2, n)
        add("measurement_availability", measure, cohort,
            "valid_both_T1_T2", valid_both, n)

transition_labels = {
    ("1", "1"): "maintained_frequent_1_to_1",
    ("0", "1"): "increased_contact_0_to_1",
    ("1", "0"): "decreased_contact_1_to_0",
    ("0", "0"): "maintained_infrequent_0_to_0",
}

for measure, (c1, c2) in domain_cols.items():
    for cohort in ("A", "B", "Combined"):
        z = merged_rows if cohort == "Combined" else [
            r for r in merged_rows if r["cohort"] == cohort
        ]
        valid = [
            r for r in z
            if r[c1] in ("0", "1") and r[c2] in ("0", "1")
        ]

        counts = Counter(
            transition_labels[(r[c1], r[c2])]
            for r in valid
        )

        for label in transition_labels.values():
            add("transition_distribution", measure, cohort,
                label, counts[label], len(valid))

# No-child validation.
for cohort in ("A", "B", "Combined"):
    keys = [k for k in child_map if cohort == "Combined" or k[0] == cohort]

    n_no_t1 = sum(child_map[k]["status_T1"] == "no_children" for k in keys)
    n_no_t2 = sum(child_map[k]["status_T2"] == "no_children" for k in keys)

    bad_t1 = sum(
        child_map[k]["status_T1"] == "no_children" and child_map[k]["T1"] != ""
        for k in keys
    )
    bad_t2 = sum(
        child_map[k]["status_T2"] == "no_children" and child_map[k]["T2"] != ""
        for k in keys
    )

    add("child_null_validation", "Children", cohort,
        "no_children_T1", n_no_t1, len(keys))
    add("child_null_validation", "Children", cohort,
        "no_children_T2", n_no_t2, len(keys))
    add("child_null_validation", "Children", cohort,
        "no_children_with_nonnull_contact_T1", bad_t1, n_no_t1)
    add("child_null_validation", "Children", cohort,
        "no_children_with_nonnull_contact_T2", bad_t2, n_no_t2)

with OUT_VALIDATION.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "section", "measure", "cohort", "statistic",
            "count", "denominator", "percent", "notes"
        ],
    )
    writer.writeheader()
    writer.writerows(validation)

print(f"Wrote {OUT_MERGED}")
print(f"Wrote {OUT_VALIDATION}")
