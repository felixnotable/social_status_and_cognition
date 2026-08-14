"""
Add CESD-7 depressive symptom scores (excluding loneliness) to the latest
one-row-per-participant HRS master with the three objective social connections.

Inputs
------
HRS_participant_level_cohorts_updated_with_social_connections.csv
randhrs1992_2022v1.dta

Cohort timing
-------------
A: T1 = Wave 8,  T2 = Wave 10
B: T1 = Wave 9,  T2 = Wave 11

CESD-7 construction
-------------------
RAND HRS RwCESD is the sum of six negative indicators plus the inverse of
2 positive indicators. To avoid overlap with the project's UCLA loneliness
exposure, the RwFLONE (felt lonely) component is excluded.

CESD7 = DEPRES + EFFORT + SLEEPR + FSAD + GOING
        + (1 - WHAPPY) + (1 - ENLIFE)

Range: 0-7, higher = more depressive symptoms.
Missingness rule: all seven retained items must be valid binary 0/1; otherwise
cesd7 is left blank/null.

Outputs
-------
HRS_participant_level_cohorts_updated_with_social_connections_cesd7.csv
HRS_cesd7_merge_validation.csv
"""

import csv
from pathlib import Path
from collections import Counter

import numpy as np
from pandas.io.stata import StataReader

MASTER = Path("HRS_participant_level_cohorts_updated_with_social_connections.csv")
RAND_DTA = Path("/mnt/data/randtmp/randhrs1992_2022v1.dta")
OUT = Path("HRS_participant_level_cohorts_updated_with_social_connections_cesd7.csv")
AUDIT = Path("HRS_cesd7_merge_validation.csv")

COHORT_WAVES = {
    "A": (8, 10),
    "B": (9, 11),
}

NEGATIVE_ITEMS = ["depres", "effort", "sleepr", "fsad", "going"]
POSITIVE_ITEMS = ["whappy", "enlife"]
NEW_COLUMNS = ["cesd7_T1", "cesd7_T2"]


def norm_id(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def open_rand_arrays(path):
    """Parse Stata metadata once and memory-map the fixed observation records."""
    reader = StataReader(path, convert_categoricals=False, columns=["hhidpn"])
    reader.read(nrows=1)

    mm = np.memmap(
        path,
        mode="r",
        dtype=reader._dtype,
        offset=reader._data_location,
        shape=(reader._nobs,),
    )
    var_index = {name: i for i, name in enumerate(reader._varlist)}

    def arr(name):
        return mm[f"s{var_index[name]}"]

    return reader, mm, arr


def build_cesd7_lookup(rand_path):
    reader, mm, arr = open_rand_arrays(rand_path)

    ids = arr("hhidpn").astype(np.int64)
    row_index = {int(pid): i for i, pid in enumerate(ids)}

    # Verify expected variables exist before processing.
    needed = []
    for wave in (8, 9, 10, 11):
        needed.extend(f"r{wave}{x}" for x in NEGATIVE_ITEMS + POSITIVE_ITEMS)
    missing = [name for name in needed if name not in reader._varlist]
    if missing:
        raise ValueError(f"Missing expected RAND CES-D variables: {missing}")

    def score(pid, wave):
        j = row_index.get(pid)
        if j is None:
            return None

        vals = {
            item: int(arr(f"r{wave}{item}")[j])
            for item in NEGATIVE_ITEMS + POSITIVE_ITEMS
        }

        # RAND special missing codes are outside 0/1 (for example 101).
        if not all(v in (0, 1) for v in vals.values()):
            return None

        total = (
            sum(vals[item] for item in NEGATIVE_ITEMS)
            + sum(1 - vals[item] for item in POSITIVE_ITEMS)
        )
        return int(total)

    return score, mm


def main():
    score, rand_memmap = build_cesd7_lookup(RAND_DTA)

    with MASTER.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)

    if any(c in original_fields for c in NEW_COLUMNS):
        raise ValueError("cesd7_T1 or cesd7_T2 already exists in the input master.")

    merged = []
    for row in rows:
        cohort = row.get("cohort", "")
        if cohort not in COHORT_WAVES:
            raise ValueError(f"Unexpected cohort: {cohort!r}")

        pid = norm_id(row.get("hhidpn", ""))
        w1, w2 = COHORT_WAVES[cohort]

        s1 = score(pid, w1) if pid is not None else None
        s2 = score(pid, w2) if pid is not None else None

        out = dict(row)
        out["cesd7_T1"] = "" if s1 is None else str(s1)
        out["cesd7_T2"] = "" if s2 is None else str(s2)
        merged.append(out)

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=original_fields + NEW_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    # Validation audit.
    audit = []

    def add(section, cohort, statistic, count, denominator=None, notes=""):
        pct = ""
        if denominator not in (None, 0):
            pct = round(100 * count / denominator, 1)
        audit.append({
            "section": section,
            "cohort": cohort,
            "statistic": statistic,
            "count": count,
            "denominator": "" if denominator is None else denominator,
            "percent": pct,
            "notes": notes,
        })

    # Structural integrity.
    original_unchanged = all(
        old.get(field, "") == new.get(field, "")
        for old, new in zip(rows, merged)
        for field in original_fields
    )
    keys = [(r.get("cohort", ""), norm_id(r.get("hhidpn", ""))) for r in merged]

    add("structural", "Combined", "input_rows", len(rows), len(rows))
    add("structural", "Combined", "output_rows", len(merged), len(rows))
    add("structural", "Combined", "unique_participant_keys", len(set(keys)), len(merged))
    add("structural", "Combined", "original_columns_retained", len(original_fields), len(original_fields))
    add("structural", "Combined", "new_columns_added", len(NEW_COLUMNS), len(NEW_COLUMNS), ", ".join(NEW_COLUMNS))
    add("structural", "Combined", "all_original_values_unchanged", int(original_unchanged), 1)

    # Availability + score distributions.
    for cohort in ("A", "B", "Combined"):
        z = merged if cohort == "Combined" else [r for r in merged if r["cohort"] == cohort]
        n = len(z)

        for col in NEW_COLUMNS:
            vals = [int(r[col]) for r in z if r[col] != ""]
            add("availability", cohort, f"valid_{col}", len(vals), n)
            if vals:
                add("distribution", cohort, f"{col}_score_0", vals.count(0), len(vals))
                add("distribution", cohort, f"{col}_score_1", vals.count(1), len(vals))
                add("distribution", cohort, f"{col}_score_2", vals.count(2), len(vals))
                add("distribution", cohort, f"{col}_score_3", vals.count(3), len(vals))
                add("distribution", cohort, f"{col}_score_4", vals.count(4), len(vals))
                add("distribution", cohort, f"{col}_score_5", vals.count(5), len(vals))
                add("distribution", cohort, f"{col}_score_6", vals.count(6), len(vals))
                add("distribution", cohort, f"{col}_score_7", vals.count(7), len(vals))

        both = sum(r["cesd7_T1"] != "" and r["cesd7_T2"] != "" for r in z)
        add("availability", cohort, "valid_cesd7_T1_and_T2", both, n)

    # Base loneliness sample availability, useful for the planned analysis.
    def as_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None

    for cohort in ("A", "B", "Combined"):
        z0 = merged if cohort == "Combined" else [r for r in merged if r["cohort"] == cohort]
        z = [
            r for r in z0
            if as_float(r.get("T1 loneliness", "")) is not None
            and as_float(r.get("T2 loneliness", "")) is not None
            and as_float(r.get("self_completed_loneliness_T1", "")) == 1
            and as_float(r.get("self_completed_loneliness_T2", "")) == 1
        ]
        n = len(z)
        t1 = sum(r["cesd7_T1"] != "" for r in z)
        t2 = sum(r["cesd7_T2"] != "" for r in z)
        both = sum(r["cesd7_T1"] != "" and r["cesd7_T2"] != "" for r in z)
        add("loneliness_base_sample", cohort, "base_N", n, n)
        add("loneliness_base_sample", cohort, "valid_cesd7_T1", t1, n)
        add("loneliness_base_sample", cohort, "valid_cesd7_T2", t2, n)
        add("loneliness_base_sample", cohort, "valid_cesd7_T1_and_T2", both, n)

    with AUDIT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "cohort", "statistic", "count", "denominator", "percent", "notes"],
        )
        writer.writeheader()
        writer.writerows(audit)

    # Compact console checks.
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(original_fields)} -> {len(original_fields) + 2}")
    print(f"Unique keys: {len(set(keys)):,}")
    print(f"Original values unchanged: {original_unchanged}")
    for cohort in ("A", "B", "Combined"):
        z = merged if cohort == "Combined" else [r for r in merged if r["cohort"] == cohort]
        n = len(z)
        t1 = sum(r["cesd7_T1"] != "" for r in z)
        t2 = sum(r["cesd7_T2"] != "" for r in z)
        both = sum(r["cesd7_T1"] != "" and r["cesd7_T2"] != "" for r in z)
        print(cohort, n, t1, t2, both)


if __name__ == "__main__":
    main()
