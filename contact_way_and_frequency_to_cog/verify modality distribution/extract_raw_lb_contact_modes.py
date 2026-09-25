#!/usr/bin/env python3
"""
Task 07 step 1: Extract relation-specific contact mode frequencies from raw HRS LB files.

Categories
----------
children:
    LB007 = has children
    LB009A = meet up
    LB009B = phone
    LB009C = write/email

other_relatives:
    LB011 = has other immediate family
    LB013A/B/C = meet up / phone / write-email

friends:
    LB015 = has friends
    LB017A/B/C = meet up / phone / write-email

Waves
-----
8=2006, 9=2008, 10=2010, 11=2012.

Raw HRS frequency coding
------------------------
1 = three or more times a week
2 = once or twice a week
3 = once or twice a month
4 = every few months
5 = once or twice a year
6 = less than once a year or never

Analysis score
--------------
Reverse-coded to:
5,4,3,2,1,0 respectively.

For friends/other relatives, respondents reporting no such relation group
are assigned 0 on all three mode scores, matching the prior binary-contact
logic. Respondents with no children remain missing for the children domain,
matching the prior cohort analysis.
"""
from pathlib import Path
import argparse, re
import numpy as np
import pandas as pd

WAVES = {
    8: ("H06LB_R.da", "H06LB_R.dct", "K", 2006),
    9: ("H08LB_R.da", "H08LB_R.dct", "L", 2008),
    10: ("H10LB_R.da", "H10LB_R.dct", "M", 2010),
    11: ("H12LB_R.da", "H12LB_R.dct", "N", 2012),
}
CATS = {
    "children": ("LB007", ["LB009A", "LB009B", "LB009C"]),
    "other_relatives": ("LB011", ["LB013A", "LB013B", "LB013C"]),
    "friends": ("LB015", ["LB017A", "LB017B", "LB017C"]),
}
MODES = ["inperson", "phone", "written_email"]
SCORE = {1:5.0, 2:4.0, 3:3.0, 4:2.0, 5:1.0, 6:0.0}

def norm_id(hhid, pn):
    s = (str(hhid).strip() + str(pn).strip()).strip()
    try:
        return str(int(s))
    except Exception:
        return s

def dct_specs(path, wanted):
    text = Path(path).read_text(errors="ignore")
    wanted_u = {x.upper() for x in wanted}
    specs = {}
    for line in text.splitlines():
        m = re.search(r"_column\((\d+)\)\s+\w+\s+(\w+)\s+%(\d+)", line, re.I)
        if not m:
            continue
        name = m.group(2).upper()
        if name in wanted_u:
            start = int(m.group(1)) - 1
            width = int(m.group(3))
            specs[name] = (start, start + width)
    missing = wanted_u - set(specs)
    if missing:
        raise ValueError(f"Missing variables in {path}: {sorted(missing)}")
    return specs

def raw_int(s):
    s = str(s).strip()
    if not s:
        return np.nan
    try:
        return int(s)
    except Exception:
        return np.nan

def extract_wave(folder, wave):
    da, dct, prefix, year = WAVES[wave]
    wanted = ["HHID", "PN"]
    for _, (has, items) in CATS.items():
        wanted.append(prefix + has)
        wanted.extend(prefix + x for x in items)
    specs = dct_specs(folder / dct, wanted)

    rows = []
    with open(folder / da, "r", errors="ignore") as f:
        for line in f:
            def val(name):
                s, e = specs[name.upper()]
                return line[s:e].strip()

            row = {
                "hhidpn_key": norm_id(val("HHID"), val("PN")),
                "wave": wave,
                "year": year,
            }

            for cat, (has, items) in CATS.items():
                hasv = raw_int(val(prefix + has))
                row[f"{cat}_has_group_raw"] = hasv
                for mode, item in zip(MODES, items):
                    rv = raw_int(val(prefix + item))
                    row[f"{cat}_{mode}_raw"] = rv
                    row[f"{cat}_{mode}_freqscore"] = SCORE.get(rv, np.nan)

                # Preserve the original cohort conventions.
                if cat in ("friends", "other_relatives") and hasv == 5:
                    for mode in MODES:
                        row[f"{cat}_{mode}_freqscore"] = 0.0
                if cat == "children" and hasv == 5:
                    for mode in MODES:
                        row[f"{cat}_{mode}_freqscore"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_lb_dir", type=Path)
    ap.add_argument("output_csv", type=Path)
    args = ap.parse_args()

    df = pd.concat(
        [extract_wave(args.raw_lb_dir, w) for w in WAVES],
        ignore_index=True
    )
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.output_csv}: {len(df):,} respondent-wave rows")

if __name__ == "__main__":
    main()
