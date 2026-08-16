#!/usr/bin/env python3
"""
Build required fields and eligibility flags from each respondent. 

Required source files
---------------------
1. PredictedCognitionDementiaMeasures.zip
   -> Dementia_HRS_2000-2016_Basic_Release1_2m.dta
   Source for Cog.

2. H_HRS_d_stata.zip
   -> H_HRS_d.dta
   Source for loneliness, UCLA-3 components, living alone, and child contact.

3. randhrs1992_2022v1_STATA.zip
   -> randhrs1992_2022v1.dta
   Source for age, sex, education, wealth, race, Hispanic ethnicity,
   and marital status.

4. cogfinalimp_9522wide.dta
   Langa-Weir source for cog27, cogfunction, interview status, and proxy status.

5. raw_lb/
   H06LB_R.da/.dct -> wave 8
   H08LB_R.da/.dct -> wave 9
   H10LB_R.da/.dct -> wave 10
   H12LB_R.da/.dct -> wave 11
   H14LB_R.da/.dct -> wave 12

   These raw Leave-Behind files supply the completion mode, the wave-specific
   "who answered" item, and the friend/other-relative contact questions.

   These raw Leave-Behind files are used to reconstruct weekly contact with
   friends and other relatives. 

Cohort timing
-------------
A: T1=W8 (2006), T2=W10 (2010), T3=W11 (2012)
B: T1=W9 (2008), T2=W11 (2012), T3=W12 (2014)

Cohort membership
-----------------
A respondent belongs to cohort A if at least two of W8/W10/W12 have a
non-missing UCLA-3 loneliness score.

A respondent belongs to cohort B if at least two of W9/W11/W13 have a
non-missing UCLA-3 loneliness score.

The two groups are expected not to overlap.

Final output
------------
HRS_participant_level_cohorts_with_contact_transitions.csv

The output has one row per participant and 76 columns.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader


WAVE_YEAR = {
    8: 2006,
    9: 2008,
    10: 2010,
    11: 2012,
    12: 2014,
    13: 2016,
}

TIME_MAP = {
    "A": {"T1": 8, "T2": 10, "T3": 11},
    "B": {"T1": 9, "T2": 11, "T3": 12},
}

ELIGIBILITY_WAVES = {
    "A": (8, 10, 12),
    "B": (9, 11, 13),
}

SEX_MAP = {1: "Male", 2: "Female"}
RACE_MAP = {
    1: "White",
    2: "Black/African American",
    3: "Other",
}
MARITAL_MAP = {
    1: "Married",
    2: "Married, spouse absent",
    3: "Partnered",
    4: "Separated",
    5: "Divorced",
    6: "Separated/Divorced",
    7: "Widowed",
    8: "Never married",
}
COGFUNCTION_LABELS = {
    1: "Normal",
    2: "CIND",
    3: "Demented",
}

FINAL_COLUMNS = [
    "cohort",
    "hhidpn",
    "T1 loneliness",
    "T2 loneliness",
    "Loneliness T2-T1",
    "cog T1",
    "cog T2",
    "cog T3",
    "cog T3-T2",
    "sex",
    "education",
    "race",
    "hispanic",
    "race_ethnicity",
    "age_T1",
    "age_T2",
    "age_T3",
    "wealth_T1",
    "wealth_T2",
    "wealth_T3",
    "marital_status_T1",
    "marital_status_T2",
    "marital_status_T3",
    "lives_alone_T1",
    "lives_alone_T2",
    "lives_alone_T3",
    "lack_companionship_T1",
    "lack_companionship_T2",
    "lack_companionship_T3",
    "left_out_T1",
    "left_out_T2",
    "left_out_T3",
    "feels_isolated_T1",
    "feels_isolated_T2",
    "feels_isolated_T3",
    "cog27_T1",
    "cog27_T2",
    "cog27_T3",
    "cog27_T3_minus_T2",
    "cogfunction_T1",
    "cogfunction_T2",
    "cogfunction_T3",
    "cognitive_classification_T1",
    "cognitive_classification_T2",
    "cognitive_classification_T3",
    "who_answered_raw_T1",
    "who_answered_raw_T2",
    "who_answered_raw_T3",
    "questionnaire_completion_T1",
    "questionnaire_completion_T2",
    "psychosocial_completed_T1",
    "psychosocial_completed_T2",
    "self_completed_loneliness_T1",
    "self_completed_loneliness_T2",
    "has_Cog_T1",
    "has_Cog_T2",
    "has_Cog_T3",
    "has_cog27_T1",
    "has_cog27_T2",
    "has_cog27_T3",
    "eligible_basic_analysis",
    "eligible_cog27_sensitivity_analysis",
    "eligible_living_alone_model",
    "eligible_fully_adjusted_model",
    "weekly_contact_friends_T1",
    "weekly_contact_friends_T2",
    "weekly_contact_other_relatives_T1",
    "weekly_contact_other_relatives_T2",
    "weekly_contact_children_T1",
    "weekly_contact_children_T2",
    "friends_contact_transition",
    "eligible_friends_contact_model",
    "other_relatives_contact_transition",
    "eligible_other_relatives_contact_model",
    "children_contact_transition",
    "eligible_children_contact_model",
]


# ---------- small helpers ----------

def norm_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except (TypeError, ValueError):
        return s


def as_int(value):
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def valid_number(value) -> bool:
    try:
        return value is not None and not pd.isna(float(value))
    except (TypeError, ValueError):
        return False


def later_minus_earlier(later, earlier):
    # These change fields were originally calculated after the source values
    # had already been written to CSV.  Decimal arithmetic reproduces that
    # calculation without adding a binary floating-point tail.
    if not valid_number(later) or not valid_number(earlier):
        return np.nan
    try:
        return float(Decimal(str(later)) - Decimal(str(earlier)))
    except (InvalidOperation, ValueError):
        return np.nan


def csv_float32(value):
    """Match the short decimal representation written from a Stata float32."""
    if not valid_number(value):
        return np.nan
    return float(str(np.float32(value)))


def six_significant_digits(value):
    """Match the original wealth step, which wrote RAND wealth with format :g."""
    if not valid_number(value):
        return np.nan
    return float(f"{float(value):g}")


def extract_member(zip_path: Path, ending: str, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.lower().endswith(ending.lower())]
        if not matches:
            raise FileNotFoundError(f"Could not find {ending!r} inside {zip_path}")
        member = matches[0]
        target = work_dir / Path(member).name
        if not target.exists():
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    return target


def find_one(folder: Path, names):
    for name in names:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these files were found in "
        f"{folder}: {', '.join(names)}"
    )


def read_stata_columns_fast(path: Path, columns):
    """Read selected numeric columns from a very wide Stata file."""
    reader = StataReader(path, convert_categoricals=False)
    reader._ensure_open()
    try:
        data = np.memmap(
            path, mode="r", dtype=reader._dtype,
            offset=reader._data_location, shape=(reader._nobs,)
        )
        out = {}
        for column in columns:
            if column not in reader._varlist:
                raise KeyError(f"{column} is not present in {path.name}")
            i = reader._varlist.index(column)
            values = np.asarray(data[reader._dtype.names[i]])
            stata_type = reader._typlist[i]
            if stata_type in reader.VALID_RANGE:
                low, high = reader.VALID_RANGE[stata_type]
                missing = (values < low) | (values > high)
                if missing.any():
                    values = values.astype(float, copy=True)
                    values[missing] = np.nan
                else:
                    values = values.copy()
            else:
                values = values.copy()
            out[column] = values
        return pd.DataFrame(out)
    finally:
        reader.close()


# ---------- source 1: Predicted Cognition ----------

def read_predicted(predicted_dta: Path) -> pd.DataFrame:
    # Predicted Cognition/Dementia Measures:
    # hhidpn = respondent ID
    # wave   = HRS wave
    # Cog    = continuous latent cognition score
    df = read_stata_columns_fast(
        predicted_dta,
        ["hhidpn", "wave", "Cog"],
    )
    df["hhidpn"] = df["hhidpn"].map(norm_id)
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype("Int64")
    df = df[df["wave"].isin(range(8, 14))].copy()
    df["Cog"] = df["Cog"].map(csv_float32)
    return df


# ---------- source 2: Harmonized HRS ----------

def read_harmonized(hhrs_dta: Path) -> pd.DataFrame:
    # Harmonized HRS D names used in this project:
    # RwLNLYS3 = 3-item UCLA loneliness mean
    # RwCOMPLAC = lacks companionship
    # RwLEFTOUT = feels left out
    # RwISOLATE = feels isolated
    # HwLVALONE = lives alone
    # RwKCNT = at-least-weekly contact with children
    columns = ["hhidpn"]

    for w in range(8, 14):
        columns += [
            f"r{w}lnlys3",
            f"r{w}complac",
            f"r{w}leftout",
            f"r{w}isolate",
            f"h{w}lvalone",
            f"r{w}kcnt",
        ]

    wide = read_stata_columns_fast(
        hhrs_dta,
        columns,
    )
    wide["hhidpn"] = wide["hhidpn"].map(norm_id)

    records = []
    for row in wide.to_dict("records"):
        pid = row["hhidpn"]
        for w in range(8, 14):
            records.append({
                "hhidpn": pid,
                "wave": w,
                "loneliness3": csv_float32(row.get(f"r{w}lnlys3")),
                "lack_companionship": csv_float32(row.get(f"r{w}complac")),
                "left_out": csv_float32(row.get(f"r{w}leftout")),
                "feels_isolated": csv_float32(row.get(f"r{w}isolate")),
                "lives_alone": csv_float32(row.get(f"h{w}lvalone")),
                "weekly_contact_children": csv_float32(row.get(f"r{w}kcnt")),
            })

    return pd.DataFrame(records)


# ---------- source 3: RAND HRS ----------

def read_rand(rand_dta: Path) -> pd.DataFrame:
    # RAND HRS Longitudinal:
    # RAGENDER = sex
    # RAEDYRS  = years of education
    # RARACEM  = race
    # RAHISPAN = Hispanic ethnicity
    # RwAGEY_E = exact age
    # HwATOTW  = total household wealth
    # RwMSTAT  = marital status
    columns = [
        "hhidpn",
        "ragender",
        "raedyrs",
        "raracem",
        "rahispan",
    ]
    for w in range(8, 14):
        columns += [
            f"r{w}agey_e",
            f"h{w}atotw",
            f"r{w}mstat",
        ]

    wide = read_stata_columns_fast(
        rand_dta,
        columns,
    )
    wide["hhidpn"] = wide["hhidpn"].map(norm_id)

    records = []
    for row in wide.to_dict("records"):
        sex_code = as_int(row.get("ragender"))
        race_code = as_int(row.get("raracem"))
        hisp_code = as_int(row.get("rahispan"))

        sex = SEX_MAP.get(sex_code, np.nan)
        race = RACE_MAP.get(race_code, np.nan)

        if hisp_code == 1:
            hispanic = "Hispanic"
            race_ethnicity = "Hispanic"
        elif hisp_code == 0:
            hispanic = "Not Hispanic"
            race_ethnicity = f"{race}, non-Hispanic" if isinstance(race, str) else np.nan
        else:
            hispanic = np.nan
            race_ethnicity = np.nan

        for w in range(8, 14):
            mstat = MARITAL_MAP.get(as_int(row.get(f"r{w}mstat")), np.nan)
            records.append({
                "hhidpn": row["hhidpn"],
                "wave": w,
                "age": row.get(f"r{w}agey_e"),
                "wealth": six_significant_digits(row.get(f"h{w}atotw")),
                "marital_status": mstat,
                "sex": sex,
                "education": row.get("raedyrs"),
                "race": race,
                "hispanic": hispanic,
                "race_ethnicity": race_ethnicity,
            })

    return pd.DataFrame(records)


# ---------- respondent-wave merge ----------

def build_long(predicted_dta: Path, hhrs_dta: Path, rand_dta: Path) -> pd.DataFrame:
    """
    merge data from cognition + social + rand
    """
    cognition = read_predicted(predicted_dta)
    social = read_harmonized(hhrs_dta)
    rand = read_rand(rand_dta)

    long = cognition.merge(
        social,
        on=["hhidpn", "wave"],
        how="left",
        validate="one_to_one",
    )
    long = long.merge(
        rand,
        on=["hhidpn", "wave"],
        how="left",
        validate="one_to_one",
    )
    return long, social


# ---------- cohort assignment ----------

def cohort_members(long: pd.DataFrame):
    """
    return cohort to hhidpn mapping
    """
    lookup = long.set_index(["hhidpn", "wave"])

    all_ids = sorted(long["hhidpn"].dropna().unique(), key=lambda x: int(x))

    result = {}
    for cohort, waves in ELIGIBILITY_WAVES.items(): 
        #A: 8,10,11
        #B: 9,11,12
        keep = []
        for pid in all_ids:
            n = 0
            for w in waves:
                try:
                    value = lookup.loc[(pid, w), "loneliness3"]
                except KeyError:
                    value = np.nan
                if valid_number(value):
                    n += 1
            if n >= 2:
                keep.append(pid)
        # result[A]: list of respondents ids who have at least 2 valid loneliness3 value in waves 8,10,11.
        # result[B]: list of respondents ids who have at least 2 valid loneliness3 value in waves 9,11,12.
        # Keep those who have 1 missing loneliness data for missingness analysis
        # It's ok that a respondent has empty T1 or T2 loneliness, because we will filter them in the future. 
        result[cohort] = keep

    overlap = set(result["A"]) & set(result["B"])
    if overlap:
        raise ValueError(
            f"{len(overlap)} respondents qualify for both cohorts."
        )

    return result


# ---------- Leave-Behind respondent/completion fields ----------

LB_RESPONSE_FIELDS = {
    8:  ("H06LB_R.da", "H06LB_R.dct", "K", "KLB051"),
    9:  ("H08LB_R.da", "H08LB_R.dct", "L", "LLB051"),
    10: ("H10LB_R.da", "H10LB_R.dct", "M", "MLB051"),
    11: ("H12LB_R.da", "H12LB_R.dct", "N", "NLB085"),
    12: ("H14LB_R.da", "H14LB_R.dct", "O", "OLB077"),
}


def load_raw_lb_response_lookup(raw_lb_dir: Path):
    """
    get the hhidpn to who_answered and completion mode
    return dict: 
    key: hhidpn
    valude: dict of who_answered value and completion mode in each wave (total 10 columns)
    """
    by_id = {}
    for wave, (da_name, dct_name, prefix, who_var) in LB_RESPONSE_FIELDS.items():
        da_path = find_raw_lb_file(raw_lb_dir, da_name)
        dct_path = find_raw_lb_file(raw_lb_dir, dct_name)
        variables = ["HHID", "PN", f"{prefix}LBCOMP", who_var]
        specs = dct_positions(dct_path, variables)
        with da_path.open("r", encoding="latin1", errors="ignore") as fh:
            for line in fh:
                vals = {name: line[a:b].strip() for name, (a, b) in specs.items()}
                if not vals["HHID"] or not vals["PN"]:
                    continue
                try:
                    pid = str(int(vals["HHID"] + vals["PN"]))
                except ValueError:
                    continue
                row = by_id.setdefault(pid, {})
                row[f"lb_completion_mode_raw_w{wave}"] = (
                    as_int(vals[f"{prefix}LBCOMP"]) if vals[f"{prefix}LBCOMP"] else np.nan
                )
                row[f"who_answered_raw_w{wave}"] = (
                    as_int(vals[who_var]) if vals[who_var] else np.nan
                )
    return by_id


def psychosocial_completed(code):
    code = as_int(code)
    if code in {1, 2, 4}:
        return 1
    if code == 5:
        return 0
    return np.nan


def self_completed_loneliness(code, a, b, c):
    return int(
        as_int(code) in {1, 2}
        and all(valid_number(x) for x in (a, b, c))
    )


# ---------- Langa-Weir ----------

def load_langa_weir(dta_path: Path):
    # Langa-Weir:
    # cogtot27_impYYYY = imputed 27-point cognition score
    # cogfunctionYYYY  = 1 Normal, 2 CIND, 3 Demented
    # proxyYYYY        = proxy respondent indicator
    # interviewYYYY    = interview participation
    years = [2006, 2008, 2010, 2012, 2014]
    columns = ["hhid", "pn"]

    for year in years:
        columns += [
            f"interview{year}",
            f"proxy{year}",
            f"cogtot27_imp{year}",
            f"cogfunction{year}",
        ]

    lw = pd.read_stata(
        dta_path,
        columns=columns,
        convert_categoricals=False,
    )

    lw["hhidpn"] = (
        lw["hhid"].astype(str).str.strip()
        + lw["pn"].astype(str).str.strip()
    ).map(norm_id)

    return {
        r["hhidpn"]: r
        for r in lw.to_dict("records")
        if r["hhidpn"]
    }


# ---------- friends / other relatives from raw LB ----------

def find_raw_lb_file(folder: Path, expected_name: str) -> Path:
    exact = folder / expected_name
    if exact.exists():
        return exact
    stem = Path(expected_name).stem
    suffix = Path(expected_name).suffix
    matches = sorted(folder.glob(f"{stem}*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Missing {expected_name} in {folder}")
    raise FileNotFoundError(
        f"More than one file matches {expected_name}: "
        + ", ".join(str(x) for x in matches)
    )


LB_WAVE_FILES = {
    8: ("H06LB_R.da", "H06LB_R.dct", "K"),
    9: ("H08LB_R.da", "H08LB_R.dct", "L"),
    10: ("H10LB_R.da", "H10LB_R.dct", "M"),
    11: ("H12LB_R.da", "H12LB_R.dct", "N"),
}


def dct_positions(dct_path: Path, variables):
    specs = {}
    text = dct_path.read_text(errors="ignore")

    for line in text.splitlines():
        m = re.search(r"_column\((\d+)\)\s+\w+\s+(\w+)\s+%(\d+)", line)
        if m and m.group(2) in variables:
            start = int(m.group(1)) - 1
            width = int(m.group(3))
            specs[m.group(2)] = (start, start + width)

    missing = set(variables) - set(specs)
    if missing:
        raise ValueError(
            f"Missing variables in {dct_path.name}: {sorted(missing)}"
        )

    return specs


def derive_weekly(has_group, mode_values):
    has_group = as_int(has_group)
    modes = [as_int(v) for v in mode_values]

    # HRS frequency codes 1/2 are weekly or more often.
    if any(v in {1, 2} for v in modes):
        return 1

    valid_modes = [v for v in modes if v in {1, 2, 3, 4, 5, 6}]

    # No such relationship group is treated as no weekly contact.
    if has_group == 5:
        return 0

    if has_group == 1 and valid_modes and all(v in {3, 4, 5, 6} for v in valid_modes):
        return 0

    return np.nan


def load_relationship_contacts(raw_lb_dir: Path):
    # Friends:
    #   LB015 = has friends
    #   LB017A/B/C = in-person / telephone / written-email frequency
    #
    # Other relatives:
    #   LB011 = has other immediate family
    #   LB013A/B/C = in-person / telephone / written-email frequency
    wave_lookup = {}

    for wave, (da_name, dct_name, prefix) in LB_WAVE_FILES.items():
        variables = [
            "HHID",
            "PN",
            f"{prefix}LB011",
            f"{prefix}LB013A",
            f"{prefix}LB013B",
            f"{prefix}LB013C",
            f"{prefix}LB015",
            f"{prefix}LB017A",
            f"{prefix}LB017B",
            f"{prefix}LB017C",
        ]

        dct_path = find_raw_lb_file(raw_lb_dir, dct_name)
        da_path = find_raw_lb_file(raw_lb_dir, da_name)
        pos = dct_positions(dct_path, variables)
        records = {}

        with da_path.open(
            "r",
            encoding="latin1",
            errors="ignore",
        ) as fh:
            for line in fh:
                vals = {
                    name: line[a:b].strip()
                    for name, (a, b) in pos.items()
                }

                try:
                    pid = str(
                        int(
                            vals["HHID"].zfill(6)
                            + vals["PN"].zfill(3)
                        )
                    )
                except ValueError:
                    continue

                records[pid] = {
                    "friends": derive_weekly(
                        vals[f"{prefix}LB015"],
                        [
                            vals[f"{prefix}LB017A"],
                            vals[f"{prefix}LB017B"],
                            vals[f"{prefix}LB017C"],
                        ],
                    ),
                    "other_relatives": derive_weekly(
                        vals[f"{prefix}LB011"],
                        [
                            vals[f"{prefix}LB013A"],
                            vals[f"{prefix}LB013B"],
                            vals[f"{prefix}LB013C"],
                        ],
                    ),
                }

        wave_lookup[wave] = records

    return wave_lookup


# ---------- final participant rows ----------

def build_rows(long, social, members, lb_lookup, lw_lookup, contact_lookup):
    long_lookup = {
        (r["hhidpn"], int(r["wave"])): r
        for r in long.to_dict("records")
    }
    social_lookup = {
        (r["hhidpn"], int(r["wave"])): r
        for r in social.to_dict("records")
    }

    rows = []
    # extract data for cohort A and B
    for cohort in ("A", "B"):
        for pid in members[cohort]:
            out = {"cohort": cohort, "hhidpn": pid}

            # These fields entered the original participant table from the T2
            # respondent-wave row.  Keep that rule so missingness is unchanged.
            static_wave = TIME_MAP[cohort]["T2"]
            static = long_lookup.get((pid, static_wave), {})
            out["sex"] = static.get("sex", np.nan)
            out["education"] = static.get("education", np.nan)
            out["race"] = static.get("race", np.nan)
            out["hispanic"] = static.get("hispanic", np.nan)
            out["race_ethnicity"] = static.get("race_ethnicity", np.nan)

            lb = lb_lookup.get(pid, {})
            lw = lw_lookup.get(pid, {})

            for tp, wave in TIME_MAP[cohort].items():
                r = long_lookup.get((pid, wave), {})
                year = WAVE_YEAR[wave]

                out[f"age_{tp}"] = r.get("age", np.nan)
                out[f"wealth_{tp}"] = r.get("wealth", np.nan)
                out[f"marital_status_{tp}"] = r.get("marital_status", np.nan)
                out[f"lives_alone_{tp}"] = r.get("lives_alone", np.nan)

                out[f"lack_companionship_{tp}"] = r.get("lack_companionship", np.nan)
                out[f"left_out_{tp}"] = r.get("left_out", np.nan)
                out[f"feels_isolated_{tp}"] = r.get("feels_isolated", np.nan)

                out[f"cog {tp}"] = r.get("Cog", np.nan)
                out[f"has_Cog_{tp}"] = int(valid_number(out[f"cog {tp}"]))

                out[f"who_answered_raw_{tp}"] = lb.get(
                    f"who_answered_raw_w{wave}",
                    np.nan,
                )

                interview = as_int(lw.get(f"interview{year}"))
                proxy = as_int(lw.get(f"proxy{year}"))
                cog27 = lw.get(f"cogtot27_imp{year}", np.nan)
                cogfunction = lw.get(f"cogfunction{year}", np.nan)

                if interview != 1:
                    cog27 = np.nan
                    cogfunction = np.nan
                    proxy = None

                # Direct 27-point score is blank for proxy interviews.
                if proxy in {1, 2}:
                    cog27 = np.nan

                out[f"cog27_{tp}"] = cog27
                out[f"cogfunction_{tp}"] = cogfunction
                out[f"cognitive_classification_{tp}"] = (
                    COGFUNCTION_LABELS.get(as_int(cogfunction), np.nan)
                )
                out[f"has_cog27_{tp}"] = int(valid_number(cog27))

                if tp in {"T1", "T2"}:
                    completion = lb.get(
                        f"lb_completion_mode_raw_w{wave}",
                        np.nan,
                    )
                    out[f"questionnaire_completion_{tp}"] = completion
                    out[f"psychosocial_completed_{tp}"] = psychosocial_completed(
                        completion
                    )
                    out[f"self_completed_loneliness_{tp}"] = (
                        self_completed_loneliness(
                            completion,
                            out[f"lack_companionship_{tp}"],
                            out[f"left_out_{tp}"],
                            out[f"feels_isolated_{tp}"],
                        )
                    )

            out["T1 loneliness"] = long_lookup.get(
                (pid, TIME_MAP[cohort]["T1"]),
                {},
            ).get("loneliness3", np.nan)

            out["T2 loneliness"] = long_lookup.get(
                (pid, TIME_MAP[cohort]["T2"]),
                {},
            ).get("loneliness3", np.nan)

            out["Loneliness T2-T1"] = later_minus_earlier(
                out["T2 loneliness"],
                out["T1 loneliness"],
            )

            out["cog T3-T2"] = later_minus_earlier(
                out["cog T3"],
                out["cog T2"],
            )

            out["cog27_T3_minus_T2"] = later_minus_earlier(
                out["cog27_T3"],
                out["cog27_T2"],
            )

            # Basic analytic sample used by the current contact models.
            basic = (
                out["self_completed_loneliness_T1"] == 1
                and out["self_completed_loneliness_T2"] == 1
                and valid_number(out["T1 loneliness"])
                and valid_number(out["T2 loneliness"])
                and out["has_Cog_T2"] == 1
                and out["has_Cog_T3"] == 1
            )
            out["eligible_basic_analysis"] = int(basic)

            out["eligible_cog27_sensitivity_analysis"] = int(
                basic
                and out["has_cog27_T2"] == 1
                and out["has_cog27_T3"] == 1
            )

            out["eligible_living_alone_model"] = int(
                basic
                and valid_number(out["lives_alone_T2"])
            )

            core_covariates_complete = all(
                pd.notna(out.get(name))
                for name in [
                    "age_T2",
                    "sex",
                    "education",
                    "race_ethnicity",
                    "wealth_T2",
                    "marital_status_T2",
                ]
            )
            out["eligible_fully_adjusted_model"] = int(
                basic and core_covariates_complete
            )

            t1_wave = TIME_MAP[cohort]["T1"]
            t2_wave = TIME_MAP[cohort]["T2"]

            out["weekly_contact_friends_T1"] = contact_lookup.get(
                t1_wave, {}
            ).get(pid, {}).get("friends", np.nan)

            out["weekly_contact_friends_T2"] = contact_lookup.get(
                t2_wave, {}
            ).get(pid, {}).get("friends", np.nan)

            out["weekly_contact_other_relatives_T1"] = contact_lookup.get(
                t1_wave, {}
            ).get(pid, {}).get("other_relatives", np.nan)

            out["weekly_contact_other_relatives_T2"] = contact_lookup.get(
                t2_wave, {}
            ).get(pid, {}).get("other_relatives", np.nan)

            # Child contact comes from Harmonized HRS RwKCNT.
            # Special missing "no children" stays missing rather than becoming 0.
            out["weekly_contact_children_T1"] = social_lookup.get(
                (pid, t1_wave), {}
            ).get("weekly_contact_children", np.nan)

            out["weekly_contact_children_T2"] = social_lookup.get(
                (pid, t2_wave), {}
            ).get("weekly_contact_children", np.nan)

            for domain in ("friends", "other_relatives", "children"):
                t1 = out[f"weekly_contact_{domain}_T1"]
                t2 = out[f"weekly_contact_{domain}_T2"]

                a = as_int(t1)
                b = as_int(t2)

                if a in {0, 1} and b in {0, 1}:
                    transition = {
                        (1, 1): "maintained_frequent",
                        (0, 1): "increased_contact",
                        (1, 0): "decreased_contact",
                        (0, 0): "maintained_infrequent",
                    }[(a, b)]
                    contact_valid = True
                else:
                    transition = np.nan
                    contact_valid = False

                out[f"{domain}_contact_transition"] = transition
                out[f"eligible_{domain}_contact_model"] = int(
                    basic and contact_valid
                )

            rows.append(out)

    return pd.DataFrame(rows)


def prepare_sources(source_dir: Path, work_dir: Path):
    """
    parse and return the paths to HRS data
    """
    direct_predicted = source_dir / "Dementia_HRS_2000-2016_Basic_Release1_2m.dta"
    if direct_predicted.exists():
        predicted_dta = direct_predicted
    else:
        predicted_zip = find_one(
            source_dir,
            [
                "PredictedCognitionDementiaMeasures.zip",
                "Dementia_HRS_2000-2016_Basic_Release1_2m.zip",
            ],
        )
        if predicted_zip.name.startswith("Predicted"):
            nested = extract_member(
                predicted_zip,
                "Dementia_HRS_2000-2016_Basic_Release1_2m.zip",
                work_dir,
            )
            predicted_dta = extract_member(nested, ".dta", work_dir)
        else:
            predicted_dta = extract_member(predicted_zip, ".dta", work_dir)

    direct_hhrs = source_dir / "H_HRS_d.dta"
    if direct_hhrs.exists():
        hhrs_dta = direct_hhrs
    else:
        hhrs_zip = find_one(
            source_dir,
            [
                "H_HRS_d_stata.zip",
            ],
        )
        hhrs_dta = extract_member(hhrs_zip, "H_HRS_d.dta", work_dir)

    direct_rand = source_dir / "randhrs1992_2022v1.dta"
    if direct_rand.exists():
        rand_dta = direct_rand
    else:
        rand_zip = find_one(
            source_dir,
            [
                "randhrs1992_2022v1_STATA.zip",
                "randhrs1992_2022v1_STATA(1).zip",
            ],
        )
        rand_dta = extract_member(
            rand_zip,
            "randhrs1992_2022v1.dta",
            work_dir,
        )

    lw_dta = find_one(
        source_dir,
        [
            "cogfinalimp_9522wide.dta",
            "cogfinalimp_9522wide(1).dta",
        ],
    )

    raw_lb_dir = source_dir / "raw_lb"
    if not raw_lb_dir.exists():
        raise FileNotFoundError(
            f"Expected raw Leave-Behind files in {raw_lb_dir}"
        )

    return predicted_dta, hhrs_dta, rand_dta, lw_dta, raw_lb_dir


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the 76-column HRS participant analysis table."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Folder containing the raw/source HRS files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("hrs_work"),
        help="Folder used for extracted .dta files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "HRS_participant_level_cohorts_with_contact_transitions.csv"
        ),
    )
    args = parser.parse_args()
    # find the paths to HRS data
    (
        predicted_dta,
        hhrs_dta,
        rand_dta,
        lw_dta,
        raw_lb_dir,
    ) = prepare_sources(args.source_dir, args.work_dir)
    # merge data long = cognition + social + rand 
    long, social = build_long(
        predicted_dta,
        hhrs_dta,
        rand_dta,
    )
    # get cohort to hhidpn mapping
    members = cohort_members(long)
    # get the hhidpn to who_answered and completion mode. Each hhidpn has 10 additonal columns (5 waves * 2)
    lb_lookup = load_raw_lb_response_lookup(
        raw_lb_dir,
    )
    # get cogtot27_impYYYY, cogfunctionYYYY, proxyYYYY, interviewYYYY
    lw_lookup = load_langa_weir(
        lw_dta,
    )
    # get specific contact frequency 
    contact_lookup = load_relationship_contacts(
        raw_lb_dir,
    )

    final = build_rows(
        long,
        social,
        members,
        lb_lookup,
        lw_lookup,
        contact_lookup,
    )

    missing = [
        c for c in FINAL_COLUMNS
        if c not in final.columns
    ]
    if missing:
        raise ValueError(
            "Final table is missing columns: "
            + ", ".join(missing)
        )

    final = final[FINAL_COLUMNS].copy()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Wrote {args.output} "
        f"({len(final):,} rows x {final.shape[1]} columns)"
    )
    print(
        "Cohort A:",
        int((final["cohort"] == "A").sum()),
    )
    print(
        "Cohort B:",
        int((final["cohort"] == "B").sum()),
    )


if __name__ == "__main__":
    main()
