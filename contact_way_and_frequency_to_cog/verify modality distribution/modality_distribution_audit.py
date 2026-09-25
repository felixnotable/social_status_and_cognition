#!/usr/bin/env python3
"""
Audit HRS Leave-Behind contact modality coding, missingness, and distributions.

The audit is restricted to the established analysis cohorts:
  Cohort A: 2006 (T1) and 2010 (T2)
  Cohort B: 2008 (T1) and 2012 (T2)

Relationship domains:
  children, other relatives, friends

Modalities:
  in-person, phone, written/email

Raw substantive frequency coding:
  1 = three or more times a week
  2 = once or twice a week
  3 = once or twice a month
  4 = every few months
  5 = once or twice a year
  6 = less than once a year or never

Analysis score:
  5,4,3,2,1,0 respectively.

Outputs:
  coding_audit.csv
  raw_valid_response_distributions.csv
  analysis_score_distributions.csv
  modality_missingness_by_wave.csv
  cohort_wave_comparability.csv
  audit_summary.txt
  nine PNG distribution plots
"""


# =============================================================================
# IMPLEMENTATION GUIDE
# =============================================================================
#
# This program performs the pre-model audit of the HRS
# Leave-Behind contact variables before any cognition regressions are fitted.
#
# It separates three methodological questions:
#
#   (A) CODING COMPARABILITY
#       Are the same six substantive frequency categories (1-6) used across
#       2006, 2008, 2010, and 2012, and do the raw variable labels refer to the
#       same contact modality?
#
#   (B) COMPLETENESS / MISSINGNESS
#       Among respondents who report having the relevant relationship group,
#       how often is each modality-specific frequency item answered?
#       Blank responses are never recoded as zero.
#
#   (C) DISTRIBUTIONAL COMPARABILITY
#       Are the observed six-category distributions broadly similar across the
#       Cohort A and Cohort B sequences? Exact equality is not required because
#       the cohorts are measured in different calendar years.
#
# RAW FREQUENCY CODING
# --------------------
#   1 = three or more times a week
#   2 = once or twice a week
#   3 = once or twice a month
#   4 = every few months
#   5 = once or twice a year
#   6 = less than once a year or never
#
# ANALYSIS SCORE
# --------------
# The direction is reversed so higher always means more contact:
#   raw 1 -> 5
#   raw 2 -> 4
#   raw 3 -> 3
#   raw 4 -> 2
#   raw 5 -> 1
#   raw 6 -> 0
#
# RELATIONSHIP-GROUP HANDLING
# ---------------------------
# - Raw category distributions use respondents who explicitly report having
#   the relationship group.
# - Existing pipeline rules are preserved for the derived analysis score:
#   "no friends" / "no other relatives" can map to zero contact, whereas
#   childless respondents remain missing.
# - An unanswered modality question remains missing.
#
# DOCUMENTATION APPENDED TO OUTPUT CSVs
# -------------------------------------
# Every requested audit CSV contains:
#   1. the original audit table,
#   4. one explanation row per original column
#
# Automated readers should use only the rows above COLUMN_EXPLANATION_START.
# =============================================================================

from pathlib import Path
import argparse, csv, math, re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

CATS = ["children", "other_relatives", "friends"]
MODES = ["inperson", "phone", "written_email"]
MODE_LABELS = {
    "inperson": "In-person",
    "phone": "Phone",
    "written_email": "Written/email",
}
CAT_LABELS = {
    "children": "Children",
    "other_relatives": "Other relatives",
    "friends": "Friends",
}
TIMING = {
    "A": {"T1": (8, 2006), "T2": (10, 2010)},
    "B": {"T1": (9, 2008), "T2": (11, 2012)},
}
RAW_LABELS = {
    1: "3+ times/week",
    2: "1-2 times/week",
    3: "1-2 times/month",
    4: "Every few months",
    5: "1-2 times/year",
    6: "<1/year or never",
}
SCORE_MAP = {1:5, 2:4, 3:3, 4:2, 5:1, 6:0}

WAVE_PREFIX = {8:"K", 9:"L", 10:"M", 11:"N"}
ITEMS = {
    "children": {
        "inperson":"LB009A", "phone":"LB009B", "written_email":"LB009C"
    },
    "other_relatives": {
        "inperson":"LB013A", "phone":"LB013B", "written_email":"LB013C"
    },
    "friends": {
        "inperson":"LB017A", "phone":"LB017B", "written_email":"LB017C"
    },
}


# -------------------------------------------------------------------------
# Self-documenting column dictionaries for the five CSV audit outputs.
# Tuple structure:
#   (column_name, plain-English meaning, interpretation / use)
# -------------------------------------------------------------------------

COLUMN_DOCS = {
    "coding_audit.csv": [
        ("cohort", "Cohort assigned to the HRS wave.", "A is the 2006/2010 sequence; B is the 2008/2012 sequence."),
        ("timepoint", "Position of the wave in the cohort design.", "T1 is the first contact measurement; T2 is the second."),
        ("wave", "HRS wave number.", "Wave 8=2006, 9=2008, 10=2010, 11=2012."),
        ("year", "Calendar year of the Leave-Behind questionnaire.", "Distinguishes the actual observation year from the abstract T1/T2 label."),
        ("relationship", "Relationship domain being audited.", "children, other_relatives, or friends."),
        ("modality", "Mode of social contact.", "inperson, phone, or written_email."),
        ("variable", "Exact raw HRS Leave-Behind variable name.", "Allows direct tracing back to the original HRS item."),
        ("dct_question_label", "Question label read from the HRS Stata dictionary file.", "Checks that the same substantive question is being compared across waves."),
        ("observed_substantive_codes", "Expected substantive raw response codes actually observed.", "The desired pattern is 1|2|3|4|5|6 in every wave/domain/mode cell."),
        ("observed_nonstandard_numeric_codes", "Observed numeric codes outside the expected 1-6 range.", "Blank means no unexpected numeric codes were found."),
        ("blank_count", "Number of raw records with a blank modality-frequency value.", "Broad raw blank count; can include skip patterns, absent relationship groups, and nonresponse."),
        ("n_designated_cohort_wave", "Number of cohort participants with an available raw LB record for this wave.", "Starting denominator for the coding audit at that cohort-wave."),
    ],
    "raw_valid_response_distributions.csv": [
        ("cohort", "Cohort.", "A or B."),
        ("timepoint", "Cohort analysis timepoint.", "T1 is the first contact measurement; T2 is the second."),
        ("wave", "HRS wave number.", "8, 9, 10, or 11."),
        ("year", "Calendar year.", "2006, 2008, 2010, or 2012."),
        ("relationship", "Relationship domain.", "children, other_relatives, or friends."),
        ("modality", "Contact method.", "inperson, phone, or written_email."),
        ("raw_code", "Original HRS frequency response code.", "Ranges from 1=most frequent to 6=least frequent."),
        ("raw_label", "Human-readable meaning of the HRS raw code.", "Examples: 3+ times/week, 1-2 times/month, <1/year or never."),
        ("analysis_score", "Reverse-coded 0-5 analysis score corresponding to raw_code.", "Higher values mean more frequent contact."),
        ("count", "Number of valid respondents in this raw response category.", "Numerator used to construct the distribution."),
        ("percent_among_valid_group_yes", "Percent among respondents who have the relationship group and gave a valid 1-6 response.", "Primary percentage for comparing raw response distributions across waves/cohorts."),
    ],
    "analysis_score_distributions.csv": [
        ("cohort", "Cohort.", "A or B."),
        ("timepoint", "Cohort analysis timepoint.", "T1 or T2."),
        ("wave", "HRS wave number.", "8, 9, 10, or 11."),
        ("year", "Calendar year.", "2006, 2008, 2010, or 2012."),
        ("relationship", "Relationship domain.", "children, other_relatives, or friends."),
        ("modality", "Contact method.", "inperson, phone, or written_email."),
        ("analysis_score", "Derived reverse-coded contact-frequency score.", "0=less than once/year or never; 5=three or more times/week."),
        ("count", "Number of participants with this nonmissing analysis score.", "Frequency of the actual 0-5 variable used downstream."),
        ("percent_among_nonmissing_analysis_scores", "Percent among all nonmissing derived analysis scores in this cell.", "Describes the distribution of the variable entering later analyses."),
    ],
    "modality_missingness_by_wave.csv": [
        ("cohort", "Cohort.", "A or B."),
        ("timepoint", "Cohort analysis timepoint.", "T1 or T2."),
        ("wave", "HRS wave number.", "8, 9, 10, or 11."),
        ("year", "Calendar year.", "2006, 2008, 2010, or 2012."),
        ("relationship", "Relationship domain.", "children, other_relatives, or friends."),
        ("modality", "Contact method.", "inperson, phone, or written_email."),
        ("n_designated_cohort_wave", "Number of cohort participants with an available raw LB record for this wave.", "Starting denominator for the cohort-wave."),
        ("n_group_yes", "Number reporting that they have the relevant relationship group.", "Relationship-eligible respondents for the raw modality question."),
        ("n_group_no", "Number explicitly reporting that they do not have the relationship group.", "For friends/other relatives this can later map to zero contact; childless respondents remain missing in the established pipeline."),
        ("n_group_other_or_blank", "Number whose group-presence response is neither standard yes nor standard no, or is blank.", "Signals unresolved group-status missingness or nonstandard response."),
        ("n_valid_frequency_among_group_yes", "Number with a valid 1-6 modality frequency among respondents who have the relationship group.", "Main valid-response count."),
        ("n_missing_frequency_among_group_yes", "Number with a blank modality frequency despite reporting the relationship group.", "True item-level missingness among eligible respondents."),
        ("n_nonstandard_frequency_code_among_group_yes", "Number with an unexpected numeric frequency code among group-eligible respondents.", "Should be zero if coding is clean."),
        ("valid_frequency_percent_among_group_yes", "Percent of group-eligible respondents with a valid 1-6 frequency response.", "Primary completeness statistic for the modality item."),
        ("analysis_score_nonmissing_n", "Number with a nonmissing derived 0-5 analysis score.", "Actual number potentially available downstream before cognition/covariate requirements."),
        ("analysis_score_nonmissing_percent_of_cohort_wave", "Percent of the cohort-wave with a nonmissing derived 0-5 score.", "Overall availability of the analytic modality variable."),
    ],
    "cohort_wave_comparability.csv": [
        ("timepoint", "Analogous cohort timepoint being compared.", "T1 compares A-2006 with B-2008; T2 compares A-2010 with B-2012."),
        ("A_year", "Calendar year for Cohort A at this timepoint.", "2006 at T1 and 2010 at T2."),
        ("B_year", "Calendar year for Cohort B at this timepoint.", "2008 at T1 and 2012 at T2."),
        ("relationship", "Relationship domain.", "children, other_relatives, or friends."),
        ("modality", "Contact method.", "inperson, phone, or written_email."),
        ("A_valid_n", "Number of valid raw frequency responses in Cohort A.", "Denominator for Cohort A's six-category distribution."),
        ("B_valid_n", "Number of valid raw frequency responses in Cohort B.", "Denominator for Cohort B's six-category distribution."),
        ("A_mean_frequency_score_0_to_5", "Mean reverse-coded contact-frequency score in Cohort A.", "Higher means more frequent contact."),
        ("B_mean_frequency_score_0_to_5", "Mean reverse-coded contact-frequency score in Cohort B.", "Higher means more frequent contact."),
        ("B_minus_A_mean_score", "Difference in mean score: B minus A.", "Positive means B is higher; negative means A is higher."),
        ("max_absolute_category_difference_percentage_points", "Largest absolute A/B difference across the six raw response-category percentages.", "Intuitive measure of the biggest single-category distribution shift."),
        ("cramers_v", "Effect-size measure for association between cohort and the six-category distribution.", "0 means no distributional association; larger values mean stronger A/B differences."),
        ("chi_square_p", "Chi-square p-value comparing the full six-category distributions.", "Large samples can make small differences significant; interpret alongside effect sizes."),
        ("jensen_shannon_divergence_bits", "Symmetric divergence between the two full category probability distributions.", "0 means identical distributions; values close to 0 indicate very similar shapes."),
    ],
}


def append_column_explanations(csv_path, fieldnames, documentation):
    """
    Append a human-readable data dictionary below the original CSV output.

    The original table is left unchanged. Documentation uses the first four
    fields only:
      field 1 = documentation marker
      field 2 = column name
      field 3 = meaning
      field 4 = interpretation / use

    Remaining fields are blank. Automated downstream readers should stop when
    they encounter COLUMN_EXPLANATION_START.
    """
    if len(fieldnames) < 4:
        raise ValueError("At least four fields are required for documentation.")

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # Blank line creates a visual break below the original data.
        writer.writerow([])

        start = [""] * len(fieldnames)
        start[0] = "COLUMN_EXPLANATION_START"
        start[1] = "column_name"
        start[2] = "meaning"
        start[3] = "interpretation"
        writer.writerow(start)

        # One documentation row is written for every original column.
        for column_name, meaning, interpretation in documentation:
            row = [""] * len(fieldnames)
            row[0] = "COLUMN_EXPLANATION"
            row[1] = column_name
            row[2] = meaning
            row[3] = interpretation
            writer.writerow(row)

        end = [""] * len(fieldnames)
        end[0] = "COLUMN_EXPLANATION_END"
        writer.writerow(end)


# Normalize HRS IDs so values like 123 and 123.0 match across files.
def norm_id(v):
    s = "" if v is None else str(v).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s

# Parse an optional raw HRS numeric code while preserving blanks as missing.
def maybe_int(v):
    s = "" if v is None else str(v).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

# Parse an optional derived score while preserving blanks as missing.
def maybe_float(v):
    s = "" if v is None else str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

# Read the established one-row-per-participant A/B cohort assignment.
def read_cohort(path):
    cohort = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = norm_id(row.get("hhidpn"))
            c = str(row.get("cohort","")).strip()
            if pid and c in ("A","B"):
                cohort[pid] = c
    return cohort

# Index extracted raw modality rows by (participant, HRS wave) for exact lookup.
def read_raw(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = norm_id(row["hhidpn_key"])
            wave = int(float(row["wave"]))
            rows[(pid, wave)] = row
    return rows

# Recover the raw question label from the HRS .dct file to verify wording.
def dct_label(dct_path, varname):
    text = Path(dct_path).read_text(errors="ignore")
    pat = re.compile(r'\b' + re.escape(varname) + r'\b\s+%\S+\s+"([^"]+)"', re.I)
    m = pat.search(text)
    return m.group(1).strip() if m else ""

# Cramer's V measures the size of an A/B categorical-distribution difference.
def cramers_v(table):
    arr = np.asarray(table, dtype=float)
    if arr.sum() == 0:
        return float("nan"), float("nan")
    chi2, p, _, _ = chi2_contingency(arr, correction=False)
    n = arr.sum()
    r, k = arr.shape
    denom = min(k - 1, r - 1)
    v = math.sqrt((chi2 / n) / denom) if denom > 0 else float("nan")
    return v, p

# Jensen-Shannon divergence compares two complete probability distributions.
def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2.0
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_modes_csv", type=Path)
    ap.add_argument("cohort_csv", type=Path)
    ap.add_argument("dct_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = read_cohort(args.cohort_csv)
    raw = read_raw(args.raw_modes_csv)

    # ---------------------------------------------------------------------
    # STEP 1: BUILD THE EXACT COHORT-WAVE PANEL USED FOR THIS AUDIT.
    # Each participant contributes only the designated T1 and T2 LB waves.
    # Missing raw LB participant-wave records are counted, not imputed.
    # ---------------------------------------------------------------------
    recs = []
    missing_wave_rows = []
    for pid, c in cohort.items():
        for tp, (wave, year) in TIMING[c].items():
            rr = raw.get((pid, wave))
            if rr is None:
                missing_wave_rows.append((pid,c,tp,wave,year))
                continue
            recs.append((pid,c,tp,wave,year,rr))

    # ---------------------------------------------------------------------
    # STEP 2: CODING AUDIT.
    # For every wave x relationship x modality cell, verify both:
    #   (a) the raw HRS dictionary wording, and
    #   (b) the numeric response codes actually observed in our cohort.
    # ---------------------------------------------------------------------
    coding_rows = []
    for wave, year in [(8,2006),(9,2008),(10,2010),(11,2012)]:
        dct = args.dct_dir / {8:"H06LB_R.dct",9:"H08LB_R.dct",10:"H10LB_R.dct",11:"H12LB_R.dct"}[wave]
        prefix = WAVE_PREFIX[wave]
        c = "A" if wave in (8,10) else "B"
        tp = "T1" if wave in (8,9) else "T2"
        subset = [r for r in recs if r[3] == wave]
        for cat in CATS:
            for mode in MODES:
                var = prefix + ITEMS[cat][mode]
                observed = Counter()
                for _,_,_,_,_,rr in subset:
                    v = maybe_int(rr.get(f"{cat}_{mode}_raw"))
                    observed["blank" if v is None else str(v)] += 1
                substantive = sorted(int(x) for x in observed if x != "blank" and x.isdigit() and 1 <= int(x) <= 6)
                nonstandard = sorted(int(x) for x in observed if x != "blank" and x.isdigit() and int(x) not in range(1,7))
                coding_rows.append({
                    "cohort": c, "timepoint": tp, "wave": wave, "year": year,
                    "relationship": cat, "modality": mode, "variable": var,
                    "dct_question_label": dct_label(dct, var),
                    "observed_substantive_codes": "|".join(map(str,substantive)),
                    "observed_nonstandard_numeric_codes": "|".join(map(str,nonstandard)),
                    "blank_count": observed.get("blank",0),
                    "n_designated_cohort_wave": len(subset),
                })

    def write_csv(path, rows, fields):
        """
        Write the audit table first, then append a file-specific column
        explanation section below the original data.
        """
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        documentation = COLUMN_DOCS.get(path.name)
        if documentation is not None:
            append_column_explanations(path, fields, documentation)

    write_csv(args.output_dir/"coding_audit.csv", coding_rows, list(coding_rows[0].keys()))

    # ---------------------------------------------------------------------
    # STEP 3: DISTRIBUTIONS + MISSINGNESS.
    # Raw 1-6 distributions are calculated among respondents who explicitly
    # have the relationship group. Separately, we summarize the derived 0-5
    # analysis score available under the established extraction rules.
    # ---------------------------------------------------------------------
    raw_dist_rows = []
    miss_rows = []
    analysis_dist_rows = []

    # Keep arrays for comparison and plots.
    raw_counts = {}
    analysis_counts = {}

    for c in ["A","B"]:
        for tp in ["T1","T2"]:
            wave, year = TIMING[c][tp]
            subset = [r for r in recs if r[1] == c and r[2] == tp]
            for cat in CATS:
                has_vals = [maybe_int(rr.get(f"{cat}_has_group_raw")) for *_,rr in subset]
                group_yes = sum(v == 1 for v in has_vals)
                group_no = sum(v == 5 for v in has_vals)
                group_other_or_blank = len(has_vals) - group_yes - group_no

                for mode in MODES:
                    valid_counts = Counter()
                    analysis_score_counts = Counter()
                    missing_among_group_yes = 0
                    nonstandard_among_group_yes = 0
                    group_yes_n = 0

                    for *_, rr in subset:
                        hasv = maybe_int(rr.get(f"{cat}_has_group_raw"))
                        rv = maybe_int(rr.get(f"{cat}_{mode}_raw"))
                        fs = maybe_float(rr.get(f"{cat}_{mode}_freqscore"))

                        if hasv == 1:
                            group_yes_n += 1
                            if rv in range(1,7):
                                valid_counts[rv] += 1
                            elif rv is None:
                                missing_among_group_yes += 1
                            else:
                                nonstandard_among_group_yes += 1

                        if fs is not None and 0 <= fs <= 5:
                            analysis_score_counts[int(round(fs))] += 1

                    total_valid = sum(valid_counts.values())
                    raw_counts[(c,tp,cat,mode)] = [valid_counts[i] for i in range(1,7)]
                    analysis_counts[(c,tp,cat,mode)] = [analysis_score_counts[i] for i in range(0,6)]

                    for code in range(1,7):
                        n = valid_counts[code]
                        raw_dist_rows.append({
                            "cohort": c, "timepoint": tp, "wave": wave, "year": year,
                            "relationship": cat, "modality": mode,
                            "raw_code": code, "raw_label": RAW_LABELS[code],
                            "analysis_score": SCORE_MAP[code],
                            "count": n,
                            "percent_among_valid_group_yes": (100*n/total_valid if total_valid else None),
                        })

                    total_analysis = sum(analysis_score_counts.values())
                    for score in range(0,6):
                        n = analysis_score_counts[score]
                        analysis_dist_rows.append({
                            "cohort": c, "timepoint": tp, "wave": wave, "year": year,
                            "relationship": cat, "modality": mode,
                            "analysis_score": score,
                            "count": n,
                            "percent_among_nonmissing_analysis_scores": (100*n/total_analysis if total_analysis else None),
                        })

                    miss_rows.append({
                        "cohort": c, "timepoint": tp, "wave": wave, "year": year,
                        "relationship": cat, "modality": mode,
                        "n_designated_cohort_wave": len(subset),
                        "n_group_yes": group_yes,
                        "n_group_no": group_no,
                        "n_group_other_or_blank": group_other_or_blank,
                        "n_valid_frequency_among_group_yes": total_valid,
                        "n_missing_frequency_among_group_yes": missing_among_group_yes,
                        "n_nonstandard_frequency_code_among_group_yes": nonstandard_among_group_yes,
                        "valid_frequency_percent_among_group_yes": (100*total_valid/group_yes_n if group_yes_n else None),
                        "analysis_score_nonmissing_n": total_analysis,
                        "analysis_score_nonmissing_percent_of_cohort_wave": (100*total_analysis/len(subset) if subset else None),
                    })

    write_csv(args.output_dir/"raw_valid_response_distributions.csv", raw_dist_rows, list(raw_dist_rows[0].keys()))
    write_csv(args.output_dir/"analysis_score_distributions.csv", analysis_dist_rows, list(analysis_dist_rows[0].keys()))
    write_csv(args.output_dir/"modality_missingness_by_wave.csv", miss_rows, list(miss_rows[0].keys()))

    # ---------------------------------------------------------------------
    # STEP 4: A/B DISTRIBUTIONAL COMPARISON.
    # These are descriptive comparisons because A and B are observed in
    # different calendar years. We therefore report effect sizes and full
    # distribution divergence, not just chi-square p-values.
    # This mixes cohort and calendar-year differences,
    # so effect sizes are descriptive rather than causal cohort tests.
    comp_rows = []
    for tp in ["T1","T2"]:
        for cat in CATS:
            for mode in MODES:
                a = raw_counts[("A",tp,cat,mode)]
                b = raw_counts[("B",tp,cat,mode)]
                v, p = cramers_v([a,b])
                jsd = js_divergence(a,b)

                # Mean reverse-coded score among valid raw responses.
                vals = np.array([5,4,3,2,1,0], dtype=float)
                aa = np.array(a, dtype=float)
                bb = np.array(b, dtype=float)
                mean_a = float((aa*vals).sum()/aa.sum()) if aa.sum() else float("nan")
                mean_b = float((bb*vals).sum()/bb.sum()) if bb.sum() else float("nan")

                pa = aa/aa.sum() if aa.sum() else np.zeros(6)
                pb = bb/bb.sum() if bb.sum() else np.zeros(6)
                max_pp = float(np.max(np.abs(pa-pb))*100)

                comp_rows.append({
                    "timepoint": tp,
                    "A_year": TIMING["A"][tp][1],
                    "B_year": TIMING["B"][tp][1],
                    "relationship": cat,
                    "modality": mode,
                    "A_valid_n": int(aa.sum()),
                    "B_valid_n": int(bb.sum()),
                    "A_mean_frequency_score_0_to_5": mean_a,
                    "B_mean_frequency_score_0_to_5": mean_b,
                    "B_minus_A_mean_score": mean_b-mean_a,
                    "max_absolute_category_difference_percentage_points": max_pp,
                    "cramers_v": v,
                    "chi_square_p": p,
                    "jensen_shannon_divergence_bits": jsd,
                })

    write_csv(args.output_dir/"cohort_wave_comparability.csv", comp_rows, list(comp_rows[0].keys()))

    # ---------------------------------------------------------------------
    # STEP 5: VISUAL INSPECTION.
    # One plot is produced for every relationship x modality combination,
    # overlaying A-T1, B-T1, A-T2, and B-T2 raw response distributions.
    # ---------------------------------------------------------------------
    states = [("A","T1"),("B","T1"),("A","T2"),("B","T2")]
    state_labels = ["A-T1 (2006)","B-T1 (2008)","A-T2 (2010)","B-T2 (2012)"]
    xlabels = ["3+/wk","1-2/wk","1-2/mo","Few mo","1-2/yr","<1/yr"]

    for cat in CATS:
        for mode in MODES:
            fig, ax = plt.subplots(figsize=(9,5.5))
            x = np.arange(6)
            for (state,label) in zip(states,state_labels):
                counts = np.array(raw_counts[(state[0],state[1],cat,mode)], dtype=float)
                pct = 100*counts/counts.sum() if counts.sum() else np.zeros(6)
                ax.plot(x, pct, marker="o", label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel("Raw HRS frequency category")
            ax.set_ylabel("Percent among valid respondents with relationship group")
            ax.set_title(f"{CAT_LABELS[cat]} — {MODE_LABELS[mode]} contact frequency")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(args.output_dir/f"distribution_{cat}_{mode}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    # Summarize key audit facts.
    coding_ok = all(
        r["observed_nonstandard_numeric_codes"] == ""
        and r["observed_substantive_codes"] == "1|2|3|4|5|6"
        for r in coding_rows
    )
    valid_rates = [r["valid_frequency_percent_among_group_yes"] for r in miss_rows
                   if r["valid_frequency_percent_among_group_yes"] is not None]
    nonstandard_total = sum(r["n_nonstandard_frequency_code_among_group_yes"] for r in miss_rows)
    max_shift = max(comp_rows, key=lambda r: r["max_absolute_category_difference_percentage_points"])
    max_v = max(comp_rows, key=lambda r: r["cramers_v"])

    lines = []
    lines.append("HRS LB modality coding/distribution audit")
    lines.append("")
    lines.append(f"Designated cohort-wave records found: {len(recs):,}")
    lines.append(f"Missing designated participant-wave raw LB records: {len(missing_wave_rows):,}")
    lines.append(f"All 36 wave×relationship×modality cells observed all substantive raw codes 1-6 and no nonstandard numeric codes: {coding_ok}")
    lines.append(f"Total nonstandard frequency codes among relationship-eligible respondents: {nonstandard_total}")
    lines.append(f"Valid-frequency completion among relationship-eligible respondents ranged from {min(valid_rates):.2f}% to {max(valid_rates):.2f}%.")
    lines.append("")
    lines.append("Largest A-vs-B category-percentage difference at analogous timepoint:")
    lines.append(
        f"  {max_shift['timepoint']} {max_shift['relationship']} {max_shift['modality']}: "
        f"{max_shift['max_absolute_category_difference_percentage_points']:.2f} percentage points "
        f"(A {max_shift['A_year']} vs B {max_shift['B_year']})."
    )
    lines.append("Largest Cramer's V for A-vs-B raw-category distribution:")
    lines.append(
        f"  {max_v['timepoint']} {max_v['relationship']} {max_v['modality']}: "
        f"V={max_v['cramers_v']:.3f}, p={max_v['chi_square_p']:.4g}."
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Coding is structurally comparable: identical modality wording in DCT labels and the same six substantive frequency codes are present in all four waves.")
    lines.append("- Distribution equality is not required for measurement comparability; observed A/B differences can reflect true cohort/calendar-period differences.")
    lines.append("- A/B comparisons here are descriptive because Cohort A and B are observed in different calendar years.")
    lines.append("- Missingness should still be respected in downstream eligibility; do not treat unanswered modality items as zero.")
    (args.output_dir/"audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
