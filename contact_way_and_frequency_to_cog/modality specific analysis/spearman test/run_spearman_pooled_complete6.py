#!/usr/bin/env python3
"""
Pooled A+B Spearman correlation analysis for HRS contact modalities.

For each relationship domain separately:
- Define a longitudinal complete-six sample requiring nonmissing T1 and T2
  in-person, phone, and written/email analysis scores.
- Pool Cohorts A and B by conceptual timepoint.
- Calculate the 3x3 Spearman correlation matrix at T1 and T2.
- Save rho matrices, p-value matrices, pairwise results, and heatmaps.

This stage does not require cognition or covariate completeness.
"""
from pathlib import Path
import argparse, csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RELATIONSHIPS = ["children", "other_relatives", "friends"]
MODALITIES = ["inperson", "phone", "written_email"]
TIMEPOINTS = ["T1", "T2"]
REL_LABEL = {"children":"Children","other_relatives":"Other relatives","friends":"Friends"}
MODE_LABEL = {"inperson":"In-person","phone":"Phone","written_email":"Written/email"}

def parse_float(v):
    s = "" if v is None else str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def save_matrix_csv(path, matrix, labels):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            w.writerow([label] + [f"{x:.9f}" for x in row])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contact_cohorts_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("spearman_pooled_complete6"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    needed = {"hhidpn_key", "cohort"}
    for rel in RELATIONSHIPS:
        for tp in TIMEPOINTS:
            for mode in MODALITIES:
                needed.add(f"{rel}_{mode}_freqscore_{tp}")

    rows = []
    with open(args.contact_cohorts_csv, "r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        missing = needed - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row in r:
            rows.append({k: row.get(k, "") for k in needed})

    long_rows = []
    sample_rows = []

    for rel in RELATIONSHIPS:
        six = [f"{rel}_{mode}_freqscore_{tp}"
               for tp in TIMEPOINTS for mode in MODALITIES]
        eligible = [row for row in rows
                    if all(parse_float(row[c]) is not None for c in six)]

        sample_rows.append({
            "relationship": rel,
            "cohort_A_n": sum(row["cohort"].strip()=="A" for row in eligible),
            "cohort_B_n": sum(row["cohort"].strip()=="B" for row in eligible),
            "pooled_n": len(eligible),
        })

        for tp in TIMEPOINTS:
            data = np.array([
                [parse_float(row[f"{rel}_{mode}_freqscore_{tp}"]) for mode in MODALITIES]
                for row in eligible
            ], dtype=float)

            rho_m = np.eye(3)
            p_m = np.zeros((3,3))

            for i in range(3):
                for j in range(i+1,3):
                    res = spearmanr(data[:,i], data[:,j])
                    rho, p = float(res.statistic), float(res.pvalue)
                    rho_m[i,j] = rho_m[j,i] = rho
                    p_m[i,j] = p_m[j,i] = p
                    long_rows.append({
                        "relationship": rel, "timepoint": tp,
                        "pooled_cohorts": "A+B", "n": len(eligible),
                        "modality_1": MODALITIES[i], "modality_2": MODALITIES[j],
                        "spearman_rho": rho, "p_value": p,
                    })

            labels = [MODE_LABEL[m] for m in MODALITIES]
            save_matrix_csv(args.output_dir/f"spearman_rho_matrix_{rel}_{tp}.csv",
                            rho_m, labels)
            save_matrix_csv(args.output_dir/f"spearman_p_matrix_{rel}_{tp}.csv",
                            p_m, labels)

            fig, ax = plt.subplots(figsize=(6.5,5.5))
            im = ax.imshow(rho_m, vmin=-1, vmax=1)
            ax.set_xticks(range(3)); ax.set_xticklabels(labels)
            ax.set_yticks(range(3)); ax.set_yticklabels(labels)
            ax.set_title(f"{REL_LABEL[rel]} — {tp} pooled A+B Spearman correlations")
            for ii in range(3):
                for jj in range(3):
                    ax.text(jj, ii, f"{rho_m[ii,jj]:.3f}",
                            ha="center", va="center")
            fig.colorbar(im, ax=ax, label="Spearman rho")
            fig.tight_layout()
            fig.savefig(args.output_dir/f"spearman_heatmap_{rel}_{tp}.png",
                        dpi=220, bbox_inches="tight")
            plt.close(fig)

    with open(args.output_dir/"spearman_pooled_sample_sizes.csv",
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader(); w.writerows(sample_rows)

    with open(args.output_dir/"spearman_pooled_pairwise_results.csv",
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader(); w.writerows(long_rows)

if __name__ == "__main__":
    main()
