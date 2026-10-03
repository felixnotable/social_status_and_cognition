#!/usr/bin/env python3
"""
Distribution audit using a fixed longitudinal complete-six contact sample.

For each relationship domain separately, eligibility requires all six derived
contact-frequency scores to be nonmissing:
    T1 in-person, phone, written/email
    T2 in-person, phone, written/email

No cognition/covariate completeness is required at this stage.

Outputs:
    complete6_sample_sizes.csv
    complete6_eligibility_flags.csv
    relation_complete6_summary.csv
    analysis_score_distributions_complete6.csv
    analysis_score_summary_complete6.csv
    valid_frequency_by_relationship.png
    modality_percentage_<relationship>.png
    analysis_score_lines_<relationship>_<modality>.png
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RELATIONSHIPS = ["children", "other_relatives", "friends"]
MODALITIES = ["inperson", "phone", "written_email"]

REL_LABEL = {
    "children": "Children",
    "other_relatives": "Other relatives",
    "friends": "Friends",
}
MODE_LABEL = {
    "inperson": "In-person",
    "phone": "Phone",
    "written_email": "Written/email",
}
STATES = [
    ("A", "T1", 2006),
    ("B", "T1", 2008),
    ("A", "T2", 2010),
    ("B", "T2", 2012),
]
WAVE = {2006: 8, 2008: 9, 2010: 10, 2012: 11}

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contact_cohorts_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("distribution_audit_complete6"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.contact_cohorts_csv, low_memory=False)
    if df["hhidpn_key"].nunique() != len(df):
        raise ValueError("Expected one row per participant.")

    masks = {}
    flags = pd.DataFrame({"hhidpn_key": df["hhidpn_key"], "cohort": df["cohort"]})
    sample_rows = []

    for rel in RELATIONSHIPS:
        six_cols = [
            f"{rel}_{mode}_freqscore_{tp}"
            for tp in ["T1", "T2"]
            for mode in MODALITIES
        ]
        mask = df[six_cols].notna().all(axis=1)
        masks[rel] = mask
        flags[f"complete6_{rel}"] = mask.astype(int)

        for cohort in ["A", "B"]:
            sample_rows.append({
                "relationship": rel,
                "cohort": cohort,
                "complete6_n": int((mask & (df["cohort"] == cohort)).sum())
            })
        sample_rows.append({
            "relationship": rel,
            "cohort": "Combined",
            "complete6_n": int(mask.sum())
        })

    pd.DataFrame(sample_rows).to_csv(
        args.output_dir/"complete6_sample_sizes.csv", index=False
    )
    flags.to_csv(args.output_dir/"complete6_eligibility_flags.csv", index=False)

    rel_rows, score_rows, mean_rows = [], [], []

    for rel in RELATIONSHIPS:
        drel = df.loc[masks[rel]].copy()

        for cohort, tp, year in STATES:
            g = drel[drel["cohort"] == cohort].copy()
            fixed_n = len(g)
            row = {
                "cohort": cohort, "timepoint": tp, "wave": WAVE[year],
                "year": year, "relationship": rel,
                "complete6_relationship_n": fixed_n,
            }

            for mode in MODALITIES:
                col = f"{rel}_{mode}_freqscore_{tp}"
                vals = pd.to_numeric(g[col], errors="coerce")
                n = int(vals.notna().sum())
                if n != fixed_n:
                    raise ValueError(f"Unexpected missingness: {rel} {cohort}-{tp} {mode}")

                row[f"{mode}_valid_n"] = n
                row[f"{mode}_percent_of_complete6"] = 100.0 if fixed_n else np.nan

                counts = vals.value_counts()
                for score in range(6):
                    count = int(counts.get(score, 0))
                    score_rows.append({
                        "cohort": cohort, "timepoint": tp, "wave": WAVE[year],
                        "year": year, "relationship": rel, "modality": mode,
                        "analysis_score": score, "count": count,
                        "percent_among_complete6":
                            100*count/fixed_n if fixed_n else np.nan,
                    })

                mean_rows.append({
                    "cohort": cohort, "timepoint": tp, "year": year,
                    "relationship": rel, "modality": mode,
                    "n": n, "mean_score": vals.mean(), "sd_score": vals.std(ddof=1),
                    "median_score": vals.median(), "min_score": vals.min(),
                    "max_score": vals.max(),
                })

            rel_rows.append(row)

    rel_df = pd.DataFrame(rel_rows).sort_values(["year","relationship"])
    score_df = pd.DataFrame(score_rows).sort_values(
        ["year","relationship","modality","analysis_score"]
    )
    mean_df = pd.DataFrame(mean_rows)

    rel_df.to_csv(args.output_dir/"relation_complete6_summary.csv", index=False)
    score_df.to_csv(args.output_dir/"analysis_score_distributions_complete6.csv", index=False)
    mean_df.to_csv(args.output_dir/"analysis_score_summary_complete6.csv", index=False)

    fig, ax = plt.subplots(figsize=(9,5.5))
    for rel in RELATIONSHIPS:
        z = rel_df[rel_df.relationship == rel].sort_values("year")
        ax.plot(z.year.astype(str), z.complete6_relationship_n,
                marker="o", linewidth=2, label=REL_LABEL[rel])
    ax.set_xlabel("Wave / year")
    ax.set_ylabel("Number of respondents in complete-six contact sample")
    ax.set_title("Complete-six contact sample size by relationship")
    ax.legend(frameon=False); ax.grid(True, alpha=.2)
    save_fig(fig, args.output_dir/"valid_frequency_by_relationship.png")

    for rel in RELATIONSHIPS:
        z = rel_df[rel_df.relationship == rel].sort_values("year")
        fig, ax = plt.subplots(figsize=(9,5.5))
        for mode in MODALITIES:
            ax.plot(z.year.astype(str), z[f"{mode}_percent_of_complete6"],
                    marker="o", linewidth=2, label=MODE_LABEL[mode])
        ax.set_xlabel("Wave / year")
        ax.set_ylabel("Percent nonmissing in complete-six sample")
        ax.set_title(f"{REL_LABEL[rel]}: modality completeness in complete-six sample")
        ax.set_ylim(95,101); ax.legend(frameon=False); ax.grid(True, alpha=.2)
        save_fig(fig, args.output_dir/f"modality_percentage_{rel}.png")

    for rel in RELATIONSHIPS:
        for mode in MODALITIES:
            z = score_df[(score_df.relationship == rel) &
                         (score_df.modality == mode)].copy()
            fig, ax = plt.subplots(figsize=(9,5.5))
            for score in range(6):
                q = z[z.analysis_score == score].sort_values("year")
                ax.plot(q.year.astype(int).astype(str),
                        q.percent_among_complete6,
                        marker="o", linewidth=2, label=f"Score {score}")
            ax.set_xlabel("Wave / year")
            ax.set_ylabel("Percent of complete-six sample")
            ax.set_title(f"{REL_LABEL[rel]} — {MODE_LABEL[mode]}: "
                         "analysis score distribution across waves")
            ax.legend(frameon=False, ncol=2); ax.grid(True, alpha=.2)
            save_fig(fig, args.output_dir /
                     f"analysis_score_lines_{rel}_{mode}.png")

if __name__ == "__main__":
    main()
