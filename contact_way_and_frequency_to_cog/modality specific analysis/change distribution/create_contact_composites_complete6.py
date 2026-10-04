#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RELATIONSHIPS = ["children", "other_relatives", "friends"]
MODALITIES = ["inperson", "phone", "written_email"]
LABELS = {"children":"Children","other_relatives":"Other relatives","friends":"Friends"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contact_cohorts_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("contact_composite_complete6"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.contact_cohorts_csv, low_memory=False)
    long_parts, summary_rows = [], []
    wide = df[["hhidpn_key", "cohort"]].copy()

    for rel in RELATIONSHIPS:
        six = [f"{rel}_{m}_freqscore_{tp}" for tp in ["T1","T2"] for m in MODALITIES]
        mask = df[six].notna().all(axis=1)

        t1 = df.loc[mask, [f"{rel}_{m}_freqscore_T1" for m in MODALITIES]].mean(axis=1)
        t2 = df.loc[mask, [f"{rel}_{m}_freqscore_T2" for m in MODALITIES]].mean(axis=1)
        change = t2 - t1

        part = pd.DataFrame({
            "hhidpn_key": df.loc[mask, "hhidpn_key"].values,
            "cohort": df.loc[mask, "cohort"].values,
            "relationship": rel,
            "overall_contact_T1": t1.values,
            "overall_contact_T2": t2.values,
            "overall_contact_change": change.values,
        })
        long_parts.append(part)

        wide[f"complete6_{rel}"] = mask.astype(int)
        wide[f"{rel}_overall_contact_T1"] = np.nan
        wide[f"{rel}_overall_contact_T2"] = np.nan
        wide[f"{rel}_overall_contact_change"] = np.nan
        wide.loc[mask, f"{rel}_overall_contact_T1"] = t1.values
        wide.loc[mask, f"{rel}_overall_contact_T2"] = t2.values
        wide.loc[mask, f"{rel}_overall_contact_change"] = change.values

        for measure, s in [
            ("overall_contact_T1", t1),
            ("overall_contact_T2", t2),
            ("overall_contact_change", change),
        ]:
            summary_rows.append({
                "relationship": rel, "measure": measure, "n": len(s),
                "mean": s.mean(), "sd": s.std(ddof=1), "median": s.median(),
                "q1": s.quantile(.25), "q3": s.quantile(.75),
                "min": s.min(), "max": s.max(),
            })

        for suffix, series, xlabel, title, bins in [
            ("T1", t1, "Overall contact score at T1", f"{LABELS[rel]} — T1 overall contact distribution", 25),
            ("T2", t2, "Overall contact score at T2", f"{LABELS[rel]} — T2 overall contact distribution", 25),
            ("change", change, "Overall contact change (T2 - T1)", f"{LABELS[rel]} — overall contact change distribution", 31),
        ]:
            fig, ax = plt.subplots(figsize=(8.5,5.5))
            ax.hist(series.dropna(), bins=bins)
            if suffix == "change":
                ax.axvline(0, linewidth=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Number of participants")
            ax.set_title(title)
            ax.grid(True, alpha=.2)
            fig.tight_layout()
            fname = f"composite_distribution_{rel}_{suffix}.png" if suffix != "change" else f"composite_change_distribution_{rel}.png"
            fig.savefig(args.output_dir/fname, dpi=220, bbox_inches="tight")
            plt.close(fig)

    pd.concat(long_parts, ignore_index=True).to_csv(
        args.output_dir/"contact_composites_complete6_long.csv", index=False
    )
    wide.to_csv(args.output_dir/"contact_composites_complete6_wide.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir/"contact_composite_summary_statistics.csv", index=False
    )

if __name__ == "__main__":
    main()
