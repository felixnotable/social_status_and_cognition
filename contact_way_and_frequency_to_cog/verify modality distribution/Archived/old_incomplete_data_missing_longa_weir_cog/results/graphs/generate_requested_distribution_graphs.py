#!/usr/bin/env python3
from pathlib import Path
import argparse
import io
import numpy as np
import pandas as pd
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

DESIGNATED = {
    "A": {8, 10},
    "B": {9, 11},
}

WAVE_TO_YEAR = {8: 2006, 9: 2008, 10: 2010, 11: 2012}
YEAR_ORDER = [2006, 2008, 2010, 2012]


def read_documented_csv(path):
    lines = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("COLUMN_EXPLANATION_START"):
                break
            lines.append(line)
    text = "".join(lines).rstrip()
    return pd.read_csv(io.StringIO(text))


def norm_id(v):
    s = "" if v is None else str(v).strip()
    if s == "":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_designated_panel(raw_csv, cohort_csv):
    raw = pd.read_csv(raw_csv, low_memory=False)
    cohort = pd.read_csv(cohort_csv, low_memory=False)

    raw["hhidpn_key"] = raw["hhidpn_key"].map(norm_id)
    cohort["hhidpn"] = cohort["hhidpn"].map(norm_id)

    cohort = cohort[["hhidpn", "cohort"]].copy()
    cohort = cohort[cohort["cohort"].isin(["A", "B"])].drop_duplicates()

    d = raw.merge(cohort, left_on="hhidpn_key", right_on="hhidpn", how="inner")
    d = d[d.apply(lambda r: int(r["wave"]) in DESIGNATED.get(r["cohort"], set()), axis=1)].copy()
    d["year"] = d["wave"].map(WAVE_TO_YEAR)
    return d


def relation_valid_summary(panel):
    rows = []
    for rel in RELATIONSHIPS:
        has_col = f"{rel}_has_group_raw"
        mode_cols = [f"{rel}_{m}_freqscore" for m in MODALITIES]

        for year in YEAR_ORDER:
            g = panel[panel["year"] == year].copy()
            has_yes = pd.to_numeric(g[has_col], errors="coerce") == 1
            any_valid = g[mode_cols].notna().any(axis=1)
            denom = int((has_yes & any_valid).sum())

            row = {
                "year": year,
                "relationship": rel,
                "valid_frequency_relationship_n": denom,
            }

            for mode in MODALITIES:
                col = f"{rel}_{mode}_freqscore"
                row[f"{mode}_valid_n"] = int((has_yes & g[col].notna()).sum())
                row[f"{mode}_percent_of_relation_valid"] = (
                    100 * row[f"{mode}_valid_n"] / denom if denom > 0 else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_valid_frequency(df, outdir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for rel in RELATIONSHIPS:
        g = df[df["relationship"] == rel].sort_values("year")
        ax.plot(
            g["year"].astype(str),
            g["valid_frequency_relationship_n"],
            marker="o",
            linewidth=2,
            label=REL_LABEL[rel],
        )
    ax.set_xlabel("Wave / year")
    ax.set_ylabel("Number of respondents with valid relationship frequency")
    ax.set_title("Valid frequency counts by relationship across waves")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    save_fig(fig, outdir / "valid_frequency_by_relationship.png")


def plot_modality_percentages(df, outdir):
    for rel in RELATIONSHIPS:
        g = df[df["relationship"] == rel].sort_values("year")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for mode in MODALITIES:
            ax.plot(
                g["year"].astype(str),
                g[f"{mode}_percent_of_relation_valid"],
                marker="o",
                linewidth=2,
                label=MODE_LABEL[mode],
            )
        ax.set_xlabel("Wave / year")
        ax.set_ylabel("Percent of valid relationship frequency respondents")
        ax.set_title(f"{REL_LABEL[rel]}: modality valid-response percentages")
        ax.set_ylim(85, 101)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2)
        save_fig(fig, outdir / f"modality_percentage_{rel}.png")


def plot_analysis_score_lines(df, outdir):
    score_order = [0, 1, 2, 3, 4, 5]
    for rel in RELATIONSHIPS:
        for mode in MODALITIES:
            g = df[(df["relationship"] == rel) & (df["modality"] == mode)].copy()
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for score in score_order:
                z = g[g["analysis_score"] == score].copy().sort_values("year")
                ax.plot(
                    z["year"].astype(int).astype(str),
                    z["percent_among_nonmissing_analysis_scores"],
                    marker="o",
                    linewidth=2,
                    label=f"Score {score}",
                )
            ax.set_xlabel("Wave / year")
            ax.set_ylabel("Percent among nonmissing analysis scores")
            ax.set_title(
                f"{REL_LABEL[rel]} — {MODE_LABEL[mode]}: analysis score distribution across waves"
            )
            ax.legend(frameon=False, ncol=2)
            ax.grid(True, alpha=0.2)
            save_fig(fig, outdir / f"analysis_score_lines_{rel}_{mode}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_modes_csv", type=Path)
    ap.add_argument("cohort_csv", type=Path)
    ap.add_argument("analysis_score_distributions_csv", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("graphs"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_designated_panel(args.raw_modes_csv, args.cohort_csv)
    relation_summary = relation_valid_summary(panel)
    relation_summary.to_csv(args.output_dir / "relation_valid_summary.csv", index=False)

    score_dist = read_documented_csv(args.analysis_score_distributions_csv)

    plot_valid_frequency(relation_summary, args.output_dir)
    plot_modality_percentages(relation_summary, args.output_dir)
    plot_analysis_score_lines(score_dist, args.output_dir)

    readme = (
        "Graph definitions\n"
        "1) valid_frequency_by_relationship.png\n"
        "   Count = respondents with has_group_raw == 1 and at least one nonmissing modality score.\n\n"
        "2) modality_percentage_<relationship>.png\n"
        "   Denominator = valid_frequency_<relationship> from graph 1.\n"
        "   Numerator = respondents with nonmissing value in the named modality.\n\n"
        "3) analysis_score_lines_<relationship>_<modality>.png\n"
        "   Each line is one analysis score (0..5). Percentages come from analysis_score_distributions.csv.\n"
    )
    (args.output_dir / "GRAPH_DEFINITIONS.txt").write_text(readme, encoding="utf-8")
    print(f"Saved graphs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
