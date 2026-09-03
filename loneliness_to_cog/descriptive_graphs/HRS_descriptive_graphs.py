"""Regenerate only the two contact-transition graphs after updating Friends.

The original descriptive-graph styling is retained. Friends uses the final
fully adjusted analytical sample; the Other relatives and Children bars keep
the original contact-model eligibility definitions and are therefore unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("colorblind", 8)

TRANSITIONS = [
    "maintained_frequent",
    "increased_contact",
    "decreased_contact",
    "maintained_infrequent",
]
TRANSITION_COLORS = {
    "maintained_frequent": PALETTE[0],
    "increased_contact": PALETTE[2],
    "decreased_contact": PALETTE[1],
    "maintained_infrequent": PALETTE[4],
}

# Only the Friends eligibility flag differs from the supplied script.
DOMAINS = [
    (
        "Friends",
        "friends_contact_transition",
        "eligible_friends_contact_fully_adjusted_model",
    ),
    (
        "Other relatives",
        "other_relatives_contact_transition",
        "eligible_other_relatives_contact_model",
    ),
    (
        "Children",
        "children_contact_transition",
        "eligible_children_contact_model",
    ),
]


def one(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") == 1


def prettify_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)


def expected_friends_sample_sizes(flow_path: Path) -> dict[str, int]:
    flow = pd.read_csv(flow_path, encoding="utf-8-sig")
    rows = flow.loc[
        (flow["contact_domain"] == "Friends")
        & (flow["stage_code"] == "06_core_covariates"),
        ["cohort", "remaining_n"],
    ]
    expected = dict(zip(rows["cohort"], rows["remaining_n"].astype(int)))
    required = {"A", "B", "Combined"}
    if set(expected) != required:
        raise ValueError(
            "The sample-flow CSV must contain Friends stage 06 rows for "
            "Cohort A, Cohort B, and Combined."
        )
    return expected


def analytical_data(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "Combined":
        return df.copy()
    return df.loc[df["cohort"] == cohort].copy()


def validate_friends_sample(
    df: pd.DataFrame, expected: dict[str, int]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for cohort in ("A", "B", "Combined"):
        d = analytical_data(df, cohort)
        eligible = d.loc[one(d["eligible_friends_contact_fully_adjusted_model"])]
        actual = len(eligible)
        if actual != expected[cohort]:
            raise ValueError(
                f"Friends {cohort} sample is {actual:,}, but the sample-flow "
                f"CSV specifies {expected[cohort]:,}."
            )
        for transition in TRANSITIONS:
            records.append(
                {
                    "contact_domain": "Friends",
                    "cohort": cohort,
                    "contact_transition": transition,
                    "count": int(
                        (eligible["friends_contact_transition"] == transition).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def make_cohort_graph(df: pd.DataFrame, output_path: Path) -> None:
    cohort_a = analytical_data(df, "A")
    cohort_b = analytical_data(df, "B")

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(bottom=0.11, top=0.98)
    group_centers = np.arange(len(DOMAINS)) * 6.0
    within_offsets = np.array([-1.2, -0.4, 0.4, 1.2])
    bar_width = 0.28
    max_count = 0

    for i, (_, transition_column, eligibility_column) in enumerate(DOMAINS):
        eligible_a = cohort_a.loc[one(cohort_a[eligibility_column])]
        eligible_b = cohort_b.loc[one(cohort_b[eligibility_column])]
        for j, transition in enumerate(TRANSITIONS):
            x_center = group_centers[i] + within_offsets[j]
            count_a = int((eligible_a[transition_column] == transition).sum())
            count_b = int((eligible_b[transition_column] == transition).sum())
            max_count = max(max_count, count_a, count_b)
            ax.bar(
                x_center - bar_width / 2,
                count_a,
                width=bar_width,
                color=TRANSITION_COLORS[transition],
                alpha=0.95,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.bar(
                x_center + bar_width / 2,
                count_b,
                width=bar_width,
                color=TRANSITION_COLORS[transition],
                alpha=0.45,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(
                x_center - bar_width / 2,
                count_a,
                str(count_a),
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                x_center + bar_width / 2,
                count_b,
                str(count_b),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, max_count * 1.32)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([domain[0] for domain in DOMAINS])
    ax.set_xlabel("Contact domain")
    ax.set_ylabel("Count")
    prettify_axes(ax)

    transition_handles = [
        Patch(
            facecolor=TRANSITION_COLORS[transition],
            edgecolor="white",
            label=transition.replace("_", " "),
        )
        for transition in TRANSITIONS
    ]
    cohort_handles = [
        Patch(facecolor="#777777", edgecolor="white", alpha=0.95, label="Cohort A"),
        Patch(facecolor="#777777", edgecolor="white", alpha=0.45, label="Cohort B"),
    ]
    first_legend = ax.legend(
        handles=transition_handles,
        title="Transition",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=cohort_handles,
        title="Cohort",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=2,
        frameon=True,
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_combined_graph(df: pd.DataFrame, output_path: Path) -> None:
    combined = analytical_data(df, "Combined")

    fig, ax = plt.subplots(figsize=(12, 7))
    group_centers = np.arange(len(DOMAINS)) * 6.0
    within_offsets = np.array([-1.2, -0.4, 0.4, 1.2])
    bar_width = 0.65

    for i, (_, transition_column, eligibility_column) in enumerate(DOMAINS):
        eligible = combined.loc[one(combined[eligibility_column])]
        for j, transition in enumerate(TRANSITIONS):
            x = group_centers[i] + within_offsets[j]
            count = int((eligible[transition_column] == transition).sum())
            ax.bar(
                x,
                count,
                width=bar_width,
                color=TRANSITION_COLORS[transition],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(x, count, str(count), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([domain[0] for domain in DOMAINS])
    ax.set_xlabel("Contact domain")
    ax.set_ylabel("Count")
    prettify_axes(ax)
    ax.legend(
        handles=[
            Patch(
                facecolor=TRANSITION_COLORS[transition],
                edgecolor="white",
                label=transition.replace("_", " "),
            )
            for transition in TRANSITIONS
        ],
        title="Transition",
        loc="upper center",
        frameon=True,
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=base_dir / "HRS_participant_level_cohorts_current.csv",
        help="Participant-level HRS analysis CSV.",
    )
    parser.add_argument(
        "--flow",
        type=Path,
        default=base_dir / "HRS_sample_flow_by_contact_domain(3).csv",
        help="Sample-flow CSV used to validate the final Friends sample sizes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output",
        help="Directory for the two regenerated graph files and count audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    expected = expected_friends_sample_sizes(args.flow)
    count_audit = validate_friends_sample(df, expected)
    count_audit.to_csv(args.output_dir / "friends_contact_transition_counts.csv", index=False)

    make_cohort_graph(df, args.output_dir / "contact_transition_A_B.png")
    make_combined_graph(df, args.output_dir / "contact_transition_combined.png")


if __name__ == "__main__":
    main()
