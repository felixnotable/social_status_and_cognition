"""
Generate revised Step 4 descriptive graphs using Seaborn.

This version follows the professor's preferred Seaborn style:
    sns.set_theme(style="whitegrid", context="notebook")
    palette="colorblind"

Input
-----
HRS_participant_level_cohorts_with_contact_transitions.csv

Graph rules
-----------
1. Loneliness T1/T2:
   - treat UCLA-3 mean score as discrete
   - canonicalize scores to exact thirds
   - x-axis displays exact values:
     1.00, 1.33, 1.67, 2.00, 2.33, 2.67, 3.00
   - one line for T1 and one for T2

2. Cognition T2/T3:
   - continuous distributions
   - use seaborn KDE lines
   - one line for T2 and one for T3

3. Loneliness change:
   - canonicalize T1/T2 scores before subtraction
   - canonicalize changes to exact thirds
   - prevents floating-point duplicate categories
   - x-axis ranges from -2.00 to +2.00 in 1/3 increments

4. Contact transitions:
   - grouped seaborn barplot
   - domains: Friends, Other relatives, Children
   - four transition categories
   - domain-specific analysis-eligible samples

Outputs
-------
step4_descriptive_graphs_seaborn/*.png
HRS_step4_continuous_plot_summaries_seaborn.csv
HRS_step4_contact_transition_summaries_seaborn.csv
HRS_step4_descriptive_graphs_seaborn.pdf
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Paths
# ============================================================

INPUT = Path("HRS_participant_level_cohorts_with_contact_transitions.csv")

OUT_DIR = Path("step4_descriptive_graphs_seaborn")
OUT_DIR.mkdir(exist_ok=True)

CONT_SUMMARY = Path("HRS_step4_continuous_plot_summaries_seaborn.csv")
TRANS_SUMMARY = Path("HRS_step4_contact_transition_summaries_seaborn.csv")
PDF_OUT = Path("HRS_step4_descriptive_graphs_seaborn.pdf")


# ============================================================
# Seaborn theme
# ============================================================

sns.set_theme(style="whitegrid", context="notebook")


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)


# ============================================================
# Helper functions
# ============================================================

def num(series):
    return pd.to_numeric(series, errors="coerce")


def one(series):
    return pd.to_numeric(series, errors="coerce") == 1


def canonical_thirds(series):
    """
    Convert small floating-point differences in UCLA-3 means to exact thirds.

    Example:
        1.3333333 and 1.3333334 -> 1.333333333...
    """
    x = num(series)
    return np.round(x * 3) / 3


def valid_lon_t1(d):
    return (
        canonical_thirds(d["T1 loneliness"]).notna()
        & one(d["self_completed_loneliness_T1"])
    )


def valid_lon_t2(d):
    return (
        canonical_thirds(d["T2 loneliness"]).notna()
        & one(d["self_completed_loneliness_T2"])
    )


def valid_lon_pair(d):
    return valid_lon_t1(d) & valid_lon_t2(d)


def valid_cog_t2(d):
    return (
        one(d["has_Cog_T2"])
        & num(d["cog T2"]).notna()
    )


def valid_cog_t3(d):
    return (
        one(d["has_Cog_T3"])
        & num(d["cog T3"]).notna()
    )


def cohort_df(cohort):
    if cohort == "Combined":
        return df.copy()

    return df[df["cohort"] == cohort].copy()


def stat_row(values, cohort, variable, sample_definition):
    s = pd.Series(values).dropna()

    return {
        "cohort": cohort,
        "variable": variable,
        "sample_definition": sample_definition,
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
        "q1": float(s.quantile(0.25)) if len(s) else np.nan,
        "q3": float(s.quantile(0.75)) if len(s) else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
    }


def save(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# Setup
# ============================================================

COHORTS = [
    ("A", "Cohort A"),
    ("B", "Cohort B"),
    ("Combined", "Combined"),
]

# Seven exact UCLA-3 mean-score values.
LON_TICKS = np.array([
    1.00,
    1 + 1/3,
    1 + 2/3,
    2.00,
    2 + 1/3,
    2 + 2/3,
    3.00,
])

# Exact possible T2-T1 changes, in thirds.
CHANGE_TICKS = np.array([
    -2 + i/3
    for i in range(13)
])

continuous_rows = []
transition_rows = []
plot_paths = []


# ============================================================
# 1. Loneliness T1 vs T2
# ============================================================

for cohort_code, cohort_title in COHORTS:

    d = cohort_df(cohort_code)

    t1 = canonical_thirds(
        d.loc[
            valid_lon_t1(d),
            "T1 loneliness"
        ]
    )

    t2 = canonical_thirds(
        d.loc[
            valid_lon_t2(d),
            "T2 loneliness"
        ]
    )

    continuous_rows.append(
        stat_row(
            t1,
            cohort_code,
            "loneliness_T1",
            "valid self-completed T1 loneliness",
        )
    )

    continuous_rows.append(
        stat_row(
            t2,
            cohort_code,
            "loneliness_T2",
            "valid self-completed T2 loneliness",
        )
    )

    plot_df = pd.concat(
        [
            pd.DataFrame({
                "score": t1,
                "time": "T1",
            }),
            pd.DataFrame({
                "score": t2,
                "time": "T2",
            }),
        ],
        ignore_index=True,
    )

    plot_df = (
        plot_df
        .groupby(
            ["time", "score"],
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "count"}
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.lineplot(
        data=plot_df,
        x="score",
        y="count",
        hue="time",
        marker="o",
        palette="colorblind",
        ax=ax,
    )

    ax.set_xticks(LON_TICKS)

    ax.set_xticklabels(
        [
            f"{x:.2f}"
            for x in LON_TICKS
        ]
    )

    ax.set_title(
        f"{cohort_title}: "
        f"Loneliness at T1 and T2"
    )

    ax.set_xlabel(
        "Loneliness score"
    )

    ax.set_ylabel(
        "Count"
    )

    ax.text(
        0.02,
        0.98,
        (
            f"T1: n={len(t1)}, "
            f"mean={t1.mean():.2f}, "
            f"SD={t1.std(ddof=1):.2f}\n"
            f"T2: n={len(t2)}, "
            f"mean={t2.mean():.2f}, "
            f"SD={t2.std(ddof=1):.2f}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    plot_paths.append(
        save(
            fig,
            f"{cohort_code}_"
            f"loneliness_T1_T2_seaborn.png",
        )
    )


# ============================================================
# 2. Cognition T2 vs T3
# ============================================================

for cohort_code, cohort_title in COHORTS:

    d = cohort_df(cohort_code)

    t2 = num(
        d.loc[
            valid_cog_t2(d),
            "cog T2"
        ]
    )

    t3 = num(
        d.loc[
            valid_cog_t3(d),
            "cog T3"
        ]
    )

    continuous_rows.append(
        stat_row(
            t2,
            cohort_code,
            "cognition_T2",
            "valid T2 cognition",
        )
    )

    continuous_rows.append(
        stat_row(
            t3,
            cohort_code,
            "cognition_T3",
            "valid T3 cognition",
        )
    )

    plot_df = pd.concat(
        [
            pd.DataFrame({
                "cognition": t2,
                "time": "T2",
            }),
            pd.DataFrame({
                "cognition": t3,
                "time": "T3",
            }),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.kdeplot(
        data=plot_df,
        x="cognition",
        hue="time",
        common_norm=False,
        palette="colorblind",
        ax=ax,
    )

    ax.set_title(
        f"{cohort_title}: "
        f"Cognition at T2 and T3"
    )

    ax.set_xlabel(
        "Cognition score"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.text(
        0.02,
        0.98,
        (
            f"T2: n={len(t2)}, "
            f"mean={t2.mean():.2f}, "
            f"SD={t2.std(ddof=1):.2f}\n"
            f"T3: n={len(t3)}, "
            f"mean={t3.mean():.2f}, "
            f"SD={t3.std(ddof=1):.2f}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    plot_paths.append(
        save(
            fig,
            f"{cohort_code}_"
            f"cognition_T2_T3_seaborn.png",
        )
    )


# ============================================================
# 3. Loneliness change
# ============================================================

for cohort_code, cohort_title in COHORTS:

    d = cohort_df(cohort_code)

    pair = d.loc[
        valid_lon_pair(d)
    ].copy()

    # Canonicalize both component scores first.
    t1 = canonical_thirds(
        pair["T1 loneliness"]
    )

    t2 = canonical_thirds(
        pair["T2 loneliness"]
    )

    # Then canonicalize their difference as a second safeguard.
    delta = (
        np.round(
            (t2 - t1) * 3
        )
        / 3
    )

    continuous_rows.append(
        stat_row(
            delta,
            cohort_code,
            "loneliness_change_T2_minus_T1",
            (
                "valid self-completed "
                "T1 and T2 loneliness"
            ),
        )
    )

    plot_df = (
        pd.Series(delta)
        .value_counts()
        .sort_index()
        .rename_axis("change")
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.lineplot(
        data=plot_df,
        x="change",
        y="count",
        marker="o",
        ax=ax,
    )

    ax.set_xticks(
        CHANGE_TICKS
    )

    ax.set_xticklabels(
        [
            f"{x:.2f}"
            for x in CHANGE_TICKS
        ],
        rotation=45,
    )

    ax.set_title(
        f"{cohort_title}: "
        f"Loneliness Change (T2 - T1)"
    )

    ax.set_xlabel(
        "Loneliness change"
    )

    ax.set_ylabel(
        "Count"
    )

    ax.text(
        0.02,
        0.98,
        (
            f"n={len(delta)}, "
            f"mean={delta.mean():.2f}, "
            f"SD="
            f"{pd.Series(delta).std(ddof=1):.2f}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    plot_paths.append(
        save(
            fig,
            f"{cohort_code}_"
            f"loneliness_change_seaborn.png",
        )
    )


# ============================================================
# 4. Contact-transition grouped bar charts
# ============================================================

TRANSITIONS = [
    "maintained_frequent",
    "increased_contact",
    "decreased_contact",
    "maintained_infrequent",
]

DOMAINS = [
    (
        "Friends",
        "friends_contact_transition",
        "eligible_friends_contact_model",
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


for cohort_code, cohort_title in COHORTS:

    d = cohort_df(cohort_code)

    plot_rows = []

    for (
        domain,
        transition_col,
        eligibility_col,
    ) in DOMAINS:

        # Domain-specific final analysis sample.
        eligible = d[
            one(
                d[eligibility_col]
            )
        ].copy()

        denominator = len(
            eligible
        )

        for transition in TRANSITIONS:

            n = int(
                (
                    eligible[
                        transition_col
                    ]
                    == transition
                ).sum()
            )

            percent = (
                100 * n / denominator
                if denominator
                else np.nan
            )

            plot_rows.append({
                "domain": domain,
                "transition": transition,
                "count": n,
                "percent": percent,
                "denominator": denominator,
            })

            transition_rows.append({
                "cohort": cohort_code,
                "contact_domain": domain,
                "transition": transition,
                "n": n,
                "denominator_eligible_domain_sample":
                    denominator,
                "percent":
                    round(percent, 1),
                "sample_definition":
                    (
                        "domain-specific "
                        "analysis-eligible sample"
                    ),
            })

    plot_df = pd.DataFrame(
        plot_rows
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=plot_df,
        x="domain",
        y="count",
        hue="transition",
        hue_order=TRANSITIONS,
        palette="colorblind",
        ax=ax,
    )

    ax.set_title(
        f"{cohort_title}: "
        f"Contact-Transition Categories"
    )

    ax.set_xlabel(
        "Contact domain"
    )

    ax.set_ylabel(
        "Count"
    )

    # Label bars with N.
    for container in ax.containers:

        labels = [
            f"{int(bar.get_height())}"
            for bar in container
        ]

        ax.bar_label(
            container,
            labels=labels,
            fontsize=8,
            padding=2,
        )

    # Add the domain-specific denominator under each x label.
    domain_names = [
        "Friends",
        "Other relatives",
        "Children",
    ]

    denoms = (
        plot_df
        .groupby("domain")[
            "denominator"
        ]
        .first()
    )

    positions = np.arange(
        len(domain_names)
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        [
            f"{name}\n"
            f"(n={int(denoms[name])})"
            for name in domain_names
        ]
    )

    fig.tight_layout()

    plot_paths.append(
        save(
            fig,
            f"{cohort_code}_"
            f"contact_transition_bar_seaborn.png",
        )
    )


# ============================================================
# Save numerical summaries
# ============================================================

pd.DataFrame(
    continuous_rows
).to_csv(
    CONT_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    transition_rows
).to_csv(
    TRANS_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Combine figures into PDF
# ============================================================

with PdfPages(
    PDF_OUT
) as pdf:

    for path in plot_paths:

        image = plt.imread(
            path
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.imshow(
            image
        )

        ax.axis(
            "off"
        )

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


# ============================================================
# Finish
# ============================================================

print(
    "Created figures:"
)

for path in plot_paths:
    print(path)

print()
print(
    "Continuous summary:",
    CONT_SUMMARY,
)

print(
    "Transition summary:",
    TRANS_SUMMARY,
)

print(
    "Combined PDF:",
    PDF_OUT,
)
