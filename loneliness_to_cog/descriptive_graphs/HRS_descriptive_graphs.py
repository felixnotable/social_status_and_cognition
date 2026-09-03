
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HRS_participant_level_cohorts_current.csv"
OUT_DIR = BASE_DIR / "hrs_descriptive_graphs"
OUT_DIR.mkdir(exist_ok=True)
PDF_PATH = BASE_DIR / "HRS_descriptive_graphs.pdf"

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("colorblind", 8)

T1_COLOR = PALETTE[0]
T2_COLOR = PALETTE[1]
A_ONLY_COLOR = PALETTE[2]
B_ONLY_COLOR = PALETTE[3]
TRANSITION_COLORS = {
    "maintained_frequent": PALETTE[0],
    "increased_contact": PALETTE[2],
    "decreased_contact": PALETTE[1],
    "maintained_infrequent": PALETTE[4],
}

def num(series):
    return pd.to_numeric(series, errors="coerce")

def one(series):
    return pd.to_numeric(series, errors="coerce") == 1

def valid_lon_t1(d):
    return num(d["T1 loneliness"]).notna() & one(d["self_completed_loneliness_T1"])

def valid_lon_t2(d):
    return num(d["T2 loneliness"]).notna() & one(d["self_completed_loneliness_T2"])

def valid_lon_pair(d):
    return valid_lon_t1(d) & valid_lon_t2(d)

def valid_cog_t2(d):
    return num(d["cog T2"]).notna() & one(d["has_Cog_T2"])

def valid_cog_t3(d):
    return num(d["cog T3"]).notna() & one(d["has_Cog_T3"])

def cohort_data(df, label):
    if label == "Combined":
        return df.copy()
    return df[df["cohort"] == label].copy()

def score_codes(series):
    # Convert allowed loneliness values like 1.6666667 into exact integer thirds.
    s = num(series).dropna()
    return np.rint(s * 3).astype(int)

def change_codes(t2_series, t1_series):
    # Same idea for change scores such as +/- 0.3333333 or +/- 0.6666667.
    delta = num(t2_series) - num(t1_series)
    delta = delta.dropna()
    return np.rint(delta * 3).astype(int)

def counts_by_code(code_series, code_ticks):
    vc = pd.Series(code_series).value_counts().sort_index()
    return np.array([int(vc.get(code, 0)) for code in code_ticks])

def hist_counts(series, bins):
    counts, _ = np.histogram(np.asarray(pd.Series(series).dropna()), bins=bins)
    return counts

def stat_text(label, values):
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return f"{label}: mean=NA, SD=NA, n=0"
    return f"{label}: mean={s.mean():.2f}, SD={s.std(ddof=1):.2f}, n={len(s)}"

def add_stats_box(ax, lines, x=0.02, y=0.98, fontsize=9):
    ax.text(
        x, y, "\n".join(lines),
        transform=ax.transAxes,
        ha="left", va="top", fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#bdbdbd"),
    )

def prettify_axes(ax):
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    sns.despine(ax=ax, top=True, right=True)

def savefig(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path

df = pd.read_csv(DATA_PATH)

A = cohort_data(df, "A")
B = cohort_data(df, "B")
C = cohort_data(df, "Combined")

# Integer-third scales for plotting loneliness and loneliness change.
lon_code_ticks = np.arange(3, 10)      # 3/3 ... 9/3 = 1.00 ... 3.00
change_code_ticks = np.arange(-6, 7)   # -6/3 ... 6/3 = -2.00 ... 2.00
lon_tick_labels = [f"{x/3:.2f}" for x in lon_code_ticks]
change_tick_labels = [f"{x/3:.2f}" for x in change_code_ticks]

A_t1 = num(A.loc[valid_lon_t1(A), "T1 loneliness"])
A_t2 = num(A.loc[valid_lon_t2(A), "T2 loneliness"])
B_t1 = num(B.loc[valid_lon_t1(B), "T1 loneliness"])
B_t2 = num(B.loc[valid_lon_t2(B), "T2 loneliness"])
C_t1 = num(C.loc[valid_lon_t1(C), "T1 loneliness"])
C_t2 = num(C.loc[valid_lon_t2(C), "T2 loneliness"])

A_t1_codes = score_codes(A_t1)
A_t2_codes = score_codes(A_t2)
B_t1_codes = score_codes(B_t1)
B_t2_codes = score_codes(B_t2)
C_t1_codes = score_codes(C_t1)
C_t2_codes = score_codes(C_t2)

A_pair = A.loc[valid_lon_pair(A)]
B_pair = B.loc[valid_lon_pair(B)]
C_pair = C.loc[valid_lon_pair(C)]

A_change_vals = num(A_pair["T2 loneliness"]) - num(A_pair["T1 loneliness"])
B_change_vals = num(B_pair["T2 loneliness"]) - num(B_pair["T1 loneliness"])
C_change_vals = num(C_pair["T2 loneliness"]) - num(C_pair["T1 loneliness"])

A_change_codes = np.rint(A_change_vals * 3).astype(int)
B_change_codes = np.rint(B_change_vals * 3).astype(int)
C_change_codes = np.rint(C_change_vals * 3).astype(int)

A_c2 = num(A.loc[valid_cog_t2(A), "cog T2"])
A_c3 = num(A.loc[valid_cog_t3(A), "cog T3"])
B_c2 = num(B.loc[valid_cog_t2(B), "cog T2"])
B_c3 = num(B.loc[valid_cog_t3(B), "cog T3"])
C_c2 = num(C.loc[valid_cog_t2(C), "cog T2"])
C_c3 = num(C.loc[valid_cog_t3(C), "cog T3"])

all_cog = pd.concat([A_c2, A_c3, B_c2, B_c3], ignore_index=True).dropna()
bins = np.linspace(all_cog.min(), all_cog.max(), 31)
centers = (bins[:-1] + bins[1:]) / 2

all_cog_combined = pd.concat([C_c2, C_c3], ignore_index=True).dropna()
bins_c = np.linspace(all_cog_combined.min(), all_cog_combined.max(), 31)
centers_c = (bins_c[:-1] + bins_c[1:]) / 2

plot_paths = []

time_legend_lon = [
    Line2D([0], [0], color=T1_COLOR, marker="o", linewidth=2.2, label="T1"),
    Line2D([0], [0], color=T2_COLOR, marker="s", linewidth=2.2, label="T2"),
]
time_legend_cog = [
    Line2D([0], [0], color=T1_COLOR, marker="o", linewidth=2.2, label="T2"),
    Line2D([0], [0], color=T2_COLOR, marker="s", linewidth=2.2, label="T3"),
]
cohort_legend = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="Cohort A"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Cohort B"),
]

# 1
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=lon_code_ticks, y=counts_by_code(A_t1_codes, lon_code_ticks), marker="o", linewidth=2.2, color=T1_COLOR, linestyle="-", ax=ax)
sns.lineplot(x=lon_code_ticks, y=counts_by_code(A_t2_codes, lon_code_ticks), marker="s", linewidth=2.2, color=T2_COLOR, linestyle="-", ax=ax)
sns.lineplot(x=lon_code_ticks, y=counts_by_code(B_t1_codes, lon_code_ticks), marker="o", linewidth=2.2, color=T1_COLOR, linestyle="--", ax=ax)
sns.lineplot(x=lon_code_ticks, y=counts_by_code(B_t2_codes, lon_code_ticks), marker="s", linewidth=2.2, color=T2_COLOR, linestyle="--", ax=ax)
ax.set_xticks(lon_code_ticks)
ax.set_xticklabels(lon_tick_labels)
ax.set_xlabel("Loneliness score")
ax.set_ylabel("Count")
prettify_axes(ax)
leg1 = ax.legend(handles=time_legend_lon, title="Time point", loc="upper right", frameon=True)
ax.add_artist(leg1)
ax.legend(handles=cohort_legend, title="Cohort", loc="upper center", frameon=True)
add_stats_box(ax, [stat_text("A T1", A_t1), stat_text("A T2", A_t2), stat_text("B T1", B_t1), stat_text("B T2", B_t2)])
plot_paths.append(savefig(fig, "loneliness_T1_T2_A_B.png"))

# 2
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=lon_code_ticks, y=counts_by_code(C_t1_codes, lon_code_ticks), marker="o", linewidth=2.2, color=T1_COLOR, label="T1", ax=ax)
sns.lineplot(x=lon_code_ticks, y=counts_by_code(C_t2_codes, lon_code_ticks), marker="s", linewidth=2.2, color=T2_COLOR, label="T2", ax=ax)
ax.set_xticks(lon_code_ticks)
ax.set_xticklabels(lon_tick_labels)
ax.set_xlabel("Loneliness score")
ax.set_ylabel("Count")
prettify_axes(ax)
ax.legend(title="Time point", frameon=True)
add_stats_box(ax, [stat_text("Combined T1", C_t1), stat_text("Combined T2", C_t2)])
plot_paths.append(savefig(fig, "loneliness_T1_T2_combined.png"))

# 3
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=change_code_ticks, y=counts_by_code(A_change_codes, change_code_ticks), marker="o", linewidth=2.2, color=A_ONLY_COLOR, linestyle="-", label="Cohort A", ax=ax)
sns.lineplot(x=change_code_ticks, y=counts_by_code(B_change_codes, change_code_ticks), marker="s", linewidth=2.2, color=B_ONLY_COLOR, linestyle="--", label="Cohort B", ax=ax)
ax.set_xticks(change_code_ticks)
ax.set_xticklabels(change_tick_labels, rotation=45)
ax.set_xlabel("Loneliness change (T2 - T1)")
ax.set_ylabel("Count")
prettify_axes(ax)
ax.legend(title="Cohort", frameon=True)
add_stats_box(ax, [stat_text("Cohort A", A_change_vals), stat_text("Cohort B", B_change_vals)])
plot_paths.append(savefig(fig, "loneliness_change_A_B.png"))

# 4
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=change_code_ticks, y=counts_by_code(C_change_codes, change_code_ticks), marker="o", linewidth=2.2, color=A_ONLY_COLOR, label="Combined", ax=ax)
ax.set_xticks(change_code_ticks)
ax.set_xticklabels(change_tick_labels, rotation=45)
ax.set_xlabel("Loneliness change (T2 - T1)")
ax.set_ylabel("Count")
prettify_axes(ax)
ax.legend(frameon=True)
add_stats_box(ax, [stat_text("Combined", C_change_vals)])
plot_paths.append(savefig(fig, "loneliness_change_combined.png"))

# 5
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=centers, y=hist_counts(A_c2, bins), marker="o", linewidth=2.2, color=T1_COLOR, linestyle="-", ax=ax)
sns.lineplot(x=centers, y=hist_counts(A_c3, bins), marker="s", linewidth=2.2, color=T2_COLOR, linestyle="-", ax=ax)
sns.lineplot(x=centers, y=hist_counts(B_c2, bins), marker="o", linewidth=2.2, color=T1_COLOR, linestyle="--", ax=ax)
sns.lineplot(x=centers, y=hist_counts(B_c3, bins), marker="s", linewidth=2.2, color=T2_COLOR, linestyle="--", ax=ax)
ax.set_xlabel("Cognition score")
ax.set_ylabel("Count")
prettify_axes(ax)
leg1 = ax.legend(handles=time_legend_cog, title="Time point", loc="upper right", frameon=True)
ax.add_artist(leg1)
ax.legend(handles=cohort_legend, title="Cohort", loc="upper center", frameon=True)
add_stats_box(ax, [stat_text("A T2", A_c2), stat_text("A T3", A_c3), stat_text("B T2", B_c2), stat_text("B T3", B_c3)])
plot_paths.append(savefig(fig, "cognition_T2_T3_A_B.png"))

# 6
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=centers_c, y=hist_counts(C_c2, bins_c), marker="o", linewidth=2.2, color=T1_COLOR, label="T2", ax=ax)
sns.lineplot(x=centers_c, y=hist_counts(C_c3, bins_c), marker="s", linewidth=2.2, color=T2_COLOR, label="T3", ax=ax)
ax.set_xlabel("Cognition score")
ax.set_ylabel("Count")
prettify_axes(ax)
ax.legend(title="Time point", frameon=True)
add_stats_box(ax, [stat_text("Combined T2", C_c2), stat_text("Combined T3", C_c3)])
plot_paths.append(savefig(fig, "cognition_T2_T3_combined.png"))

transitions = ["maintained_frequent", "increased_contact", "decreased_contact", "maintained_infrequent"]
domains = [
    ("Friends", "friends_contact_transition", "eligible_friends_contact_model"),
    ("Other relatives", "other_relatives_contact_transition", "eligible_other_relatives_contact_model"),
    ("Children", "children_contact_transition", "eligible_children_contact_model"),
]

# 7
fig, ax = plt.subplots(figsize=(13, 7.5))
fig.subplots_adjust(bottom=0.11, top=0.98)
group_centers = np.arange(len(domains)) * 6.0
within_offsets = np.array([-1.2, -0.4, 0.4, 1.2])
bar_width = 0.28
max_count = 0
for i, (domain_label, trans_col, elig_col) in enumerate(domains):
    A_elig = A[one(A[elig_col])]
    B_elig = B[one(B[elig_col])]
    for j, tr in enumerate(transitions):
        x_center = group_centers[i] + within_offsets[j]
        A_count = int((A_elig[trans_col] == tr).sum())
        B_count = int((B_elig[trans_col] == tr).sum())
        max_count = max(max_count, A_count, B_count)
        ax.bar(x_center - bar_width / 2, A_count, width=bar_width, color=TRANSITION_COLORS[tr], alpha=0.95, edgecolor="white", linewidth=0.8)
        ax.bar(x_center + bar_width / 2, B_count, width=bar_width, color=TRANSITION_COLORS[tr], alpha=0.45, edgecolor="white", linewidth=0.8)
        ax.text(x_center - bar_width / 2, A_count, str(A_count), ha="center", va="bottom", fontsize=8)
        ax.text(x_center + bar_width / 2, B_count, str(B_count), ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max_count * 1.32)
ax.set_xticks(group_centers)
ax.set_xticklabels([d[0] for d in domains])
ax.set_xlabel("Contact domain")
ax.set_ylabel("Count")
prettify_axes(ax)
transition_handles = [Patch(facecolor=TRANSITION_COLORS[tr], edgecolor="white", label=tr.replace("_", " ")) for tr in transitions]
cohort_handles_bar = [
    Patch(facecolor="#777777", edgecolor="white", alpha=0.95, label="Cohort A"),
    Patch(facecolor="#777777", edgecolor="white", alpha=0.45, label="Cohort B"),
]
leg1 = ax.legend(handles=transition_handles, title="Transition", loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True)
ax.add_artist(leg1)
ax.legend(handles=cohort_handles_bar, title="Cohort", loc="upper center", bbox_to_anchor=(0.5, 0.82), ncol=2, frameon=True)
plot_paths.append(savefig(fig, "contact_transition_A_B.png"))

# 8
fig, ax = plt.subplots(figsize=(12, 7))
group_centers = np.arange(len(domains)) * 6.0
within_offsets = np.array([-1.2, -0.4, 0.4, 1.2])
bar_width = 0.65
for i, (domain_label, trans_col, elig_col) in enumerate(domains):
    eligible = C[one(C[elig_col])]
    for j, tr in enumerate(transitions):
        x = group_centers[i] + within_offsets[j]
        count = int((eligible[trans_col] == tr).sum())
        ax.bar(x, count, width=bar_width, color=TRANSITION_COLORS[tr], alpha=0.9, edgecolor="white", linewidth=0.8)
        ax.text(x, count, str(count), ha="center", va="bottom", fontsize=8)
ax.set_xticks(group_centers)
ax.set_xticklabels([d[0] for d in domains])
ax.set_xlabel("Contact domain")
ax.set_ylabel("Count")
prettify_axes(ax)
ax.legend(handles=[Patch(facecolor=TRANSITION_COLORS[tr], edgecolor="white", label=tr.replace("_", " ")) for tr in transitions],
          title="Transition", loc="upper center", frameon=True)
plot_paths.append(savefig(fig, "contact_transition_combined.png"))

with PdfPages(PDF_PATH) as pdf:
    for path in plot_paths:
        img = plt.imread(path)
        fig = plt.figure(figsize=(11, 7))
        plt.imshow(img)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
