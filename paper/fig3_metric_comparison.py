"""
Figure 3 — Config A vs Config B: Best-Epoch Metric Comparison (Bar Chart)
Outputs:  fig3_metric_comparison.pdf  (IEEE double-column ready)

Data sourced directly from Phase 3b report, Section 7 summary table.
Run: python fig3_metric_comparison.py
Requires: matplotlib, numpy
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data from Phase 3b report, Table in Section 7 ────────────────────────────
# Config A best epoch = 78, Config B best epoch = 59
metrics = [
    "Det\nmAP@50",
    "Det\nmAP@50:95",
    "Seg\nmAP@50",
    "Seg\nmAP@50:95",
    "Precision",
    "Recall",
    "F1 Score",
]

config_a_vals = [97.16, 84.48, 97.00, 79.29, 94.18, 93.65, 93.11]
config_b_vals = [97.14, 85.00, 97.08, 79.46, 94.53, 93.47, 93.12]

# Winner per metric (B wins 5/7, A wins recall, det mAP@50 is tie)
# A wins: Recall (index 5); tie: Det mAP@50 (index 0)
winners = ["tie", "B", "B", "B", "B", "A", "B"]

# ── Style ─────────────────────────────────────────────────────────────────────
COL_A = "#4c8cbf"
COL_B = "#d94f4f"
COL_A_LT = "#a8c8e8"
COL_B_LT = "#f0a8a8"
FS_TICK = 7.5
FS_LAB = 8.5
FS_VAL = 6.8
FS_WIN = 8
FS_LEG = 8

x = np.arange(len(metrics))
bar_w = 0.38  # wider bars for better readability

# Taller figure + more top margin so value labels + stars don't clip
fig, ax = plt.subplots(figsize=(7.2, 4.2))
fig.subplots_adjust(bottom=0.20, top=0.88)

bars_a = ax.bar(
    x - bar_w / 2,
    config_a_vals,
    bar_w,
    label=f"Config A — lr=1×10⁻⁴  (best ep. 78)",
    color=COL_A,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

bars_b = ax.bar(
    x + bar_w / 2,
    config_b_vals,
    bar_w,
    label=f"Config B — lr=5×10⁻⁵  (best ep. 59)",
    color=COL_B,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

# Value labels: stagger A (lower offset) and B (higher offset) vertically
# so they never sit at the same height and crowd each other.
# A labels: placed slightly lower, rotated to save horizontal space
# B labels: placed higher
for bar, val in zip(bars_a, config_a_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.10,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=FS_VAL,
        color="#1a4f7a",
        rotation=0,
    )

for bar, val in zip(bars_b, config_b_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.10,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=FS_VAL,
        color="#7a1a1a",
        rotation=0,
    )

# Winner star markers — placed well above the value labels to avoid overlap
for i, (w, a_val, b_val) in enumerate(zip(winners, config_a_vals, config_b_vals)):
    if w == "B":
        ax.text(
            i + bar_w / 2,
            b_val + 0.85,
            "★",
            ha="center",
            va="bottom",
            fontsize=FS_WIN,
            color="#b02020",
        )
    elif w == "A":
        ax.text(
            i - bar_w / 2,
            a_val + 0.85,
            "★",
            ha="center",
            va="bottom",
            fontsize=FS_WIN,
            color="#1a5c8a",
        )

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=FS_TICK)
ax.set_ylabel("Score (%)", fontsize=FS_LAB)
ax.set_title(
    "Config A vs Config B — Best-Epoch Metric Comparison (EMA Model)",
    fontsize=FS_LAB,
    fontweight="bold",
    pad=6,
)
ax.set_ylim(70, 101)  # extra headroom so stars at ~99+ don't clip
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.grid(axis="y", lw=0.4, alpha=0.5, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend with star explanation
star_a = mpatches.Patch(color=COL_A, label=f"Config A — lr=1×10⁻⁴  (best ep. 78)")
star_b = mpatches.Patch(color=COL_B, label=f"Config B — lr=5×10⁻⁵  (best ep. 59)")
star_note = mpatches.Patch(color="none", label="★ = winning configuration per metric")
ax.legend(
    handles=[star_a, star_b, star_note],
    loc="lower right",
    fontsize=FS_LEG,
    frameon=True,
    edgecolor="#cccccc",
    handlelength=1.0,
)

fig.savefig("fig3_metric_comparison.pdf", bbox_inches="tight", dpi=300, format="pdf")
fig.savefig("fig3_metric_comparison.png", bbox_inches="tight", dpi=300)
print("Saved: fig3_metric_comparison.pdf + .png")
