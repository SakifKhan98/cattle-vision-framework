"""
Figure 2 — RF-DETR-Seg Training Curves: Config A vs Config B
Outputs:  fig2_training_curves.pdf  (IEEE double-column ready)

Data sourced directly from Phase 3b report (Table 10.1 and 10.2).
Run: python fig2_training_curves.py
Requires: matplotlib, numpy
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data from Phase 3b report, Tables 10.1 and 10.2 (EMA model) ──────────────
# Columns: epoch, det_map50, det_map5095, seg_map50, seg_map5095

config_a = [
    (1, 96.68, 80.00, 96.49, 73.13),
    (2, 96.89, 81.04, 96.72, 74.51),
    (3, 96.97, 81.92, 96.85, 75.41),
    (4, 97.11, 82.57, 96.97, 76.03),
    (5, 97.07, 82.55, 96.96, 76.05),
    (10, 97.05, 82.82, 96.95, 76.76),
    (14, 97.16, 83.73, 97.00, 77.61),
    (18, 97.04, 84.06, 96.91, 77.77),
    (20, 97.02, 83.63, 96.84, 77.59),
    (30, 97.07, 83.35, 96.94, 77.85),
    (40, 97.01, 83.96, 96.86, 78.40),
    (50, 96.93, 83.95, 96.68, 78.41),
    (60, 96.95, 84.00, 96.70, 78.64),
    (70, 96.99, 84.03, 96.75, 78.99),
    (78, 96.93, 84.48, 96.67, 79.29),
    (80, 96.76, 83.96, 96.43, 79.02),
    (90, 96.72, 83.69, 96.45, 78.94),
    (100, 96.46, 83.63, 96.31, 78.78),
]

config_b = [
    (1, 96.30, 78.23, 96.05, 72.38),
    (2, 96.62, 79.15, 96.32, 72.69),
    (3, 96.66, 80.03, 96.44, 73.81),
    (4, 96.81, 79.91, 96.67, 74.08),
    (5, 96.86, 80.52, 96.66, 74.72),
    (10, 96.99, 83.52, 96.89, 76.78),
    (14, 97.14, 84.05, 97.08, 77.42),
    (18, 96.99, 83.66, 96.85, 77.49),
    (20, 97.10, 84.21, 97.04, 78.08),
    (30, 97.11, 84.43, 97.01, 78.30),
    (40, 96.95, 84.69, 96.86, 78.69),
    (50, 96.90, 84.57, 96.77, 79.00),
    (59, 97.06, 85.00, 96.86, 79.46),
    (60, 96.90, 84.59, 96.72, 79.02),
    (70, 96.81, 84.43, 96.63, 79.01),
    (78, 96.87, 84.46, 96.67, 79.22),
    (90, 96.78, 84.61, 96.57, 79.33),
    (100, 96.55, 84.25, 96.46, 79.11),
]

a = np.array(config_a)
b = np.array(config_b)

ep_a, det50_a, det5095_a, seg50_a, seg5095_a = (
    a[:, 0],
    a[:, 1],
    a[:, 2],
    a[:, 3],
    a[:, 4],
)
ep_b, det50_b, det5095_b, seg50_b, seg5095_b = (
    b[:, 0],
    b[:, 1],
    b[:, 2],
    b[:, 3],
    b[:, 4],
)

# ── IEEE style constants ──────────────────────────────────────────────────────
COL_A = "#1f77b4"  # blue  — Config A
COL_B = "#d62728"  # red   — Config B
ANNO_A = "#1a5c8a"
ANNO_B = "#8b1a1a"
FS_TICK = 7
FS_LAB = 8
FS_LEG = 7.5
FS_ANNO = 6.8
LW = 1.4
MARKER = 5

# ── Figure: 2 rows × 2 cols ───────────────────────────────────────────────────
# Taller figure + more hspace to give each subplot room so annotations
# do not collide with the subplot title above.
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4))
fig.subplots_adjust(hspace=0.52, wspace=0.32)

panels = [
    # (ax, x_a, y_a, x_b, y_b, ylabel, title, best_a, best_b, epoch_a, epoch_b)
    (
        axes[0, 0],
        ep_a,
        det50_a,
        ep_b,
        det50_b,
        "mAP@50 (%)",
        "(a) Detection mAP@50",
        97.16,
        97.14,
        14,
        14,
    ),
    (
        axes[0, 1],
        ep_a,
        det5095_a,
        ep_b,
        det5095_b,
        "mAP@50:95 (%)",
        "(b) Detection mAP@50:95",
        84.48,
        85.00,
        78,
        59,
    ),
    (
        axes[1, 0],
        ep_a,
        seg50_a,
        ep_b,
        seg50_b,
        "Mask mAP@50 (%)",
        "(c) Segmentation mAP@50",
        97.00,
        97.08,
        14,
        14,
    ),
    (
        axes[1, 1],
        ep_a,
        seg5095_a,
        ep_b,
        seg5095_b,
        "Mask mAP@50:95 (%)",
        "(d) Segmentation mAP@50:95",
        79.29,
        79.46,
        78,
        59,
    ),
]

# Per-panel custom annotation positions to guarantee no overlap with titles.
# Format: (text_x_a, text_y_a, text_x_b, text_y_b)
# Positive y offset = below the point; we place labels in the lower portion
# of each panel well away from the top title area.
anno_pos = {
    # Panel (a): both best epochs at ep 14, values are very close (97.16 vs 97.14)
    # Place A label to the left-below, B label to the right-below
    "(a) Detection mAP@50": (43, 97.05, 30, 96.76),
    # Panel (b): A best at ep 78 (84.48), B best at ep 59 (85.00)
    # Values sit in upper-mid range; place both labels in lower half of plot
    "(b) Detection mAP@50:95": (85, 82, 40, 82),
    # Panel (c): both best at ep 14; values very close (97.00 vs 97.08)
    "(c) Segmentation mAP@50": (50, 96.88, 30, 96.60),
    # Panel (d): A best at ep 78 (79.29), B best at ep 59 (79.46)
    "(d) Segmentation mAP@50:95": (85, 76, 40, 76),
}

for ax, xa, ya, xb, yb, ylabel, title, ba, bb, ea, eb in panels:
    ax.plot(
        xa,
        ya,
        color=COL_A,
        lw=LW,
        marker="o",
        ms=MARKER - 1,
        markevery=[-1],
        label="Config A (lr=1e-4)",
    )
    ax.plot(
        xb,
        yb,
        color=COL_B,
        lw=LW,
        marker="s",
        ms=MARKER - 1,
        markevery=[-1],
        linestyle="--",
        label="Config B (lr=5e-5)",
    )

    # Vertical dotted lines at best epochs
    ax.axvline(ea, color=ANNO_A, lw=0.8, ls=":", alpha=0.7)
    ax.axvline(eb, color=ANNO_B, lw=0.8, ls=":", alpha=0.7)

    # Per-panel annotation positions — no collision with titles
    tx_a, ty_a, tx_b, ty_b = anno_pos[title]

    ax.annotate(
        f"{ba:.2f}%\n(ep {ea})",
        xy=(ea, ba),
        xytext=(tx_a, ty_a),
        fontsize=FS_ANNO,
        color=ANNO_A,
        arrowprops=dict(arrowstyle="-", color=ANNO_A, lw=0.7, shrinkA=2, shrinkB=2),
    )
    ax.annotate(
        f"{bb:.2f}%\n(ep {eb})",
        xy=(eb, bb),
        xytext=(tx_b, ty_b),
        fontsize=FS_ANNO,
        color=ANNO_B,
        arrowprops=dict(arrowstyle="-", color=ANNO_B, lw=0.7, shrinkA=2, shrinkB=2),
    )

    ax.set_xlabel("Epoch", fontsize=FS_LAB)
    ax.set_ylabel(ylabel, fontsize=FS_LAB)
    ax.set_title(title, fontsize=FS_LAB, fontweight="bold", pad=4)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis="y", lw=0.4, alpha=0.5, linestyle="--")
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Single shared legend below the figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    fontsize=FS_LEG,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
    edgecolor="#cccccc",
)

fig.savefig("fig2_training_curves.pdf", bbox_inches="tight", dpi=300, format="pdf")
fig.savefig("fig2_training_curves.png", bbox_inches="tight", dpi=300)
print("Saved: fig2_training_curves.pdf + .png")
