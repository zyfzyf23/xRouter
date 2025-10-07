# -*- coding: utf-8 -*-
# Three bar graphs (horizontal subplots) ranking by ACCURACY but plotting COST UTILITY (Acc/Cost).
# Figure size ~ (18, 8). Axis labels are large & clear. Y-axis is log-scaled.
# Colors: soft Japanese-style pastels; one consistent color per model across all subplots.

import math
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Data from your tables
# -----------------------------
# Each entry: model -> {task: (accuracy, cost)}
# We include all tasks here; the "selected_tasks" below picks the 3 to plot.
data = {
    "GPT-4.1": {
        "LiveCodeBenchv5": (0.4480, 0.005953),
        "GPQADiamond":     (0.3384, 0.000568),
        "AIME25":          (0.4000, 0.025765),
        "MTBench":         (9.4465, 0.006725),
        "IFEval_m1":       (0.906,  0.000440),  # metric1 accuracy, cost
        "IFEval_m2":       (0.9389, 0.000440),  # metric2 accuracy, cost
        "LiveBench":       (57.2776, 10.782766),
    },
    "GPT-5-mini": {
        "LiveCodeBenchv5": (0.2832, 0.006656),
        "GPQADiamond":     (0.7677, 0.004484),
        "AIME25":          (0.8333, 0.011523),
        "MTBench":         (9.3031, 0.004810),
        "IFEval_m1":       (0.904,  0.000551),
        "IFEval_m2":       (0.9311, 0.000551),
        "LiveBench":       (78.5805, 6.576172),
    },
    "GPT-5": {
        "LiveCodeBenchv5": (0.1756, 0.049634),
        "GPQADiamond":     (0.8586, 0.033716),
        "AIME25":          (0.7333, 0.048160),
        "MTBench":         (9.3019, 0.033150),
        "IFEval_m1":       (0.948,  0.003693),
        "IFEval_m2":       (0.9662, 0.003693),
        "LiveBench":       (83.8888, 35.863581),
    },
    "GPT-OSS-20B": {
        "LiveCodeBenchv5": (0.3190, 0.000363),
        "GPQADiamond":     (0.4848, 0.000281),
        "AIME25":          (0.1333, 0.000405),
        "MTBench":         (8.6761, 0.000462),
        "IFEval_m1":       (0.786,  0.000038),
        "IFEval_m2":       (0.8309, 0.000038),
        "LiveBench":       (47.6848, 0.405241),
    },
    "Kimi-K2": {
        "LiveCodeBenchv5": (0.4480, 0.003089),
        "GPQADiamond":     (0.6313, 0.004225),
        "AIME25":          (0.2333, 0.005865),
        "MTBench":         (9.2438, 0.002626),
        "IFEval_m1":       (0.918,  0.000237),
        "IFEval_m2":       (0.9454, 0.000237),
        "LiveBench":       (57.7387, 3.783430),
    },
    "Deepseek-R1": {
        "LiveCodeBenchv5": (0.0609, 0.015368),
        "GPQADiamond":     (0.1970, 0.014830),
        "AIME25":          (0.7333, 0.014701),
        "MTBench":         (7.7531, 0.023351),
        "IFEval_m1":       (0.806,  0.001216),
        "IFEval_m2":       (0.8674, 0.001216),
        "LiveBench":       (35.3873, 16.602912),
    },
    "Qwen3-235B-Instruct": {
        "LiveCodeBenchv5": (0.4265, 0.000473),
        "GPQADiamond":     (0.4848, 0.000148),
        "AIME25":          (0.2333, 0.001199),
        "MTBench":         (9.2906, 0.000847),
        "IFEval_m1":       (0.918,  0.000046),
        "IFEval_m2":       (0.9454, 0.000046),
        "LiveBench":       (47.7381, 0.924759),
    },
    "Qwen3-235B-Thinking": {
        "LiveCodeBenchv5": (0.0251, 0.006353),
        "GPQADiamond":     (0.2121, 0.006180),
        "AIME25":          (0.7333, 0.006232),
        "MTBench":         (7.8469, 0.009962),
        "IFEval_m1":       (0.350,  0.000607),
        "IFEval_m2":       (0.4889, 0.000607),
        "LiveBench":       (17.0317, 7.745709),
    },
    "xRouter-7b-λ1": {
        "LiveCodeBenchv5": (0.6344, 0.007825),
        "GPQADiamond":     (0.7121, 0.004872),
        "AIME25":          (0.7667, 0.010513),
        "MTBench":         (8.0227, 0.007064),
        "IFEval_m1":       (0.778,  0.000573),
        "IFEval_m2":       (0.8479, 0.000573),
        "LiveBench":       (57.0392, 6.424311),
    },
    "xRouter-7b-λ2": {
        "LiveCodeBenchv5": (0.6774, 0.007166),
        "GPQADiamond":     (0.7172, 0.004548),
        "AIME25":          (0.7667, 0.015377),
        "MTBench":         (7.9780, 0.006981),
        "IFEval_m1":       (0.784,  0.000569),
        "IFEval_m2":       (0.8518, 0.000569),
        "LiveBench":       (56.9508, 6.411576),
    },
    "xRouter-7b-λ3": {
        "LiveCodeBenchv5": (0.4229, 0.006447),
        "GPQADiamond":     (0.6061, 0.001320),
        "AIME25":          (0.5000, 0.008646),
        "MTBench":         (8.3165, 0.005970),
        "IFEval_m1":       (0.806,  0.000496),
        "IFEval_m2":       (0.8635, 0.000496),
        "LiveBench":       (61.4092, 7.873074),
    },
}

# ------------------------------------
# 2) Choose the three tables to plot
# ------------------------------------
# Replace these with the three you selected earlier. "AIME25" is explicitly included per your selection.
# Examples you might choose: "LiveCodeBenchv5", "GPQADiamond", "AIME25", "MTBench", "IFEval_m1", "IFEval_m2", "LiveBench"
selected_tasks = ["AIME25", "GPQADiamond", "LiveCodeBenchv5"]

# Titles (shown exactly on subplots)
pretty_titles = {
    "AIME25": "AIME25",
    "GPQADiamond": "GPQADiamond",
    "LiveCodeBenchv5": "LiveCodeBenchv5",
    "MTBench": "MTBench",
    "IFEval_m1": "IFEval (metric 1)",
    "IFEval_m2": "IFEval (metric 2)",
    "LiveBench": "LiveBench",
}

# ------------------------------------
# 3) Build a tidy DataFrame
# ------------------------------------
records = []
for model, tasks in data.items():
    for task, (acc, cost) in tasks.items():
        records.append({
            "Model": model,
            "Task": task,
            "Accuracy": acc,
            "Cost": cost,
            "CostUtility": acc / cost if cost > 0 else float('nan')
        })
df = pd.DataFrame(records)

# ------------------------------------
# 4) Pastel Japanese-style color palette
#    (soft, muted, consistent per model)
# ------------------------------------
# 11 distinct gentle colors for the 11 models:
palette = [
    "#A8D8B9",  # soft green
    "#F6C6C7",  # sakura pink
    "#C5D6F7",  # pale blue
    "#F7E7A9",  # light yellow
    "#E6C9F2",  # soft lavender
    "#FAD4B8",  # peach
    "#BFE3DA",  # mint
    "#FDE2E4",  # blush
    "#D7E3FC",  # powder blue
    "#FFE5B4",  # apricot
    "#E2F0CB",  # pistachio
]

models = list(data.keys())
color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}

# ------------------------------------
# 5) Plot: three horizontal subplots
# ------------------------------------
plt.figure(figsize=(18, 6.5))
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True, sharey=False)

for ax, task in zip(axes, selected_tasks):
    # Slice and sort by ACCURACY (desc), but plot COST UTILITY
    sub = df[df["Task"] == task].copy()

    # Sort based on Accuracy
    # sub = sub.sort_values("Accuracy", ascending=False)
    # Sort based on Cost Utility
    sub = sub.sort_values("CostUtility", ascending=True)

    # Bars: horizontal; y = models, x = cost utility
    y_labels = sub["Model"].tolist()
    x_vals = sub["CostUtility"].tolist()
    colors = [color_map[m] for m in y_labels]

    # Horizontal bar plot
    ax.barh(y_labels, x_vals, color=colors, edgecolor="#555555", linewidth=0.6)

    # Aesthetics
    ax.set_xscale("log")  # log y-axis requested, but here cost utility is on x because bars are horizontal.
                          # If you prefer log on vertical axis, switch to vertical bars.
    ax.set_title(pretty_titles.get(task, task), fontsize=18, pad=10)
    ax.set_xlabel("Cost Utility (Accuracy ÷ Cost)", fontsize=18, labelpad=8)
    ax.tick_params(axis='both', which='both', labelsize=15)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    # Optional: custom min/max for the cost utility axis (log scale)
    # If you want to force limits, uncomment and set reasonable bounds:
    # xmin = min(v for v in x_vals if math.isfinite(v) and v > 0)
    # xmax = max(v for v in x_vals if math.isfinite(v))
    # ax.set_xlim(xmin * 0.8, xmax * 1.2)

# One legend for all models (outside the plots)
# We'll create a dummy legend using unique model labels and their colors
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=color_map[m], edgecolor="#555555", label=m) for m in models]
# fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))

# Overall styling
# fig.suptitle("Cost Utility (ranked by Accuracy within each benchmark)", fontsize=20, y=1.02)

plt.savefig("evaluation/analysis/cost_utility.png", dpi=300, bbox_inches="tight")
