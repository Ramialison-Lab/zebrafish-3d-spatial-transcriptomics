# ============================================================
# Quantitative evaluation of z-stack alignment: six fish
#
# PURPOSE:
#   Compare adjacent-section image similarity before and after
#   sequential section alignment using two image-based metrics.
#
# METRICS:
#   1. Adjacent-section normalized cross-correlation (NCC)
#   2. Gradient-based structural similarity, calculated as NCC
#      between the Sobel gradient-magnitude images
#
# EVALUATION:
#   - Compare consecutive sections within each z-stack.
#   - Exclude blank / zero-signal slices automatically.
#   - Exclude adjacent-section pairs that are invalid in either
#     the before-alignment or after-alignment stack from paired
#     comparisons.
#   - Use a tissue mask to reduce the contribution of background
#     pixels when calculating the metrics.
#
# STATISTICS:
#   - Each fish contributes one median value per metric and
#     condition.
#   - Paired, two-sided t-tests compare fish-level medians
#     before and after alignment.
#   - Report mean paired changes and 95% confidence intervals.
#   - Adjacent-section pairs are not treated as independent
#     biological replicates.
#
# IMPORTANT:
#   - Previously identified bad sections must have been excluded
#     before z-stack generation.
#   - Higher similarity scores indicate greater image similarity,
#     but do not independently establish anatomical accuracy.
#   - No residual displacement is calculated in this script.
#
# OUTPUTS:
#   Per fish:
#       <fish>_alignment_metrics.csv
#       <fish>_NCC.png
#       <fish>_edge_similarity.png
#       <fish>_alignment_summary.png
#
#   Combined:
#       all_fish_alignment_metrics.csv
#       all_fish_summary.csv
#       all_fish_mean_sd_summary.csv
#       overall_statistical_tests.csv
#       all_fish_NCC.png
#       all_fish_edge_similarity.png
#       all_fish_alignment_summary.png
#
# DEPENDENCIES:
#   pip install pynrrd scipy pandas matplotlib scikit-image
# ============================================================

import os
import numpy as np
import pandas as pd
import nrrd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, t
from skimage.filters import sobel


# ============================================================
# 1. Paths and parameters
# ============================================================

BASE_DIR = "/group/tran3/swei/MERFISH/MERSCOPE"

BEFORE_BASE = os.path.join(
    BASE_DIR, "data", "results", "zstack_no_alignment"
)

AFTER_BASE = os.path.join(
    BASE_DIR, "data", "zstack"
)

OUTPUT_BASE = os.path.join(
    BASE_DIR, "data", "results", "alignment_evaluation", "wstats"
)

FISHES = ["WT1", "WT2", "WT3", "mut1", "mut2", "mut3"]

# NRRD array axes are (y, x, z).
# Move z to the first axis for processing.
Z_AXIS = 2

USE_TISSUE_MASK = True
TISSUE_PERCENTILE = 20

MINIMUM_NONZERO_FRACTION = 0.0001
MINIMUM_STD = 1e-8

BEFORE_COLOR = "0.45"
AFTER_COLOR = "#0072B2"

os.makedirs(OUTPUT_BASE, exist_ok=True)


# ============================================================
# 2. Load NRRD stack
# ============================================================

def load_stack(path, z_axis=Z_AXIS):
    stack, header = nrrd.read(path)
    stack = np.asarray(stack, dtype=np.float32)

    print(f"  Original shape: {stack.shape}")

    stack = np.moveaxis(stack, z_axis, 0)

    print(f"  z-first shape:  {stack.shape}")

    return stack, header


# ============================================================
# 3. Detect valid / blank slices
# ============================================================

def slice_has_signal(image):
    image = np.asarray(image, dtype=np.float32)
    values = image[np.isfinite(image)]

    if len(values) == 0:
        return False

    nonzero_fraction = np.count_nonzero(values > 0) / len(values)
    return (
        nonzero_fraction >= MINIMUM_NONZERO_FRACTION
        and np.std(values) > MINIMUM_STD
    )


def inspect_slices(stack):
    valid = []

    print("\n  Slice signal check:")

    for z, image in enumerate(stack):
        values = image[np.isfinite(image)]

        nonzero_fraction = (
            np.count_nonzero(values > 0) / len(values)
            if len(values) else 0
        )

        std = np.std(values) if len(values) else 0
        has_signal = slice_has_signal(image)
        valid.append(has_signal)

        print(
            f"    z={z:2d} | nonzero={nonzero_fraction:.6f} | "
            f"std={std:.4f} | "
            f"{'VALID' if has_signal else 'BLANK'}"
        )

    return np.asarray(valid, dtype=bool)


# ============================================================
# 4. Robust image normalization
# ============================================================

def robust_normalize(image):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros_like(image)

    values = image[finite]
    p_low, p_high = np.percentile(values, [1, 99])

    if p_high <= p_low:
        return np.zeros_like(image)

    result = np.clip(
        (image - p_low) / (p_high - p_low),
        0,
        1
    )

    result[~finite] = 0

    return result


# ============================================================
# 5. Generate tissue mask
# ============================================================

def make_tissue_mask(a, b, percentile=TISSUE_PERCENTILE):
    a = np.asarray(a)
    b = np.asarray(b)

    a_finite = a[np.isfinite(a)]
    b_finite = b[np.isfinite(b)]

    a_positive = a_finite[a_finite > 0]
    b_positive = b_finite[b_finite > 0]

    threshold_a = (
        np.percentile(a_positive, percentile)
        if len(a_positive) > 100
        else np.percentile(a_finite, percentile)
    )

    threshold_b = (
        np.percentile(b_positive, percentile)
        if len(b_positive) > 100
        else np.percentile(b_finite, percentile)
    )

    # Union of positive-signal regions in the two images.
    return (a > threshold_a) | (b > threshold_b)


# ============================================================
# 6. Normalized cross-correlation
# ============================================================

def normalized_cross_correlation(a, b, mask=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    valid = np.isfinite(a) & np.isfinite(b)

    if mask is not None:
        valid &= mask

    a = a[valid]
    b = b[valid]

    if len(a) < 10:
        return np.nan

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = np.sqrt(
        np.sum(a ** 2) * np.sum(b ** 2)
    )

    if denominator == 0:
        return np.nan

    return float(np.sum(a * b) / denominator)


# ============================================================
# 7. Gradient-based structural similarity
# ============================================================

def make_gradient_image(image):
    """
    Robustly normalize an image and calculate its Sobel
    gradient magnitude to emphasize structural boundaries.
    """
    image = robust_normalize(image)
    return sobel(image)


def edge_similarity(a, b, mask=None):
    """
    Calculate NCC between the Sobel gradient-magnitude images.
    Higher values indicate greater correspondence of edges.
    """
    gradient_a = make_gradient_image(a)
    gradient_b = make_gradient_image(b)

    return normalized_cross_correlation(
        gradient_a,
        gradient_b,
        mask=mask
    )


# ============================================================
# 8. Calculate metrics for adjacent sections in one stack
# ============================================================

def calculate_stack_metrics(stack, valid_slices):
    results = []

    for z in range(stack.shape[0] - 1):
        z_next = z + 1

        reference = stack[z]
        moving = stack[z_next]

        if not valid_slices[z] or not valid_slices[z_next]:
            reason = (
                "blank_reference"
                if not valid_slices[z]
                else "blank_moving"
            )

            results.append({
                "z": z,
                "z_next": z_next,
                "valid": False,
                "invalid_reason": reason,
                "ncc": np.nan,
                "edge_similarity": np.nan
            })
            continue

        if USE_TISSUE_MASK:
            tissue_mask = make_tissue_mask(
                reference,
                moving,
                percentile=TISSUE_PERCENTILE
            )
        else:
            tissue_mask = None

        ncc = normalized_cross_correlation(
            reference,
            moving,
            mask=tissue_mask
        )

        edge_sim = edge_similarity(
            reference,
            moving,
            mask=tissue_mask
        )

        results.append({
            "z": z,
            "z_next": z_next,
            "valid": True,
            "invalid_reason": "",
            "ncc": ncc,
            "edge_similarity": edge_sim
        })

    return pd.DataFrame(results)


# ============================================================
# 9. Evaluate one fish
# ============================================================

def evaluate_fish(fish):
    print(f"\n{'=' * 75}\nPROCESSING {fish}\n{'=' * 75}")

    before_path = os.path.join(
        BEFORE_BASE, fish, f"{fish}_zstack_noalign.nrrd"
    )

    after_path = os.path.join(
        AFTER_BASE, fish, f"{fish}_zstack.nrrd"
    )

    output_dir = os.path.join(OUTPUT_BASE, fish)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(before_path):
        print(f"ERROR: missing before stack: {before_path}")
        return None

    if not os.path.exists(after_path):
        print(f"ERROR: missing after stack: {after_path}")
        return None

    print(f"\nBefore: {before_path}")
    print(f"After:  {after_path}")

    before, _ = load_stack(before_path)
    after, _ = load_stack(after_path)

    n_slices = before.shape[0]
    print(f"\nNumber of slices: {n_slices}")

    print("\nBEFORE STACK")
    before_valid = inspect_slices(before)

    print("\nAFTER STACK")
    after_valid = inspect_slices(after)

    print(
        f"\nValid slices: before {before_valid.sum()}/{n_slices}; "
        f"after {after_valid.sum()}/{n_slices}"
    )

    print("\nCalculating BEFORE metrics...")
    before_metrics = calculate_stack_metrics(before, before_valid)

    print("Calculating AFTER metrics...")
    after_metrics = calculate_stack_metrics(after, after_valid)

    results = pd.DataFrame({
        "fish": fish,
        "z": before_metrics["z"],
        "z_next": before_metrics["z_next"],
        "before_valid": before_metrics["valid"],
        "after_valid": after_metrics["valid"],
        "before_invalid_reason": before_metrics["invalid_reason"],
        "after_invalid_reason": after_metrics["invalid_reason"],
        "before_ncc": before_metrics["ncc"],
        "after_ncc": after_metrics["ncc"],
        "before_edge_similarity": before_metrics["edge_similarity"],
        "after_edge_similarity": after_metrics["edge_similarity"]
    })

    results["ncc_improvement"] = (
        results["after_ncc"] - results["before_ncc"]
    )

    results["edge_similarity_improvement"] = (
        results["after_edge_similarity"]
        - results["before_edge_similarity"]
    )

    # A pair is eligible only when both the before and after
    # adjacent pairs contain valid slices.
    results["paired_valid"] = (
        results["before_valid"] & results["after_valid"]
    )

    csv_path = os.path.join(
        output_dir, f"{fish}_alignment_metrics.csv"
    )

    results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    paired = results[results["paired_valid"]]

    print(
        f"\n{fish}: {len(results)} adjacent pairs; "
        f"{len(paired)} pairs valid before and after."
    )

    if not paired.empty:
        for metric, label in [
            ("ncc", "NCC"),
            ("edge_similarity", "Structural similarity")
        ]:
            before_col = f"before_{metric}"
            after_col = f"after_{metric}"

            values = paired[[before_col, after_col]].dropna()

            if not values.empty:
                print(
                    f"  {label} median: "
                    f"{values[before_col].median():.4f} -> "
                    f"{values[after_col].median():.4f}"
                )

    make_fish_figures(results, fish, output_dir)

    return results


# ============================================================
# 10. Per-fish plots
# ============================================================

def make_fish_figures(results, fish, output_dir):
    # Plot adjacent-pair NCC.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        results["z"], results["before_ncc"],
        color=BEFORE_COLOR, linewidth=1.5, label="Before alignment"
    )

    ax.plot(
        results["z"], results["after_ncc"],
        color=AFTER_COLOR, linewidth=1.5, label="After alignment"
    )

    ax.set(
        xlabel="Adjacent-section pair (z)",
        ylabel="Adjacent-section NCC",
        title=f"{fish}: adjacent-section NCC"
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, f"{fish}_NCC.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot adjacent-pair gradient similarity.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        results["z"], results["before_edge_similarity"],
        color=BEFORE_COLOR, linewidth=1.5, label="Before alignment"
    )

    ax.plot(
        results["z"], results["after_edge_similarity"],
        color=AFTER_COLOR, linewidth=1.5, label="After alignment"
    )

    ax.set(
        xlabel="Adjacent-section pair (z)",
        ylabel="Gradient-based structural similarity",
        title=f"{fish}: structural similarity"
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, f"{fish}_edge_similarity.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Compact before/after summary for the fish.
    paired = results[results["paired_valid"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))

    metrics = [
        ("before_ncc", "after_ncc", "Adjacent-section NCC"),
        (
            "before_edge_similarity",
            "after_edge_similarity",
            "Structural similarity"
        )
    ]

    for ax, (before_col, after_col, ylabel) in zip(axes, metrics):
        values = paired[[before_col, after_col]].dropna()

        before = values[before_col].to_numpy()
        after = values[after_col].to_numpy()

        for b, a in zip(before, after):
            ax.plot(
                [0, 1], [b, a],
                color="0.8", linewidth=0.6, alpha=0.5
            )

        ax.scatter(
            np.zeros(len(before)), before,
            s=14, color=BEFORE_COLOR, alpha=0.7
        )

        ax.scatter(
            np.ones(len(after)), after,
            s=14, color=AFTER_COLOR, alpha=0.7
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before", "After"])
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{fish}")
    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, f"{fish}_alignment_summary.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 11. Run all fish
# ============================================================

all_results = []

for fish in FISHES:
    result = evaluate_fish(fish)

    if result is not None:
        all_results.append(result)

if not all_results:
    raise RuntimeError("No fish were successfully processed.")

all_results = pd.concat(all_results, ignore_index=True)

combined_csv = os.path.join(
    OUTPUT_BASE, "all_fish_alignment_metrics.csv"
)

all_results.to_csv(combined_csv, index=False)
print(f"\nSaved combined metrics: {combined_csv}")


# ============================================================
# 12. Fish-level summary
#
# One median per fish, per metric, per condition.
# Use only adjacent pairs valid in both stacks.
# ============================================================

summary_rows = []

for fish in FISHES:
    fish_data = all_results[all_results["fish"] == fish]
    paired = fish_data[fish_data["paired_valid"]]

    row = {
        "fish": fish,
        "n_adjacent_pairs": len(fish_data),
        "n_valid_paired_sections": len(paired),
        "n_invalid_pairs": len(fish_data) - len(paired)
    }

    for metric in ["ncc", "edge_similarity"]:
        before_col = f"before_{metric}"
        after_col = f"after_{metric}"

        values = paired[[before_col, after_col]].dropna()

        row[f"n_{metric}_pairs"] = len(values)

        row[f"before_{metric}_median"] = (
            values[before_col].median() if not values.empty else np.nan
        )

        row[f"after_{metric}_median"] = (
            values[after_col].median() if not values.empty else np.nan
        )

        row[f"{metric}_improvement_median"] = (
            values[after_col].median() - values[before_col].median()
            if not values.empty else np.nan
        )

        row[f"before_{metric}_mean"] = (
            values[before_col].mean() if not values.empty else np.nan
        )

        row[f"after_{metric}_mean"] = (
            values[after_col].mean() if not values.empty else np.nan
        )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary_csv = os.path.join(OUTPUT_BASE, "all_fish_summary.csv")
summary.to_csv(summary_csv, index=False)

print(f"\nSaved fish-level summary: {summary_csv}")

print("\nFISH-LEVEL MEDIAN SUMMARY")

summary_columns = [
    "fish",
    "n_valid_paired_sections",
    "before_ncc_median",
    "after_ncc_median",
    "before_edge_similarity_median",
    "after_edge_similarity_median"
]

print(
    summary[summary_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. Descriptive mean and SD across fish
# ============================================================

mean_sd_rows = []

for metric, label in [
    ("ncc", "Adjacent-section NCC"),
    ("edge_similarity", "Gradient-based structural similarity")
]:
    for condition in ["before", "after"]:
        col = f"{condition}_{metric}_median"
        values = summary[col].dropna()

        mean_sd_rows.append({
            "metric": label,
            "condition": condition,
            "n_fish": len(values),
            "mean_of_fish_medians": values.mean(),
            "sd_of_fish_medians": values.std(ddof=1)
            if len(values) >= 2 else np.nan
        })

mean_sd_df = pd.DataFrame(mean_sd_rows)

mean_sd_path = os.path.join(
    OUTPUT_BASE, "all_fish_mean_sd_summary.csv"
)

mean_sd_df.to_csv(mean_sd_path, index=False)
print(f"\nSaved descriptive summary: {mean_sd_path}")



# ============================================================
# 14. Primary statistical tests across fish
#
# Paired, two-sided t-tests.
# Each fish contributes one before and one after median.
# ============================================================

test_definitions = [
    {
        "metric": "Adjacent-section NCC",
        "key": "ncc",
        "before": "before_ncc_median",
        "after": "after_ncc_median"
    },
    {
        "metric": "Gradient-based structural similarity",
        "key": "edge_similarity",
        "before": "before_edge_similarity_median",
        "after": "after_edge_similarity_median"
    }
]

test_rows = []

for item in test_definitions:
    test_data = summary[
        ["fish", item["before"], item["after"]]
    ].dropna()

    before = test_data[item["before"]].to_numpy(dtype=float)
    after = test_data[item["after"]].to_numpy(dtype=float)

    n = len(test_data)
    differences = after - before

    if n >= 2:
        test_result = ttest_rel(
            after,
            before,
            alternative="two-sided"
        )

        t_statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
        df = n - 1

        mean_change = float(np.mean(differences))
        median_change = float(np.median(differences))
        sd_change = float(np.std(differences, ddof=1))
        se_change = sd_change / np.sqrt(n)

        # 95% confidence interval for the mean paired change
        t_critical = t.ppf(0.975, df)

        ci_lower = mean_change - t_critical * se_change
        ci_upper = mean_change + t_critical * se_change

    else:
        t_statistic = np.nan
        p_value = np.nan
        df = np.nan
        mean_change = np.nan
        median_change = np.nan
        sd_change = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    test_rows.append({
        "metric": item["metric"],
        "n_fish": n,
        "test": "Paired t-test, two-sided",
        "t_statistic": t_statistic,
        "df": df,
        "p_value": p_value,
        "mean_change_after_minus_before": mean_change,
        "sd_of_paired_changes": sd_change,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "median_change_after_minus_before": median_change,
        "increased_in_mean": (
            mean_change > 0 if np.isfinite(mean_change) else np.nan
        )
    })

test_df = pd.DataFrame(test_rows)

test_path = os.path.join(
    OUTPUT_BASE, "overall_statistical_tests.csv"
)

test_df.to_csv(test_path, index=False)

print("\nFISH-LEVEL PAIRED T-TESTS")

for _, row in test_df.iterrows():
    if pd.notna(row["p_value"]):
        print(f"\n{row['metric']} (n={int(row['n_fish'])} fish)")
        print(
            f"  t({int(row['df'])}) = {row['t_statistic']:.3f}"
        )
        print(f"  p = {row['p_value']:.6g}")
        print(
            f"  Mean change = "
            f"{row['mean_change_after_minus_before']:.4f}"
        )
        print(
            f"  95% CI = "
            f"[{row['ci95_lower']:.4f}, "
            f"{row['ci95_upper']:.4f}]"
        )
    else:
        print(f"\n{row['metric']}: insufficient data for testing.")

print(f"\nSaved statistical results: {test_path}")


# ============================================================
# 15. Combined plots across adjacent-section pairs
#
# These plots are descriptive; pairs are not independent fish.
# ============================================================

for metric, ylabel, filename, title in [
    (
        "ncc",
        "Adjacent-section NCC",
        "all_fish_NCC.png",
        "Adjacent-section image similarity"
    ),
    (
        "edge_similarity",
        "Gradient-based structural similarity",
        "all_fish_edge_similarity.png",
        "Adjacent-section structural similarity"
    )
]:
    fig, ax = plt.subplots(figsize=(9, 5))

    for fish in FISHES:
        fish_data = all_results[all_results["fish"] == fish]

        ax.plot(
            fish_data["z"],
            fish_data[f"before_{metric}"],
            linewidth=1,
            alpha=0.5,
            label=f"{fish} before"
        )

        ax.plot(
            fish_data["z"],
            fish_data[f"after_{metric}"],
            linewidth=1,
            alpha=0.8,
            linestyle="--",
            label=f"{fish} after"
        )

    ax.set(
        xlabel="Adjacent-section pair (z)",
        ylabel=ylabel,
        title=title
    )

    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_BASE, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")



# ============================================================
# 16. Main thesis figure
#
# A: NCC
# B: Gradient-based structural similarity
#
# Circles = individual fish-level medians.
# Connecting lines = paired before/after measurements.
# Diamonds with error bars = mean +/- SD across fish.
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 14
SUPTITLE_SIZE = 15

plot_definitions = [
    (
        axes[0],
        "before_ncc_median",
        "after_ncc_median",
        "Median adjacent-section NCC",
        " Image similarity"
    ),
    (
        axes[1],
        "before_edge_similarity_median",
        "after_edge_similarity_median",
        "Median gradient-based structural similarity",
        " Structural similarity"
    )
]

plot_summary = summary.reset_index(drop=True)
fish_positions = np.arange(len(plot_summary))
mean_position = len(plot_summary)

for ax, before_col, after_col, ylabel, title in plot_definitions:

    before_values = plot_summary[before_col].to_numpy(dtype=float)
    after_values = plot_summary[after_col].to_numpy(dtype=float)

    # Individual paired fish-level medians
    for i, (before_value, after_value) in enumerate(
        zip(before_values, after_values)
    ):
        if not np.isfinite(before_value) or not np.isfinite(after_value):
            continue

        ax.plot(
            [i, i],
            [before_value, after_value],
            color="0.75",
            linewidth=1.2,
            zorder=1
        )

        ax.scatter(
            i, before_value,
            s=55,
            color=BEFORE_COLOR,
            zorder=2
        )

        ax.scatter(
            i, after_value,
            s=55,
            color=AFTER_COLOR,
            zorder=2
        )

    # Group mean +/- SD across fish-level medians
    valid_before = before_values[np.isfinite(before_values)]
    valid_after = after_values[np.isfinite(after_values)]

    if len(valid_before) >= 2:
        ax.errorbar(
            mean_position - 0.10,
            np.mean(valid_before),
            yerr=np.std(valid_before, ddof=1),
            fmt="D",
            markersize=7,
            color=BEFORE_COLOR,
            ecolor=BEFORE_COLOR,
            capsize=4,
            elinewidth=1.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=4
        )

    if len(valid_after) >= 2:
        ax.errorbar(
            mean_position + 0.10,
            np.mean(valid_after),
            yerr=np.std(valid_after, ddof=1),
            fmt="D",
            markersize=7,
            color=AFTER_COLOR,
            ecolor=AFTER_COLOR,
            capsize=4,
            elinewidth=1.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=4
        )

    ax.set_xticks(
        list(fish_positions) + [mean_position]
    )

    ax.set_xticklabels(
        list(plot_summary["fish"]) + ["Mean"],
        fontsize=TICK_SIZE
    )

    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(
        title,
        loc="left",
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlim(-0.5, mean_position + 0.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend: individual fish and group summaries
handles = [
    plt.Line2D(
        [], [], marker="o", linestyle="None",
        markerfacecolor=BEFORE_COLOR,
        markeredgecolor=BEFORE_COLOR,
        markersize=7,
        label="Before alignment"
    ),
    plt.Line2D(
        [], [], marker="o", linestyle="None",
        markerfacecolor=AFTER_COLOR,
        markeredgecolor=AFTER_COLOR,
        markersize=7,
        label="After alignment"
    ),
    plt.Line2D(
        [], [], marker="D", linestyle="None",
        markerfacecolor="white",
        markeredgecolor="0.3",
        markersize=7,
        label="Group mean ± SD"
    )
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=LEGEND_SIZE
)

fig.suptitle(
    "Quantitative evaluation of sequential section alignment",
    fontsize=SUPTITLE_SIZE,
    y=1.10
)

fig.tight_layout(rect=[0, 0, 1, 0.91])

main_figure_path = os.path.join(
    OUTPUT_BASE, "all_fish_alignment_summary.png"
)

fig.savefig(
    main_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"\nSaved main thesis figure: {main_figure_path}")

# ============================================================
# 17. Final report
# ============================================================

print("\n" + "=" * 75)
print("ALIGNMENT EVALUATION FINISHED")
print("=" * 75)

print(f"\nOutput directory:\n{OUTPUT_BASE}")

print("\nFinal per-fish median values:")

for _, row in summary.iterrows():
    print(f"\n{row['fish']}")

    print(
        f"  NCC: "
        f"{row['before_ncc_median']:.4f} -> "
        f"{row['after_ncc_median']:.4f}"
    )

    print(
        f"  Structural similarity: "
        f"{row['before_edge_similarity_median']:.4f} -> "
        f"{row['after_edge_similarity_median']:.4f}"
    )

print("\nPaired t-tests:")
print(test_df.to_string(index=False))
