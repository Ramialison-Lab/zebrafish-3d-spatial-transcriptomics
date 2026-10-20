"""
Pseudobulk differential expression analysis:
mutant vs WT across MERFISH brain regions.

Outputs:
    - Pseudobulk counts and sample metadata
    - DESeq2 results per region
    - Publication-quality volcano plots (PNG, 300 dpi)
    - Pseudobulk library-size QC plot
    - Per-region PCA and sample-correlation QC plots
    - QC and DEG summary tables

Requirements:
    pip install pydeseq2 scanpy anndata pandas numpy scipy \
        matplotlib seaborn scikit-learn

IMPORTANT:
    DESeq2 requires raw, non-negative integer counts.
    Confirm the correct count source before interpreting results.
"""
### used for final figure

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/merged_3d/full_region_label"
)

OUT_DIR = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/results/"
    "pseudobulk_deg_results2"
)

FISH_LIST = ["mut1", "mut2", "mut3", "WT1", "WT2", "WT3"]

# Change this if your raw counts are stored elsewhere.
# Priority: layers["counts"], then adata.X.
COUNT_LAYER = "counts"

# Use None to retain all region_broad categories.
# These exclusions match the region exclusions used in your
# previous regional DEG workflow. Check the exact label spelling.
EXCLUDE_REGIONS = {
    "unassigned"
}

MIN_CELLS_PER_PSEUDOBULK = 10
MIN_TOTAL_COUNT = 10
ALPHA = 0.05

# The combined dot plot shows up to this many top significant
# genes per region, ranked by adjusted p-value.
# Set to None to display every significant gene.
TOP_N_GENES_PER_REGION = 15

# Label this many top significant genes on each volcano plot.
N_VOLCANO_LABELS = 10

DPI = 300
FIG_FORMAT = "png"

# Plot styling
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "volcano_plots").mkdir(exist_ok=True)
(OUT_DIR / "qc").mkdir(exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def genotype_from_fish(fish_id):
    return "mut" if str(fish_id).startswith("mut") else "WT"


def safe_filename(value):
    """Create a filename-safe label."""
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in str(value)
    )


def get_raw_counts(adata):
    """
    Return the selected raw count matrix as a sparse matrix
    where possible.

    Do not use log-normalised or scaled expression values here.
    """
    if COUNT_LAYER is not None and COUNT_LAYER in adata.layers:
        X = adata.layers[COUNT_LAYER]
        source = f"layers['{COUNT_LAYER}']"
    else:
        X = adata.X
        source = "X"

    if sp.issparse(X):
        X = X.tocsr()
        data = X.data
    else:
        X = np.asarray(X)
        data = X.ravel()

    if not np.all(np.isfinite(data)):
        raise ValueError(f"{source} contains NaN or infinite values.")

    if np.any(data < 0):
        raise ValueError(f"{source} contains negative values.")

    if not np.allclose(data, np.round(data), atol=1e-6):
        raise ValueError(
            f"{source} does not contain integer-like raw counts. "
            "Check whether it contains normalised or log-transformed "
            "expression. Configure COUNT_LAYER to point to raw counts."
        )

    print(f"Using {source} as the count source.")
    return X


def save_figure(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 3. LOAD ALL FISH
# ============================================================

def load_all_fish():
    adatas = []

    for fish in FISH_LIST:
        path = DATA_DIR / f"{fish}_full_labelled.h5ad"

        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        print(f"Loading {fish}: {path}")
        a = sc.read_h5ad(path)

        if "major_region" not in a.obs.columns:
            raise KeyError(f"{fish}: missing obs['major_region'].")

        if a.n_obs == 0 or a.n_vars == 0:
            raise ValueError(f"{fish}: empty AnnData object.")

        a.obs["fish_id"] = fish
        a.obs["genotype"] = genotype_from_fish(fish)

        print(f"  Cells: {a.n_obs:,}; genes: {a.n_vars:,}")
        adatas.append(a)

    # Outer join accommodates genes that may be absent from some fish.
    combined = ad.concat(
        adatas,
        join="outer",
        label="source_batch",
        index_unique="-",
        fill_value=0,
    )

    combined.obs["major_region"] = (
        combined.obs["major_region"].astype("string").fillna("unassigned")
    )

    combined.obs["fish_id"] = combined.obs["fish_id"].astype(str)
    combined.obs["genotype"] = combined.obs["genotype"].astype(str)

    return combined


# ============================================================
# 4. BUILD PSEUDOBULK COUNTS
# ============================================================

def build_pseudobulk(adata):
    X = get_raw_counts(adata)

    obs = adata.obs[
        ["fish_id", "genotype", "major_region"]
    ].copy().reset_index(drop=True)

    excluded = {str(x).strip().lower() for x in EXCLUDE_REGIONS}
    region_norm = obs["major_region"].astype(str).str.strip().str.lower()
    keep = ~region_norm.isin(excluded)

    n_excluded = int((~keep).sum())
    if n_excluded:
        print(f"Excluding {n_excluded:,} cells in excluded regions.")

    obs = obs.loc[keep].reset_index(drop=True)
    X = X[np.flatnonzero(keep.to_numpy())]

    rows = []
    metadata = []

    grouped = obs.groupby(
        ["fish_id", "major_region"],
        observed=True,
        sort=True,
    ).indices

    for (fish_id, region), positions in grouped.items():
        positions = np.asarray(positions)
        n_cells = len(positions)

        if n_cells < MIN_CELLS_PER_PSEUDOBULK:
            print(
                f"[skip] {fish_id}, {region}: "
                f"only {n_cells} cells"
            )
            continue

        # Sum raw counts across cells without densifying the full matrix.
        summed = np.asarray(X[positions].sum(axis=0)).ravel()

        rows.append(summed)

        metadata.append({
            "sample": f"{fish_id}__{region}",
            "fish_id": fish_id,
            "major_region": str(region),
            "genotype": genotype_from_fish(fish_id),
            "n_cells": n_cells,
        })

    if not rows:
        raise ValueError("No pseudobulk samples were created.")

    meta = pd.DataFrame(metadata).set_index("sample")
    counts = pd.DataFrame(
        np.vstack(rows),
        index=meta.index,
        columns=adata.var_names,
    )

    counts = counts.astype(np.int64)

    counts.to_csv(OUT_DIR / "pseudobulk_counts.csv")
    meta.to_csv(OUT_DIR / "pseudobulk_sample_metadata.csv")

    # Replicate availability by region
    replicate_table = (
        meta.groupby(["major_region", "genotype"])["fish_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    replicate_table.to_csv(OUT_DIR / "replicate_counts_by_region.csv")

    print("\nPseudobulk replicate counts:")
    print(replicate_table.to_string())

    return counts, meta


# ============================================================
# 5. PSEUDOBULK LIBRARY-SIZE QC
# ============================================================

def plot_library_size_qc(counts, meta):
    qc = meta.copy()
    qc["library_size"] = counts.sum(axis=1)
    qc["log10_library_size"] = np.log10(
        qc["library_size"].clip(lower=1)
    )

    qc.to_csv(OUT_DIR / "qc" / "pseudobulk_library_sizes.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=qc.reset_index(),
        x="major_region",
        y="library_size",
        hue="genotype",
        errorbar=None,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Total raw counts (log scale)")
    ax.set_title("Pseudobulk library sizes by region and genotype")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Genotype", frameon=False)
    sns.despine(ax=ax)

    save_figure(fig, OUT_DIR / "qc" / "pseudobulk_library_sizes.png")

    # Number of cells contributing to each pseudobulk sample
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(
        data=qc.reset_index(),
        x="major_region",
        y="n_cells",
        hue="genotype",
        dodge=True,
        size=6,
        ax=ax,
    )
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Cells per pseudobulk sample")
    ax.set_title("Cell numbers contributing to pseudobulk samples")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Genotype", frameon=False)
    sns.despine(ax=ax)

    save_figure(fig, OUT_DIR / "qc" / "pseudobulk_cell_numbers.png")

    return qc


# ============================================================
# 6. RUN DESEQ2 SEPARATELY IN EACH REGION
# ============================================================

def run_deseq_per_region(counts, meta):
    all_results = {}
    diagnostic_rows = []

    regions = sorted(meta["major_region"].unique())

    for region in regions:
        safe_region = safe_filename(region)
        sub_meta = meta.loc[meta["major_region"] == region].copy()

        # Explicitly check biological replicates per genotype.
        replicate_counts = (
            sub_meta.groupby("genotype")["fish_id"].nunique()
        )
        n_mut = int(replicate_counts.get("mut", 0))
        n_wt = int(replicate_counts.get("WT", 0))

        if n_mut < 2 or n_wt < 2:
            print(
                f"[skip] {region}: insufficient biological replicates "
                f"(mut={n_mut}, WT={n_wt})."
            )
            diagnostic_rows.append({
                "major_region": region,
                "n_mut": n_mut,
                "n_WT": n_wt,
                "status": "skipped: insufficient replicates",
                "n_genes_tested": 0,
                "n_significant": 0,
            })
            continue

        sub_counts = counts.loc[sub_meta.index].copy()

        # Remove genes with very low total counts in this region.
        keep_genes = sub_counts.sum(axis=0) >= MIN_TOTAL_COUNT
        sub_counts = sub_counts.loc[:, keep_genes]

        if sub_counts.shape[1] == 0:
            print(f"[skip] {region}: no genes pass count filtering.")
            continue

        # Ensure category reference is WT, so positive log2FC means
        # higher expression in mutants.
        sub_meta["genotype"] = pd.Categorical(
            sub_meta["genotype"],
            categories=["WT", "mut"],
        )

        print(
            f"\nRunning DESeq2: {region} "
            f"(mut={n_mut}, WT={n_wt}, genes={sub_counts.shape[1]})"
        )

        try:
            dds = DeseqDataSet(
                counts=sub_counts,
                metadata=sub_meta,
                design="~genotype",
                refit_cooks=True,
                n_cpus=1,
                quiet=True,
            )
            dds.deseq2()

            stats = DeseqStats(
                dds,
                contrast=["genotype", "mut", "WT"],
                alpha=ALPHA,
                cooks_filter=True,
                independent_filter=True,
                n_cpus=1,
            )
            stats.summary()

            res = stats.results_df.copy()
            res.index.name = "gene"
            res = res.sort_values("padj", na_position="last")

            # Retain the DESeq2 Wald statistic (stat), p-value,
            # adjusted p-value and log2 fold change.
            result_path = (
                OUT_DIR / f"{safe_region}_mut_vs_WT_DESeq2.csv"
            )
            res.to_csv(result_path)

            all_results[region] = res

            significant = res["padj"] < ALPHA
            n_sig = int(significant.sum())
            n_up = int(
                (significant & (res["log2FoldChange"] > 0)).sum()
            )
            n_down = int(
                (significant & (res["log2FoldChange"] < 0)).sum()
            )

            print(
                f"  Significant: {n_sig}; "
                f"up in mutant: {n_up}; down in mutant: {n_down}"
            )

            diagnostic_rows.append({
                "major_region": region,
                "n_mut": n_mut,
                "n_WT": n_wt,
                "status": "completed",
                "n_genes_tested": len(res),
                "n_significant": n_sig,
                "n_up_in_mut": n_up,
                "n_down_in_mut": n_down,
                "median_abs_log2FC_sig": (
                    res.loc[significant, "log2FoldChange"].abs().median()
                    if n_sig else np.nan
                ),
            })

            # Save per-region normalised counts for QC plots.
            # These are for visualisation only, not DESeq2 input.
            norm_counts = pd.DataFrame(
                np.asarray(dds.layers["normed_counts"]),
                index=sub_counts.index,
                columns=sub_counts.columns,
            )
            norm_counts.to_csv(
                OUT_DIR / "qc" / f"{safe_region}_normalised_counts.csv"
            )

        except Exception as exc:
            warnings.warn(f"DESeq2 failed for {region}: {exc}")
            diagnostic_rows.append({
                "major_region": region,
                "n_mut": n_mut,
                "n_WT": n_wt,
                "status": f"failed: {exc}",
                "n_genes_tested": 0,
                "n_significant": 0,
            })

    pd.DataFrame(diagnostic_rows).to_csv(
        OUT_DIR / "DEG_summary_by_region.csv", index=False
    )

    return all_results


# ============================================================
# 7. VOLCANO PLOTS: PNG, 300 DPI
# ============================================================

def volcano_plot(res, region):
    df = res.dropna(
        subset=["log2FoldChange", "padj"]
    ).copy()

    if df.empty:
        return

    TITLE_SIZE = 13
    TEXT_SIZE = 12
    LEGEND_SIZE = 10

    df["neg_log10_padj"] = -np.log10(
        df["padj"].clip(lower=1e-300)
    )

    sig = df["padj"] < ALPHA
    up = sig & (df["log2FoldChange"] > 0)
    down = sig & (df["log2FoldChange"] < 0)
    ns = ~sig

    fig, ax = plt.subplots(figsize=(4, 3))

    # Non-significant genes
    ax.scatter(
        df.loc[ns, "log2FoldChange"],
        df.loc[ns, "neg_log10_padj"],
        c="#B8B8B8",
        s=20,
        alpha=0.55,
        linewidths=0,
        label=f"Not significant ({int(ns.sum())})",
        rasterized=True,
        zorder=2,
    )

    # Downregulated in mutant
    ax.scatter(
        df.loc[down, "log2FoldChange"],
        df.loc[down, "neg_log10_padj"],
        c="#2878B5",
        s=70,
        alpha=0.90,
        linewidths=0,
        label=f"Down in mutant ({int(down.sum())})",
        rasterized=True,
        zorder=3,
    )

    # Upregulated in mutant
    ax.scatter(
        df.loc[up, "log2FoldChange"],
        df.loc[up, "neg_log10_padj"],
        c="#D64B40",
        s=70,
        alpha=0.90,
        linewidths=0,
        label=f"Up in mutant ({int(up.sum())})",
        rasterized=True,
        zorder=3,
    )

    # Threshold lines
    ax.axhline(
        -np.log10(ALPHA),
        color="#666666",
        linestyle="--",
        linewidth=1,
        label=f"FDR = {ALPHA}",
        zorder=1,
    )

    ax.axvline(
        0,
        color="#888888",
        linewidth=0.7,
        zorder=1,
    )

    # Label top significant genes
    labels = (
        df.loc[sig]
        .sort_values("padj")
        .head(N_VOLCANO_LABELS)
    )

    for gene, row in labels.iterrows():
        ax.annotate(
            str(gene),
            (row["log2FoldChange"], row["neg_log10_padj"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=TEXT_SIZE+1,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(
        "log2 fold change (Mutant / WT)",
        fontsize=TEXT_SIZE,
    )

    ax.set_ylabel(
        r"$-\log_{10}$(adjusted p-value)",
        fontsize=TEXT_SIZE,
    )

    ax.set_title(
        f"{region}: Mutant vs WT",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=25,
    )

    ax.tick_params(
        axis="both",
        labelsize=TEXT_SIZE,
    )
    
    ax.grid(False)

    fig.legend(
    *ax.get_legend_handles_labels(),
    loc="center left",
    bbox_to_anchor=(0.88, 0.5),
    frameon=False,
    fontsize=LEGEND_SIZE,
    )

    sns.despine(ax=ax)

    fig.subplots_adjust(right=0.84)

    path = (
        OUT_DIR
        / "volcano_plots"
        / f"{safe_filename(region)}_volcano.png"
    )

    save_figure(fig, path)

# ============================================================
# 8. COMBINED SIGNIFICANT DEG TABLE
# ============================================================

def save_combined_deg_table(results):
    """
    Combine significant DEGs from all regions.

    One row = one gene in one region.
    Only genes with adjusted p-value < ALPHA are included.
    """

    records = []

    for region, res in results.items():
        sig = res.loc[
            res["padj"] < ALPHA
        ].copy()

        for gene, row in sig.iterrows():
            records.append({
                "Region": region,
                "Gene": str(gene),
                "log2FC (mut/WT)": row["log2FoldChange"],
                "p-value": row["pvalue"],
                "Adjusted p-value": row["padj"],
            })

    columns = [
        "Region",
        "Gene",
        "log2FC (mut/WT)",
        "p-value",
        "Adjusted p-value",
    ]

    deg_table = pd.DataFrame(records, columns=columns)

    if not deg_table.empty:
        deg_table = deg_table.sort_values(
            ["Region", "Adjusted p-value", "Gene"]
        ).reset_index(drop=True)

    output_path = OUT_DIR / "combined_significant_DEG_table.csv"
    deg_table.to_csv(output_path, index=False)

    print(f"\nCombined significant DEG table saved to: {output_path}")
    print(f"Total significant region-gene pairs: {len(deg_table)}")

    if not deg_table.empty:
        print("\nSignificant DEGs per region:")
        print(
            deg_table.groupby("Region")
            .size()
            .sort_values(ascending=False)
            .to_string()
        )
    else:
        print(f"No significant DEGs at adjusted p-value < {ALPHA}.")

    return deg_table


def export_deg_table(
    deseq_results,
    out_dir,
    alpha=0.05,
    top_n=None,
):
    """
    Export complete and significant regional DEG tables,
    and create a significant-DEG dot plot.

    Table formatting:
      - log2FC: 2 decimal places
      - p-values: scientific notation, 2 decimal places
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Combine results across regions
    # --------------------------------------------------------
    combined = []

    for region, res_df in deseq_results.items():
        df = res_df.copy()
        df["Gene"] = df.index.astype(str)
        df["Region"] = str(region)

        combined.append(
            df[
                [
                    "Region",
                    "Gene",
                    "log2FoldChange",
                    "pvalue",
                    "padj",
                ]
            ]
        )

    columns = [
        "Region",
        "Gene",
        "log2FC (mut/WT)",
        "p-value",
        "Adjusted p-value",
    ]

    if not combined:
        print("No regional DESeq2 results were provided.")
        return None, None

    all_results = pd.concat(combined, ignore_index=True)

    all_results = all_results.rename(
        columns={
            "log2FoldChange": "log2FC (mut/WT)",
            "pvalue": "p-value",
            "padj": "Adjusted p-value",
        }
    )

    all_results = all_results[columns].sort_values(
        ["Region", "Adjusted p-value"],
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 2. Export all tested region-gene pairs
    # --------------------------------------------------------
    all_results.to_csv(
        out_dir / "all_region_gene_DE_results.csv",
        index=False,
        float_format="%.2e",
    )

    # Reformat log2FC to 2 decimal places in the exported CSV,
    # while retaining scientific notation for p-values.
    all_export = all_results.copy()
    all_export["log2FC (mut/WT)"] = (
        all_export["log2FC (mut/WT)"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )
    )
    for col in ["p-value", "Adjusted p-value"]:
        all_export[col] = all_export[col].map(
            lambda x: f"{x:.2e}" if pd.notna(x) else ""
        )

    all_export.to_csv(
        out_dir / "all_region_gene_DE_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Filter and export significant DEGs
    # --------------------------------------------------------
    sig = all_results.loc[
        all_results["Adjusted p-value"].notna()
        & (all_results["Adjusted p-value"] < alpha)
    ].copy()

    sig = sig.sort_values(
        ["Adjusted p-value", "Region", "Gene"]
    ).reset_index(drop=True)

    sig_export = sig.copy()
    sig_export["log2FC (mut/WT)"] = (
        sig_export["log2FC (mut/WT)"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )
    )

    for col in ["p-value", "Adjusted p-value"]:
        sig_export[col] = sig_export[col].map(
            lambda x: f"{x:.2e}" if pd.notna(x) else ""
        )

    sig_export.to_csv(
        out_dir / "significant_DEG_table.csv",
        index=False,
    )

    print(f"All tested region-gene pairs: {len(all_results)}")
    print(
        f"Significant region-gene pairs "
        f"(adjusted p < {alpha}): {len(sig)}"
    )
    print(f"Tables saved to: {out_dir}")

    if sig.empty:
        print("No significant genes found at this alpha.")
        return all_results, sig

    # --------------------------------------------------------
    # 4. Prepare significant results for plotting
    # --------------------------------------------------------
    plot_df = sig.copy()

    # Optional limit on plotted region-gene pairs only.
    # CSV tables still contain every significant result.
    if top_n is not None:
        plot_df = (
            plot_df.sort_values("Adjusted p-value")
            .head(top_n)
            .copy()
        )

    plot_df["neg_log10_padj"] = -np.log10(
        plot_df["Adjusted p-value"].clip(lower=1e-300)
    )

    # Order genes by mean log2FC across the plotted results.
    gene_order = (
        plot_df.groupby("Gene", observed=True)["log2FC (mut/WT)"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    plot_df["Gene"] = pd.Categorical(
        plot_df["Gene"],
        categories=gene_order,
        ordered=True,
    )

    region_order = [
        str(region)
        for region in deseq_results.keys()
        if str(region) in plot_df["Region"].unique()
    ]

    return all_results, sig


# ============================================================
# 9. PER-REGION PCA AND SAMPLE CORRELATION QC
# ============================================================

def plot_region_sample_qc(results, counts, meta):
    """
    Exploratory PCA and sample correlation based on log2 CPM.
    These QC plots do not replace DESeq2 normalisation.
    """
    for region in sorted(meta["major_region"].unique()):
        safe_region = safe_filename(region)

        sub_meta = meta.loc[meta["major_region"] == region].copy()
        if len(sub_meta) < 3:
            continue

        # Use the genes tested in DESeq2 when results are available.
        if region in results:
            genes = results[region].index.intersection(counts.columns)
            # Keep a manageable set of variable genes for PCA.
            X = counts.loc[sub_meta.index, genes].to_numpy(dtype=float)
        else:
            X = counts.loc[sub_meta.index].to_numpy(dtype=float)

        library_sizes = X.sum(axis=1)
        library_sizes[library_sizes == 0] = 1

        cpm = X / library_sizes[:, None] * 1e6
        log_cpm = np.log2(cpm + 1)

        # Remove genes with no variation.
        variances = log_cpm.var(axis=0)
        valid = variances > 0
        log_cpm = log_cpm[:, valid]

        if log_cpm.shape[1] < 2:
            continue

        # Select the most variable genes to reduce noise.
        n_var = min(500, log_cpm.shape[1])
        var_idx = np.argsort(
            log_cpm.var(axis=0)
        )[-n_var:]
        X_pca = log_cpm[:, var_idx]

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_pca)

        pca_df = sub_meta.copy()
        pca_df["PC1"] = pcs[:, 0]
        pca_df["PC2"] = pcs[:, 1]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=pca_df.reset_index(),
            x="PC1",
            y="PC2",
            hue="genotype",
            style="fish_id",
            s=90,
            ax=ax,
        )

        for _, row in pca_df.iterrows():
            ax.annotate(
                row["fish_id"],
                (row["PC1"], row["PC2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
        )
        ax.set_title(f"{region}: pseudobulk PCA")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        sns.despine(ax=ax)

        save_figure(
            fig,
            OUT_DIR / "qc" / f"{safe_region}_pseudobulk_PCA.png",
        )

        # Pearson correlation between log2 CPM profiles.
        corr = np.corrcoef(log_cpm)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            corr,
            xticklabels=sub_meta["fish_id"].tolist(),
            yticklabels=sub_meta["fish_id"].tolist(),
            cmap="vlag",
            vmin=-1,
            vmax=1,
            square=True,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            ax=ax,
            cbar_kws={"label": "Pearson correlation"},
        )

        ax.set_title(f"{region}: sample correlation")
        ax.set_xlabel("Fish")
        ax.set_ylabel("Fish")

        save_figure(
            fig,
            OUT_DIR / "qc" / f"{safe_region}_sample_correlation.png",
        )


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("PSEUDOBULK DESEQ2: MUTANT VS WT")
    print("=" * 65)

    print("\n[1/6] Loading fish...")
    adata = load_all_fish()

    print("\n[2/6] Building pseudobulk counts...")
    pb_counts, pb_meta = build_pseudobulk(adata)

    print("\n[3/6] Generating pseudobulk QC...")
    plot_library_size_qc(pb_counts, pb_meta)

    print("\n[4/6] Running regional DESeq2...")
    results = run_deseq_per_region(pb_counts, pb_meta)

    print("\n[5/6] Writing volcano plots...")
    for region, res in results.items():
        volcano_plot(res, region)

    print("\n[6/6] Saving combined DEG table and dot plot...")
    all_deg_results, significant_deg_results = export_deg_table(
        deseq_results=results,
        out_dir=OUT_DIR,
        alpha=ALPHA,
        top_n=None,
    )


    print("\nAnalysis complete.")
    print(f"Results saved to: {OUT_DIR}")