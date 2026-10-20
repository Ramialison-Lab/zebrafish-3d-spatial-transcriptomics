from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
## used for final figure


# ============================================================
# CONFIG
# ============================================================
# Input pseudobulk count matrix generated from all cells,
# with one column per fish and one row per gene.
COUNTS_FILE = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/DEG/"
    "degust_counts_all/all_cells_mut_vs_WT_counts.csv"
)

OUT_DIR = Path(
    "/group/tran3/swei/MERFISH/MERSCOPE/data/results/"
    "all_cells_pseudobulk_deg_results"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "qc").mkdir(exist_ok=True)
(OUT_DIR / "volcano_plots").mkdir(exist_ok=True)

FISH_LIST = [
    "WT1",
    "WT2",
    "WT3",
    "mut1",
    "mut2",
    "mut3",
]

ALPHA = 0.05
MIN_TOTAL_COUNT = 10
N_VOLCANO_LABELS = 10
N_PCA_VARIABLE_GENES = 500
DPI = 300

sns.set_theme(style="white")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


# ============================================================
# HELPERS
# ============================================================

def genotype_from_fish(fish_id):
    """Return the genotype label associated with a fish ID."""
    return "mut" if fish_id.startswith("mut") else "WT"


def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# LOAD COUNTS
# ============================================================

def load_pseudobulk_counts():
    """
    Load the pseudobulk count matrix and construct sample metadata.

    Transpose the input so rows represent fish and columns represent
    genes, as required by PyDESeq2. Export the reformatted counts,
    sample metadata, and biological replicate summary for reference.
    """

    counts = pd.read_csv(
        COUNTS_FILE,
        index_col=0
    )

    counts = counts[FISH_LIST]

    counts = counts.T

    counts = counts.astype(np.int64)

    meta = pd.DataFrame(
        {
            "fish_id": FISH_LIST,
            "genotype": [
                genotype_from_fish(x)
                for x in FISH_LIST
            ],
        },
        index=FISH_LIST
    )

    counts.to_csv(
        OUT_DIR / "all_cells_pseudobulk_counts.csv"
    )

    meta.to_csv(
        OUT_DIR / "all_cells_sample_metadata.csv"
    )

    replicate_summary = (
        meta.groupby("genotype")["fish_id"]
        .nunique()
        .rename("n_biological_replicates")
        .reset_index()
    )

    replicate_summary.to_csv(
        OUT_DIR / "replicate_summary.csv",
        index=False
    )

    return counts, meta


# ============================================================
# LIBRARY SIZE QC
# ============================================================

def plot_library_size_qc(counts, meta):
    """
    Compare total pseudobulk counts across fish.

    Library size is the sum of counts across all genes for each
    sample. A log-scaled plot helps reveal differences in sequencing
    depth or total transcript counts between biological replicates.
    """

    qc = meta.copy()

    qc["library_size"] = counts.sum(axis=1)

    qc.to_csv(
        OUT_DIR / "qc" / "library_sizes.csv"
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=qc.reset_index(),
        x="fish_id",
        y="library_size",
        hue="genotype",
        order=FISH_LIST,
        errorbar=None,
        ax=ax,
    )

    ax.set_yscale("log")

    ax.set_title(
        "All-cell pseudobulk library sizes"
    )

    sns.despine(ax=ax)

    save_figure(
        fig,
        OUT_DIR / "qc"
        / "library_sizes.png"
    )


# ============================================================
# DESEQ2
# ============================================================

def run_deseq(counts, meta):
    """
    Perform pseudobulk differential expression analysis with PyDESeq2.

    Genes with fewer than MIN_TOTAL_COUNT counts across all samples
    are excluded. The model compares mutant fish with WT fish, using
    WT as the reference genotype. Results, significant DEGs,
    normalised counts, and a summary table are exported.
    """
    keep_genes = (
        counts.sum(axis=0)
        >= MIN_TOTAL_COUNT
    )

    counts = counts.loc[:, keep_genes]

    meta = meta.copy()

    meta["genotype"] = pd.Categorical(
        meta["genotype"],
        categories=["WT", "mut"]
    )

    dds = DeseqDataSet(
        counts=counts,
        metadata=meta,
        design="~genotype",
        refit_cooks=True,
        quiet=True,
        n_cpus=1,
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

    res.sort_values(
        "padj",
        inplace=True,
        na_position="last"
    )

    res.to_csv(
        OUT_DIR
        / "all_cells_mut_vs_WT_DESeq2.csv"
    )

    sig = res.loc[
        res["padj"] < ALPHA
    ]

    sig.to_csv(
        OUT_DIR
        / "all_cells_significant_DEGs.csv"
    )

    # Export DESeq2 size-factor-normalised counts for downstream
    # inspection and visualisation. These are not variance-stabilised
    # counts and are not used here as input to the DESeq2 model.
    norm_counts = pd.DataFrame(
        np.asarray(
            dds.layers["normed_counts"]
        ),
        index=counts.index,
        columns=counts.columns,
    )

    norm_counts.to_csv(
        OUT_DIR
        / "qc"
        / "normalised_counts.csv"
    )

    # Summarise the total number of tested genes and significant
    # genes, separating significant genes by fold-change direction.
    
    summary = pd.DataFrame([{
        "n_genes_tested": len(res),
        "n_significant": int(
            (res["padj"] < ALPHA).sum()
        ),
        "n_up": int(
            (
                (res["padj"] < ALPHA)
                &
                (res["log2FoldChange"] > 0)
            ).sum()
        ),
        "n_down": int(
            (
                (res["padj"] < ALPHA)
                &
                (res["log2FoldChange"] < 0)
            ).sum()
        ),
    }])

    summary.to_csv(
        OUT_DIR
        / "DEG_summary.csv",
        index=False,
    )

    return res


# ============================================================
# VOLCANO
# ============================================================

def volcano_plot(res):
    """
    Visualise differential expression results for mutant versus WT.

    Genes are plotted by log2 fold change and negative log10
    adjusted p-value. Significant genes are coloured by the
    direction of change, and the most significant genes are labelled.
    """
    
    df = res.dropna(
        subset=["log2FoldChange", "padj"]
    ).copy()

    TITLE_SIZE = 10
    TEXT_SIZE = 9
    LEGEND_SIZE = 7

    df["neg_log10_padj"] = (
        -np.log10(
            df["padj"].clip(lower=1e-300)
        )
    )

    sig = df["padj"] < ALPHA
    up = sig & (df["log2FoldChange"] > 0)
    down = sig & (df["log2FoldChange"] < 0)
    ns = ~sig

    fig, ax = plt.subplots(
        figsize=(5, 3.5)
    )

    ax.scatter(
        df.loc[ns, "log2FoldChange"],
        df.loc[ns, "neg_log10_padj"],
        s=20,
        c="#B8B8B8",
        alpha=0.4,
    )

    ax.scatter(
        df.loc[up, "log2FoldChange"],
        df.loc[up, "neg_log10_padj"],
        s=70,
        c="#D64B40",
        label=f"Up in mutant ({up.sum()})",
    )

    ax.scatter(
        df.loc[down, "log2FoldChange"],
        df.loc[down, "neg_log10_padj"],
        s=70,
        c="#2878B5",
        label=f"Down in mutant ({down.sum()})",
    )

    ax.axhline(
        -np.log10(ALPHA),
        color="grey",
        linestyle="--"
    )

    ax.axvline(
        0,
        color="grey",
    )

    labels = (
        df.loc[sig]
        .sort_values("padj")
        .head(N_VOLCANO_LABELS)
    )

    for gene, row in labels.iterrows():

        ax.annotate(
            gene,
            (
                row["log2FoldChange"],
                row["neg_log10_padj"],
            ),
            fontsize=9,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "log2 fold change (Mut / WT)"
    )

    ax.set_ylabel(
        r"$-\log_{10}$(adjusted p)"
    )

    ax.set_title(
        "All Cells: Mutant vs WT",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=25,
    )
    ax.grid(False)

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=False,
        fontsize=LEGEND_SIZE,
    )

    sns.despine()

    fig.subplots_adjust(right=0.84)

    save_figure(
        fig,
        OUT_DIR
        / "volcano_plots"
        / "all_cells_volcano.png"
    )


# ============================================================
# PCA + CORRELATION
# ============================================================

def plot_sample_qc(counts, meta):
    """
    Assess similarity between biological replicates using PCA
    and pairwise sample correlations.

    Counts are converted to counts per million (CPM) and log2
    transformed. PCA is performed on the most variable genes to
    visualise sample-level expression patterns. Pearson correlations
    are calculated between the transformed expression profiles.
    """
    keep = (
        counts.sum(axis=0)
        >= MIN_TOTAL_COUNT
    )

    X = counts.loc[:, keep].to_numpy(
        dtype=float
    )

    libsize = X.sum(axis=1)

    cpm = (
        X
        / libsize[:, None]
        * 1e6
    )

    log_cpm = np.log2(cpm + 1)

    vars_ = log_cpm.var(axis=0)

    keep = vars_ > 0

    log_cpm = log_cpm[:, keep]

    n_var = min(
        N_PCA_VARIABLE_GENES,
        log_cpm.shape[1]
    )

    idx = np.argsort(
        log_cpm.var(axis=0)
    )[-n_var:]

    pca = PCA(
        n_components=2
    )

    pcs = pca.fit_transform(
        log_cpm[:, idx]
    )

    pca_df = meta.copy()

    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.scatterplot(
        data=pca_df.reset_index(),
        x="PC1",
        y="PC2",
        hue="genotype",
        style="fish_id",
        s=120,
        ax=ax
    )

    save_figure(
        fig,
        OUT_DIR
        / "qc"
        / "pseudobulk_PCA.png"
    )

    corr = np.corrcoef(log_cpm)

    corr_df = pd.DataFrame(
        corr,
        index=meta.index,
        columns=meta.index,
    )

    corr_df.to_csv(
        OUT_DIR
        / "qc"
        / "sample_correlations.csv"
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        corr_df,
        annot=True,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    save_figure(
        fig,
        OUT_DIR
        / "qc"
        / "sample_correlation.png"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Load the counts and sample metadata.
    counts, meta = load_pseudobulk_counts()

    # Inspect total counts across biological replicates.
    plot_library_size_qc(
        counts,
        meta,
    )

    # Run differential expression analysis.
    results = run_deseq(
        counts,
        meta,
    )

    # Visualise the differential expression results.
    volcano_plot(results)

    # Generate sample-level PCA and correlation plots.
    plot_sample_qc(
        counts,
        meta,
    )

    print("\nAnalysis complete")
    print(OUT_DIR)