from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH_A = "benchmarking/results/benchmark2_t200_rFalse_cFalse_all.csv"
CSV_PATH_B = "benchmarking/results/benchmark2_t200_rTrue_cFalse_all.csv"
OUT_PNG = "vs_comp.pdf"


def to_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    try:
        return float(s) != 0.0
    except Exception:
        return False


def parse_suffix_to_int(name: str):
    last_two = name[9:11] if len(name) >= 2 else name
    try:
        return int(last_two)
    except Exception:
        return None


def load_and_clean(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "file name" not in df.columns:
        raise SystemExit(f"CSV missing `file name` column: {path}")
    df["automaton_size_num"] = pd.to_numeric(df.get("automaton_size", pd.Series(["0"] * len(df))),
                                             errors="coerce").fillna(0).astype(int)
    df["succeeded_raw"] = df.get("succeeded", pd.Series(["False"] * len(df)))
    df["time_raw"] = df.get("total_time", pd.Series([None] * len(df)))
    df["queries_raw"] = df.get("queries_learning", pd.Series([None] * len(df)))
    df["validity_raw"] = df.get("validity_query", pd.Series([None] * len(df)))
    # keep only rows that succeeded and have a numeric measurement (this is the 'succeeded' check)
    df["is_success"] = df.apply(lambda r: (r["automaton_size_num"] != 0) and pd.notna(r["time_raw"]) and str(
        r["time_raw"]).strip() != "" and float(r["time_raw"]) < 200, axis=1)
    df = df[df["is_success"]].copy()
    if df.empty:
        return pd.DataFrame()
    df["suffix_x"] = df["file name"].apply(parse_suffix_to_int)
    df["total_time_num"] = pd.to_numeric(df.get("total_time", pd.Series([None] * len(df))), errors="coerce")
    df = df.dropna(subset=["suffix_x", "total_time_num"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["suffix_int"] = df["suffix_x"].astype(int)
    df["source"] = label
    # keep `file name` so we can intersect successful benchmarks across files
    return df[["file name", "suffix_int", "total_time_num", "source"]]


def main():
    a = load_and_clean(CSV_PATH_A, Path(CSV_PATH_A).stem)
    b = load_and_clean(CSV_PATH_B, Path(CSV_PATH_B).stem)

    if a.empty or b.empty:
        print("One or both files have no successful benchmarks to compare.")
        return

    # Print the number of successful benchmarks in each file out of total
    total_a = len(pd.read_csv(CSV_PATH_A))
    total_b = len(pd.read_csv(CSV_PATH_B))
    print(f"File A `{CSV_PATH_A}`: {len(a)}/{total_a} succeeded")
    print(f"File B `{CSV_PATH_B}`: {len(b)}/{total_b} succeeded")

    # keep only benchmarks (by `file name`) that succeeded in BOTH files
    common_files = set(a["file name"]).intersection(set(b["file name"]))
    if not common_files:
        print("No benchmarks succeeded in both files.")
        return

    a = a[a["file name"].isin(common_files)].copy()
    b = b[b["file name"].isin(common_files)].copy()

    # Align by both file name and suffix to compute pairwise comparisons
    merged = pd.merge(a, b, on=["file name", "suffix_int"], suffixes=("_a", "_b"))
    if merged.empty:
        print("No matching (file name, suffix) pairs to compare.")
        return

    merged["total_time_num_a"] = merged["total_time_num_a"].astype(float)
    merged["total_time_num_b"] = merged["total_time_num_b"].astype(float)

    mean_a = merged["total_time_num_a"].mean()
    mean_b = merged["total_time_num_b"].mean()

    if pd.isna(mean_a) or pd.isna(mean_b):
        print("Could not compute means for comparison.")
    else:
        if mean_a == 0:
            print("Mean for A is zero, cannot compute percentage improvement.")
        else:
            improvement_pct = (mean_a - mean_b) / mean_a * 100.0
            # Positive improvement_pct means B is better (lower) on average
            print(f"Compared across {len(merged)} matched benchmarks:")
            print(f"Mean A: {mean_a:.2f}, Mean B: {mean_b:.2f}")
            print(f"Benchmark B is better than A by {improvement_pct:.2f}% on average")

    # unified suffix order so boxes align (suffixes come from the common subset)
    suffix_order = sorted(set(a["suffix_int"].unique()).union(set(b["suffix_int"].unique())))
    if not suffix_order:
        print("No suffixes found among common benchmarks.")
        return

    # prepare per-suffix lists for each dataset
    data_a = []
    data_b = []
    counts_a = []
    counts_b = []
    for s in suffix_order:
        vals_a = a.loc[a["suffix_int"] == s, "total_time_num"].dropna().astype(
            float).values if not a.empty else np.array([])
        vals_b = b.loc[b["suffix_int"] == s, "total_time_num"].dropna().astype(
            float).values if not b.empty else np.array([])
        data_a.append(vals_a)
        data_b.append(vals_b)
        counts_a.append(len(vals_a))
        counts_b.append(len(vals_b))

    x = np.arange(len(suffix_order))
    width = 0.35
    pos_a = x - width / 2
    pos_b = x + width / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    sns_palette = ("#ff424b", "#008f89")
    nonempty_a_idx = [i for i, d in enumerate(data_a) if len(d) > 0]
    nonempty_b_idx = [i for i, d in enumerate(data_b) if len(d) > 0]

    if nonempty_a_idx:
        data_a_plot = [data_a[i] for i in nonempty_a_idx]
        pos_a_plot = [pos_a[i] for i in nonempty_a_idx]
        ax.boxplot(data_a_plot, positions=pos_a_plot, widths=width, patch_artist=True, showfliers=False, whis=(25, 100),
                   boxprops={"facecolor": sns_palette[0], "edgecolor": "#444444"}, medianprops={"color": "#000000"},
                   whiskerprops={"color": "#444444"}, capprops={"color": "#444444"})

    if nonempty_b_idx:
        data_b_plot = [data_b[i] for i in nonempty_b_idx]
        pos_b_plot = [pos_b[i] for i in nonempty_b_idx]
        ax.boxplot(data_b_plot, positions=pos_b_plot, widths=width, patch_artist=True, showfliers=False, whis=(25, 100),
                   boxprops={"facecolor": sns_palette[1], "edgecolor": "#444444"}, medianprops={"color": "#000000"},
                   whiskerprops={"color": "#444444"}, capprops={"color": "#444444"})

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in suffix_order])
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Running Time (seconds)")
    # ax.set_title("Comparison of Basis Replacement (only benchmarks succeeded in both files)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.25)

    from matplotlib.patches import Patch
    legend_handles = []
    if any(counts_a):
        legend_handles.append(Patch(facecolor=sns_palette[0], edgecolor="#444444", label="Using apartness"))
    if any(counts_b):
        legend_handles.append(Patch(facecolor=sns_palette[1], edgecolor="#444444", label="Using compatibility"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to `{OUT_PNG}`")
    plt.show()


if __name__ == "__main__":
    main()
