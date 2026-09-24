import math

import pandas as pd

CSV_PATH = "benchmarking/results/benchmark_t200_rTrue_cFalse_all.csv"


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


def main():
    df = pd.read_csv(CSV_PATH, dtype=str)  # read as str to normalize parsing
    if "file name" not in df.columns:
        raise SystemExit("CSV missing `file name` column")
    # group key: first 11 characters of file_name
    df["group"] = df["file name"].astype(str).str[:11]

    # normalize numeric columns
    df["automaton_size_num"] = pd.to_numeric(df.get("automaton_size", pd.Series(["0"] * len(df))),
                                             errors="coerce").fillna(0).astype(int)

    # compute success: automaton_size != 0 AND succeeded is truthy
    df["succeeded_raw"] = df.get("succeeded", pd.Series(["False"] * len(df)))
    df["time_raw"] = df.get("total_time", pd.Series([None] * len(df)))
    df["is_success"] = df.apply(
        lambda r: (r["automaton_size_num"] != 0) and to_bool(r["succeeded_raw"]) and float(r["time_raw"]) < 200, axis=1)

    # columns to average for succeeded items
    avg_cols = ["automaton_size", "total_time", "queries_learning", "validity_query","nodes","informative_nodes"]

    for name, group in df.groupby("group", sort=True):
        total = len(group)
        succ_group = group[group["is_success"]]
        succ_count = len(succ_group)
        print(f"{name}: {succ_count}/{total} succeeded")

        if succ_count == 0:
            continue

        # ensure numeric conversion for averaging
        means = {}
        for col in avg_cols:
            means[col] = pd.to_numeric(succ_group.get(col, pd.Series([math.nan] * len(succ_group))),
                                       errors="coerce").mean()

        # print nicely formatted means (two decimal places, or 'n/a' if no value)
        for col in avg_cols:
            val = means[col]
            out = f"{val:.2f}" if not pd.isna(val) else "n/a"
            print(f"  mean {col}: {out}")
        print()


if __name__ == "__main__":
    main()
