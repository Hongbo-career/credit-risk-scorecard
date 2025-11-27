# src/woe_binning/binning.py

import pandas as pd
import numpy as np

def is_monotonic(series):
    return series.is_monotonic_increasing or series.is_monotonic_decreasing

def auto_binning(df, y, max_bins=10):
    """
    自动分箱（连续变量）
    - 连续变量使用 qcut 尝试分箱
    - 单调性检查，不单调则减少箱数重试
    - 返回 {variable: list of bin edges}
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [col for col in numeric_cols if col != y]

    binning_results = {}

    for col in numeric_cols:
        x = df[col]
        tmp_df = pd.DataFrame({col: x, y: df[y]})

        bins = max_bins
        edges = None

        while bins >= 2:
            try:
                tmp_df["bin"], edges = pd.qcut(tmp_df[col], q=bins, retbins=True, duplicates="drop")
            except:
                bins -= 1
                continue

            woe_table = tmp_df.groupby("bin")[y].mean()

            # 检查单调
            if is_monotonic(woe_table):
                break

            bins -= 1

        binning_results[col] = edges

    return binning_results
