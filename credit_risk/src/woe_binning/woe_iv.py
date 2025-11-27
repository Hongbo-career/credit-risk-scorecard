# src/woe_binning/woe_iv.py

import pandas as pd
import numpy as np

def compute_bin_woe_iv(df, y, col, edges):
    """
    计算某个变量的 WOE / IV
    """
    tmp = df[[col, y]].copy()
    tmp["bin"] = pd.cut(tmp[col], bins=edges, include_lowest=True)

    grouped = tmp.groupby("bin")
    dist_good = (grouped[y].count() - grouped[y].sum()) / (df[y].count() - df[y].sum())
    dist_bad = grouped[y].sum() / df[y].sum()

    woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
    iv = np.sum((dist_good - dist_bad) * woe)

    return woe.to_dict(), iv

def compute_woe_iv(df, y, binning_results):
    """
    计算所有变量的 WOE/IV + 生成 WOE 转换后的数据
    """
    woe_data = df.copy()
    iv_list = []

    for col, edges in binning_results.items():
        woe_map, iv = compute_bin_woe_iv(df, y, col, edges)

        # 保存 IV 报告
        iv_list.append({"variable": col, "iv": iv})

        # 生成 WOE 转换列
        bin_col = pd.cut(df[col], bins=edges, include_lowest=True)
        woe_data[col] = bin_col.map(woe_map)

    iv_report = pd.DataFrame(iv_list).sort_values(by="iv", ascending=False)

    return woe_data, iv_report
