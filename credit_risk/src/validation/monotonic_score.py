import pandas as pd


def make_monotonic_score_pd(score_df,
                            score_col: str = "score",
                            pd_col: str = "pd",
                            n_bins: int = 20) -> pd.DataFrame:
    """
    根据 score_df 生成 “分数区间 vs 平滑后 PD” 表，保证：
        分数越高，PD 非增（单调下降）

    步骤：
        1. 按 score 分位数分成 n_bins 组
        2. 每组计算 mean_score, mean_pd
        3. 从高分到低分，对 mean_pd 做 running-min 平滑，确保单调

    返回：
        DataFrame: ['mean_score', 'raw_pd', 'mono_pd', 'bin']
    """
    df = score_df[[score_col, pd_col]].copy()

    # 1) 分位数分箱
    df["bin"] = pd.qcut(df[score_col], q=n_bins, duplicates="drop")

    grp = df.groupby("bin").agg(
        mean_score=(score_col, "mean"),
        raw_pd=(pd_col, "mean"),
        count=(pd_col, "size"),
    ).reset_index()

    # 2) 按分数从低到高排序
    grp = grp.sort_values("mean_score").reset_index(drop=True)

    # 3) 从右往左做 running-min，让 PD 随分数单调下降
    mono_pd = grp["raw_pd"].values.copy()
    running_min = float("inf")
    for i in range(len(mono_pd) - 1, -1, -1):  # 从高分到低分
        running_min = min(running_min, mono_pd[i])
        mono_pd[i] = running_min

    grp["mono_pd"] = mono_pd

    return grp
