import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .monotonic_score import make_monotonic_score_pd


# 通用风格设置：偏金融报表风
def _set_finance_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.labelcolor"] = "#333333"
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"
    plt.rcParams["figure.facecolor"] = "white"


def plot_score_distribution_pro(score_df: pd.DataFrame,
                                score_col: str = "score",
                                save_path: str = "figures/score_distribution_pro.png"):
    _set_finance_style()
    plt.figure(figsize=(8, 4.5))

    sns.histplot(
        score_df[score_col],
        bins=30,
        kde=True,
        stat="count",
    )
    plt.title("Score Distribution", fontsize=14, weight="bold")
    plt.xlabel("Score")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_score_vs_pd_pro(score_df: pd.DataFrame,
                         score_col: str = "score",
                         pd_col: str = "pd",
                         n_bins: int = 20,
                         save_path: str = "figures/score_vs_pd_pro.png"):
    """
    使用单调平滑后的曲线，画金融风格 Score–PD 图
    """
    _set_finance_style()
    plt.figure(figsize=(8, 4.5))

    mono_tbl = make_monotonic_score_pd(
        score_df, score_col=score_col, pd_col=pd_col, n_bins=n_bins
    )

    # 原始各 bin PD（虚线）
    plt.plot(
        mono_tbl["mean_score"],
        mono_tbl["raw_pd"],
        linestyle="--",
        marker="o",
        alpha=0.5,
        label="Raw PD",
    )

    # 平滑后 PD（实线）
    plt.plot(
        mono_tbl["mean_score"],
        mono_tbl["mono_pd"],
        linestyle="-",
        marker="o",
        label="Monotonic PD",
    )

    plt.title("Score vs PD (Monotonic Smoothed)", fontsize=14, weight="bold")
    plt.xlabel("Score")
    plt.ylabel("Average PD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_strategy_tradeoff_pro(strategy_df: pd.DataFrame,
                               save_path: str = "figures/strategy_tradeoff_pro.png"):
    """
    高级版审批率–坏账率 Tradeoff 图
    """
    _set_finance_style()
    plt.figure(figsize=(7, 4.5))

    x = strategy_df["approval_rate"]
    y = strategy_df["bad_rate"]

    plt.plot(x, y, marker="o")

    for _, row in strategy_df.iterrows():
        plt.text(
            row["approval_rate"] + 0.01,
            row["bad_rate"],
            row["strategy"],
            fontsize=10,
        )

    plt.title("Approval–Risk Tradeoff", fontsize=14, weight="bold")
    plt.xlabel("Approval Rate")
    plt.ylabel("Bad Rate")
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
