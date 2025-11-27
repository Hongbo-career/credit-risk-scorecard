import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_score_distribution(score_df, save_path="figures/score_distribution.png"):
    plt.figure(figsize=(10, 5))
    sns.histplot(score_df["score"], bins=40, kde=True)
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_strategy_tradeoff(strategy_df, save_path="figures/strategy_tradeoff.png"):
    """
    画 Approval Rate vs Bad Rate
    """
    plt.figure(figsize=(7, 5))

    plt.plot(strategy_df["approval_rate"], strategy_df["bad_rate"], marker="o")
    for i, row in strategy_df.iterrows():
        plt.text(row["approval_rate"], row["bad_rate"], row["strategy"])

    plt.xlabel("Approval Rate")
    plt.ylabel("Bad Rate")
    plt.title("Approval–Risk Tradeoff")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pd_distribution(score_df, save_path="figures/pd_distribution.png"):
    plt.figure(figsize=(10, 5))
    sns.histplot(score_df["pd"], bins=40, kde=True, color="orange")
    plt.title("PD Distribution")
    plt.xlabel("Predicted Default Probability (PD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_score_vs_pd(score_df, save_path="figures/score_vs_pd.png"):
    """
    分数 vs 预测违约概率 的单调性图
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=score_df["score"], y=score_df["pd"], s=10, alpha=0.3)
    plt.title("Score vs PD")
    plt.xlabel("Score")
    plt.ylabel("PD")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
