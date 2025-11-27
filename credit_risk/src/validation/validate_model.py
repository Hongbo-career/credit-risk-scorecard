import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # ★ 强制无界面后端，解决 Tkinter 报错
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc



def plot_roc_curve(y_true, y_prob, save_path=None):
    """绘制 ROC 曲线并返回 AUC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return roc_auc


def plot_ks_curve(y_true, y_prob, save_path=None):
    """绘制 KS 曲线并返回 KS 值"""
    data = pd.DataFrame({"y": y_true, "prob": y_prob})
    data = data.sort_values("prob", ascending=False)

    data["cum_bad"] = (data["y"] == 1).cumsum() / (data["y"] == 1).sum()
    data["cum_good"] = (data["y"] == 0).cumsum() / (data["y"] == 0).sum()
    data["ks"] = data["cum_bad"] - data["cum_good"]

    ks_value = data["ks"].max()

    plt.figure(figsize=(8, 6))
    plt.plot(data["cum_bad"].values, label="Cumulative Bad")
    plt.plot(data["cum_good"].values, label="Cumulative Good")
    plt.plot(data["ks"].values, label=f"KS Curve (KS={ks_value:.4f})")
    plt.legend()
    plt.title("KS Curve")
    plt.xlabel("Sample Index")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return ks_value


def plot_lift_gain(y_true, y_prob, save_path=None):
    """绘制 Lift / Gain 曲线"""
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["bucket"] = pd.qcut(df.index, 10, labels=False)

    lift = df.groupby("bucket")["y"].mean() / df["y"].mean()
    gain = df.groupby("bucket")["y"].sum().cumsum() / df["y"].sum()

    x = np.arange(1, 11)

    plt.figure(figsize=(8, 6))
    plt.plot(x, lift, marker="o")
    plt.xlabel("Decile")
    plt.ylabel("Lift")
    plt.title("Lift Curve")
    if save_path:
        plt.savefig(save_path.replace(".png", "_lift.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(x, gain, marker="o")
    plt.xlabel("Decile")
    plt.ylabel("Gain")
    plt.title("Gain Curve")
    if save_path:
        plt.savefig(save_path.replace(".png", "_gain.png"), dpi=300, bbox_inches="tight")
    plt.close()


def psi(expected, actual, bins=10):
    """计算 PSI（可用于不同时间、不同系统之间的稳定性比较）"""
    def get_distribution(x, bins):
        c, _ = np.histogram(x, bins=bins)
        return c / len(x)

    expected_dist = get_distribution(expected, bins)
    actual_dist = get_distribution(actual, bins)

    psi_value = np.sum((expected_dist - actual_dist) * np.log(expected_dist / actual_dist))
    return psi_value
