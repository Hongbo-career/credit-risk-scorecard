import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # ★ 添加这行
# （注意：这个文件没有 pyplot，但加上保持一致性）

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def compute_ks(y_true, y_score):
    """KS 统计量"""
    data = pd.DataFrame({"y": y_true, "score": y_score})
    data = data.sort_values("score", ascending=False)
    data["cum_good"] = (1 - data["y"]).cumsum() / (1 - data["y"]).sum()
    data["cum_bad"] = data["y"].cumsum() / data["y"].sum()
    ks = (data["cum_bad"] - data["cum_good"]).abs().max()
    return ks


def compute_lift_at_topk(y_true, y_score, top_ratio=0.2):
    """Top k% 的 Lift（默认 20%）"""
    data = pd.DataFrame({"y": y_true, "score": y_score})
    data = data.sort_values("score", ascending=False)
    n = int(len(data) * top_ratio)
    top = data.iloc[:n]
    overall_rate = data["y"].mean()
    top_rate = top["y"].mean()
    if overall_rate == 0:
        return np.nan
    return top_rate / overall_rate


def run_ml_models(df, features, target):
    """
    运行多模型对比：
    - Logistic (sklearn 版)
    - RandomForest
    - GradientBoosting

    返回：比较结果 DataFrame，并在 data/processed 下输出：
    - model_compare.csv: 模型级别指标
    - model_scores.csv: 逐样本预测得分
    - model_feature_importance.csv: 特征重要性/系数
    - shap_rf_top.csv: 若安装 shap，则输出 RF 的 SHAP 重要性
    """
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    results = []
    scores_df = pd.DataFrame({"y_true": y_test.reset_index(drop=True)})
    feat_imp_df = pd.DataFrame({"feature": features})

    # ---------------- Logistic (sklearn) ----------------
    lr = LogisticRegression(
        max_iter=5000,
        solver="liblinear"
    )
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    auc_lr = roc_auc_score(y_test, lr_prob)
    ks_lr = compute_ks(y_test, lr_prob)
    lift_lr = compute_lift_at_topk(y_test, lr_prob, top_ratio=0.2)

    results.append({
        "model": "logistic_sklearn",
        "auc": auc_lr,
        "ks": ks_lr,
        "lift_top20": lift_lr
    })

    scores_df["logistic_sklearn"] = lr_prob
    feat_imp_df["logistic_coef"] = lr.coef_[0]

    # ---------------- RandomForest ----------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    auc_rf = roc_auc_score(y_test, rf_prob)
    ks_rf = compute_ks(y_test, rf_prob)
    lift_rf = compute_lift_at_topk(y_test, rf_prob, top_ratio=0.2)

    results.append({
        "model": "random_forest",
        "auc": auc_rf,
        "ks": ks_rf,
        "lift_top20": lift_rf
    })

    scores_df["random_forest"] = rf_prob
    feat_imp_df["rf_importance"] = rf.feature_importances_

    # ---------------- GradientBoosting ----------------
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gbm.fit(X_train, y_train)
    gbm_prob = gbm.predict_proba(X_test)[:, 1]

    auc_gbm = roc_auc_score(y_test, gbm_prob)
    ks_gbm = compute_ks(y_test, gbm_prob)
    lift_gbm = compute_lift_at_topk(y_test, gbm_prob, top_ratio=0.2)

    results.append({
        "model": "gradient_boosting",
        "auc": auc_gbm,
        "ks": ks_gbm,
        "lift_top20": lift_gbm
    })

    scores_df["gradient_boosting"] = gbm_prob
    feat_imp_df["gbm_importance"] = gbm.feature_importances_

    # ---------------- SHAP（可选，如果已安装 shap 包） ----------------
    try:
        import shap

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        # 二分类时 shap_values 可能是 list
        if isinstance(shap_values, list):
            shap_arr = shap_values[1]
        else:
            shap_arr = shap_values

        mean_abs_shap = np.abs(shap_arr).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        shap_df.to_csv("data/processed/shap_rf_top.csv", index=False)
        print("SHAP importance (RandomForest) saved to data/processed/shap_rf_top.csv")
    except ImportError:
        print("shap is not installed, skip SHAP. You can install it with: pip install shap")
    except Exception as e:
        print(f"SHAP computation failed: {e}")

    # ---------------- 保存结果 ----------------
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("data/processed/model_compare.csv", index=False)
    scores_df.to_csv("data/processed/model_scores.csv", index=False)
    feat_imp_df.to_csv("data/processed/model_feature_importance.csv", index=False)

    return metrics_df
