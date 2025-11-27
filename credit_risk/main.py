import pandas as pd
import numpy as np
from src.modeling.vif import calculate_vif
from src.modeling.stepwise import stepwise_selection
from src.modeling.logistic_model import train_logistic_model
from src.modeling.ml_models import run_ml_models
from src.validation.validate_model import (
    plot_roc_curve,
    plot_ks_curve,
    plot_lift_gain,
)
from src.scorecard.scorecard_builder import ScorecardBuilder
from src.scorecard.score_transform import ScoreTransformer
from src.business.strategy_simulation import simulate_strategy


def main():
    # ---------------------------
    # 1. 加载建模数据
    # ---------------------------
    df = pd.read_csv("data/processed/model_data.csv")

    target = "target"

    exclude_cols = [target]
    for col in ["ID", "id"]:
        if col in df.columns:
            exclude_cols.append(col)

    features = [c for c in df.columns if c not in exclude_cols]

    # ---------------------------
    # 2. VIF 检查
    # ---------------------------
    print("Calculating VIF...")
    vif_df = calculate_vif(df, features)
    print(vif_df)

    filtered_features = vif_df[vif_df["VIF"] < 10]["feature"].tolist()

    # ---------------------------
    # 3. Stepwise
    # ---------------------------
    print("Running stepwise selection...")
    selected_features = stepwise_selection(df[filtered_features], df[target])
    print("Selected features:", selected_features)

    # ---------------------------
    # 4. Logistic 模型训练
    # ---------------------------
    print("Training logistic model (scorecard baseline)...")
    model, X_test, y_test, y_pred_prob = train_logistic_model(
        df,
        selected_features,
        target
    )

    pd.DataFrame({
        "y_true": y_test,
        "pd": y_pred_prob
    }).to_csv("data/processed/pd_output.csv", index=False)
    print("PD saved to data/processed/pd_output.csv")
    print("Step 5 completed (Logistic PD model).")
    # 给全量样本打 PD，用于后续评分卡 & 策略模拟
    df["pd"] = model.predict_proba(df[selected_features])[:, 1]


    # ---------------------------
    # 5. 模型验证
    # ---------------------------
    print("\nRunning Step 7: Validation for baseline logistic model...")

    auc_value = plot_roc_curve(
        y_test,
        y_pred_prob,
        save_path="figures/roc_curve.png"
    )
    ks_value = plot_ks_curve(
        y_test,
        y_pred_prob,
        save_path="figures/ks_curve.png"
    )
    plot_lift_gain(
        y_test,
        y_pred_prob,
        save_path="figures/lift_gain.png"
    )

    print(f"AUC (Logistic baseline) = {auc_value:.4f}")
    print(f"KS  (Logistic baseline) = {ks_value:.4f}")

    # ---------------------------
    # 6. 机器学习对比模型
    # ---------------------------
    print("\nRunning Step 6: ML model benchmark (RandomForest / GBM / Logistic)...")
    metrics_df = run_ml_models(df, selected_features, target)

    print("\nModel comparison:")
    print(metrics_df)

    print("\nStep 6 completed.")
    print("Metrics saved to: data/processed/model_compare.csv")
    print("Scores saved to:  data/processed/model_scores.csv")
    print("Feature importance saved to: data/processed/model_feature_importance.csv")

    # ---------------------------
    # 8. 评分卡构建（Step 8）
    # ---------------------------
    print("\nRunning Step 8: Scorecard building...")

    # ====== 8.1 分箱 + WOE ======
    def make_bins_and_woe(df_in, feature_list, target_col, n_bins=5):
        df_local = df_in.copy()
        y = df_local[target_col]

        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()
        eps = 1e-6

        bin_rows = []

        for var in feature_list:
            try:
                bins = pd.qcut(df_local[var], q=n_bins, duplicates="drop")
            except ValueError:
                bins = pd.cut(df_local[var], bins=n_bins)

            bin_col = f"{var}_bin"
            df_local[bin_col] = bins.astype(str)

            grp = df_local.groupby(bin_col)[target_col].agg(
                bad=lambda x: (x == 1).sum(),
                good=lambda x: (x == 0).sum()
            ).reset_index()

            grp["bad_rate_all"] = grp["bad"] / max(total_bad, 1)
            grp["good_rate_all"] = grp["good"] / max(total_good, 1)
            grp["woe"] = np.log((grp["bad_rate_all"] + eps) /
                                (grp["good_rate_all"] + eps))

            for _, r in grp.iterrows():
                bin_rows.append({
                    "variable": var,
                    "bin": r[bin_col],
                    "woe": r["woe"]
                })

        return pd.DataFrame(bin_rows), df_local

    bin_table, woe_df = make_bins_and_woe(df, selected_features, target, n_bins=5)

    bin_table.to_csv("data/processed/bin_table.csv", index=False)
    woe_df.to_csv("data/processed/woe_transformed_data.csv", index=False)
    print("Bin table & WOE data saved to data/processed/.")

    # ====== 8.2 Scorecard ======
    builder = ScorecardBuilder()
    scorecard_df = builder.build_scorecard(model, bin_table)
    base_score = builder.calculate_base_score(model)

    scorecard_df.to_csv("data/processed/scorecard_table.csv", index=False)
    print("Scorecard table saved to data/processed/scorecard_table.csv")

    # ====== 8.3 生成最终 Score ======
    transformer = ScoreTransformer(scorecard_df, base_score)
    score_df = transformer.transform(woe_df)

    # 保留 PD 和 target（与 df 同索引）
    score_df["pd"] = df["pd"].values
    score_df["target"] = df["target"].values

    # ====== 8.4 重标定分数：拉到信用分常见区间 ======
    # 线性变换：new_score = (old - mean) / std * 50 + 600
    # 保留排序，只是把分布拉开
    mean_score = score_df["score"].mean()
    std_score = score_df["score"].std()

    if std_score > 0:
        score_df["score"] = (score_df["score"] - mean_score) / std_score * 50 + 600
    else:
        # 极端情况：std=0，就直接把所有分数设为 600
        score_df["score"] = 600.0

    score_df.to_csv("data/processed/score_output.csv", index=False)
    print("Final score output saved to data/processed/score_output.csv")
    print("\nStep 8 completed.")


    # ---------------------------
    # 9. 业务策略模拟（Step 9）
    # ---------------------------
    print("\nRunning Step 9: Strategy Simulation...")

    df_results = simulate_strategy(score_df)
    print(df_results)

    df_results.to_csv("data/processed/strategy_results.csv", index=False)
    print("Strategy results saved to data/processed/strategy_results.csv")
    # ---------------------------
    # 10. 高级金融风格图表 (Pro)
    # ---------------------------
    print("\nRunning Step 10: Pro visualizations...")

    from src.validation.plots_scorecard_pro import (
        plot_score_distribution_pro,
        plot_score_vs_pd_pro,
        plot_strategy_tradeoff_pro,
    )

    plot_score_distribution_pro(score_df)
    plot_score_vs_pd_pro(score_df)
    plot_strategy_tradeoff_pro(df_results)

    print("Pro plots saved to figures/:")
    print(" - score_distribution_pro.png")
    print(" - score_vs_pd_pro.png")
    print(" - strategy_tradeoff_pro.png")




if __name__ == "__main__":
    main()
