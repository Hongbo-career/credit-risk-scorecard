import pandas as pd
import statsmodels.api as sm

def stepwise_selection(X, y, threshold_in=0.01, threshold_out=0.05):
    """
    基于 p-value 的 stepwise（前向 + 后向）
    X: 自变量 DataFrame
    y: 目标 Series
    """
    included = []

    while True:
        changed = False

        # ---------- 前向 ----------
        excluded = list(set(X.columns) - set(included))
        new_pvals = pd.Series(dtype=float)

        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_col]])).fit(disp=False)
            new_pvals.loc[new_col] = model.pvalues[new_col]

        if not new_pvals.empty:
            best_pval = new_pvals.min()
            if best_pval < threshold_in:
                best_feature = new_pvals.idxmin()
                included.append(best_feature)
                changed = True

        # ---------- 后向 ----------
        if included:
            model = sm.Logit(y, sm.add_constant(X[included])).fit(disp=False)
            pvalues = model.pvalues.iloc[1:]  # 排除截距
            worst_pval = pvalues.max()

            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True

        if not changed:
            break

    return included
