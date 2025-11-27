import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    """
    计算每个变量的 VIF
    """
    X = df[features].copy()
    X = X.fillna(0)  # 避免 VIF 报错
    vif_list = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_list.append([col, vif])
    return pd.DataFrame(vif_list, columns=["feature", "VIF"])
