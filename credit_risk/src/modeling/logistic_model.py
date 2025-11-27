# src/modeling/logistic_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_logistic_model(df, features, target):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 换成更稳的 solver，并加大迭代次数
    model = LogisticRegression(
        max_iter=5000,
        solver="liblinear"   # 对于类似评分卡的小数据更稳定
    )
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred_prob
