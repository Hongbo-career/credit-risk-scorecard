import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    """
    Step 3: 特征工程
    输入: model_data.csv（用户真实字段）
    输出: feature_ready.csv
    """

    def __init__(self, input_path):
        self.input_path = input_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self.df

    def assign_variable_groups(self):
        """
        根据你 model_data.csv 的字段精准分类
        """

        self.application_vars = [
            "limit_bal", "sex", "education", "marriage", "age"
        ]

        self.behavioral_vars = [
            "pay_0","pay_2","pay_3","pay_4","pay_5","pay_6",
            "bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6",
            "pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6"
        ]

        self.derived_vars = [
            "bill_avg", "utilization", "pay_avg", "pay_ratio", "num_overdue"
        ]

        # 目标变量
        self.label = "target"

        return self.df

    def drop_useless_vars(self):
        # 删除 ID
        if "id" in self.df.columns:
            self.df.drop(columns=["id"], inplace=True)

        # 删除方差为0的变量（无信息）
        constant_cols = [c for c in self.df.columns if self.df[c].nunique() <= 1]
        self.df.drop(columns=constant_cols, inplace=True)

        return self.df

    def generate_basic_eda(self, save_path="data/processed/eda_summary.csv"):
        missing_rate = self.df.isnull().mean().rename("missing_rate")
        desc = self.df.describe().T
        eda = pd.concat([desc, missing_rate], axis=1)

        eda.to_csv(save_path)
        return eda

    def save_processed(self, output_path="data/processed/feature_ready.csv"):
        self.df.to_csv(output_path, index=False)

    def run_all(self):
        self.load_data()
        self.assign_variable_groups()
        self.drop_useless_vars()
        self.generate_basic_eda()
        self.save_processed()
        return self.df
