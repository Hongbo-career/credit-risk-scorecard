import numpy as np
import pandas as pd


class ScorecardBuilder:
    """
    根据 Logistic 模型系数 + WOE 分箱表 构建评分卡表

    核心公式（行业常用）：
        score = offset + factor * (intercept + Σ beta_j * WOE_j)

    其中：
        factor = PDO / ln(2)
        offset = base_score - factor * ln(base_odds)

    再拆成：
        base_score  +  Σ ( - factor * beta_j * WOE_j )   （每个变量、每个 bin 一个得分）

    最终我们输出：
        - scorecard_df: 每个变量、每个 bin 的得分（供 ScoreTransformer 使用）
        - base_score:   基础分
    """

    def __init__(self,
                 pdo: float = 50.0,
                 base_odds: float = 1 / 19,   # good:bad = 19:1，对应坏账率约 5%
                 base_score: float = 600.0):
        """
        pdo: points to double odds，例如 50 分翻倍赔率
        base_odds: 在 base_score 下的好坏赔率（good/bad）
        base_score: 设定的基准分，一般在 600 左右
        """
        self.pdo = pdo
        self.base_odds = base_odds
        self.base_score_target = base_score

        # 因子与偏移量
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    # ------------------------------------------------------------------
    # 内部工具：从模型中取出各变量系数
    # ------------------------------------------------------------------
    def _get_coef_series(self, model, variables):
        """
        输入：
            model: sklearn LogisticRegression 拟合后的模型
            variables: 评分卡中出现的变量名列表
        输出：
            pandas.Series，index 为变量名，value 为对应系数（找不到的置 0）
        """
        # 尝试使用 sklearn 1.x 的 feature_names_in_
        try:
            feature_names = list(model.feature_names_in_)
            coef = model.coef_[0]
            coef_s = pd.Series(coef, index=feature_names)
        except AttributeError:
            # 如果没有 feature_names_in_，只能假设顺序与 variables 一致
            coef = model.coef_[0]
            coef_s = pd.Series(coef, index=variables)

        # 保留评分卡涉及的变量，缺失的填 0
        coef_s = coef_s.reindex(variables).fillna(0.0)
        return coef_s

    # ------------------------------------------------------------------
    # 对外接口：构建评分卡表
    # ------------------------------------------------------------------
    def build_scorecard(self, model, bin_table: pd.DataFrame) -> pd.DataFrame:
        """
        model: Logistic 模型（sklearn LogisticRegression）
        bin_table: 包含列 ['variable', 'bin', 'woe'] 的 DataFrame

        返回：
            scorecard_df: DataFrame，包含 ['variable', 'bin', 'woe', 'score']
                          其中 score 为该 variable-bin 的分值
        """
        required_cols = {"variable", "bin", "woe"}
        if not required_cols.issubset(bin_table.columns):
            raise ValueError(f"bin_table 必须包含列: {required_cols}")

        variables = bin_table["variable"].unique().tolist()
        coef_s = self._get_coef_series(model, variables)

        rows = []
        for var in variables:
            beta = coef_s.get(var, 0.0)
            sub = bin_table[bin_table["variable"] == var].copy()

            # 变量分值：常用写法为 - factor * beta * WOE
            # beta > 0 (风险增加) 时，WOE 高 → 分数下降
            sub["score"] = - self.factor * beta * sub["woe"]

            rows.append(sub[["variable", "bin", "woe", "score"]])

        scorecard_df = pd.concat(rows, ignore_index=True)
        return scorecard_df

    # ------------------------------------------------------------------
    # 对外接口：计算基础分
    # ------------------------------------------------------------------
    def calculate_base_score(self, model) -> float:
        """
        计算 base_score，用 intercept + offset + factor 的关系校准到指定区间。
        """
        intercept = float(model.intercept_[0])

        # 根据 offset 与 factor，把截距也折算到分值里
        base_score = self.offset + self.factor * intercept
        return base_score
