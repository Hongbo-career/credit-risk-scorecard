import pandas as pd

class ScoreTransformer:
    def __init__(self, scorecard_df, base_score):
        self.scorecard = scorecard_df
        self.base_score = base_score

    def transform(self, woe_df):
        """
        woe_df: 每条记录已转换 WOE 的 DataFrame
        """
        score_df = pd.DataFrame(index=woe_df.index)
        score_df['score'] = self.base_score

        # 对每个变量加分
        for var in self.scorecard['variable'].unique():
            sc_map = self.scorecard[self.scorecard['variable'] == var][['bin', 'score']]
            sc_map = sc_map.set_index('bin')['score']

            # 记录每个样本的 bin 归属
            bin_col = f"{var}_bin"

            if bin_col not in woe_df.columns:
                continue

            score_df['score'] += woe_df[bin_col].map(sc_map)

        return score_df
