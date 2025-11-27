import pandas as pd

def simulate_strategy(data, score_col="score", target_col="target"):
    """
    输入：
        data: DataFrame，至少包含 [score_col, target_col, 'pd']
        score_col: 分数列名
        target_col: 违约标记（1=坏，0=好）

    策略设计（按分位数自动切分）：
        lenient  : 分数 >= 20 分位数阈值（大约通过率 80%）
        baseline : 分数 >= 50 分位数阈值（大约通过率 50%）
        strict   : 分数 >= 80 分位数阈值（大约通过率 20%）

    返回：
        各策略的指标 DataFrame
    """

    scores = data[score_col]

    # 自动按当前分布计算阈值（不用写死 500/550/600）
    thr_lenient = scores.quantile(0.20)
    thr_baseline = scores.quantile(0.50)
    thr_strict = scores.quantile(0.80)

    strategies = {
        "lenient": thr_lenient,
        "baseline": thr_baseline,
        "strict": thr_strict,
    }

    results = []

    for name, threshold in strategies.items():
        approved = data[data[score_col] >= threshold]

        approval_rate = len(approved) / len(data)

        if len(approved) > 0:
            bad_rate = approved[target_col].mean()
            if "pd" in approved.columns:
                expected_loss = approved["pd"].mean()
            else:
                expected_loss = None
        else:
            bad_rate = None
            expected_loss = None

        results.append({
            "strategy": name,
            "threshold": float(threshold),
            "approval_rate": approval_rate,
            "bad_rate": bad_rate,
            "expected_loss": expected_loss,
        })

    return pd.DataFrame(results)
