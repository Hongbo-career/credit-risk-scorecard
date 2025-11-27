import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.01) -> pd.Series:
    """
    简单 Winsorize：按分位数截断极端值，不依赖 scipy
    lower / upper 是左/右侧截掉的比例
    """
    if not np.issubdtype(s.dtype, np.number):
        return s

    lower_q = s.quantile(lower)
    upper_q = s.quantile(1 - upper)
    return s.clip(lower=lower_q, upper=upper_q)


def load_and_clean_data(input_path: str, output_path: str):
    """
    加载原始数据 -> 列名规范 -> 缺失值 -> Winsorize -> 衍生变量 -> 输出
    """
    print(f"[INFO] 读取原始数据: {input_path}")
    df = pd.read_csv(input_path)

    # 1. 列名规范化：小写 + 非字母数字转下划线
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
    )
    print("[INFO] 列名示例:", df.columns[:10].tolist())

    # 2. 标签列重命名
    if "default_payment_next_month" in df.columns:
        df = df.rename(columns={"default_payment_next_month": "target"})
    elif "default" in df.columns:
        df = df.rename(columns={"default": "target"})
    else:
        raise ValueError("未找到违约标签列，请检查原始数据的列名。")

    # 3. 删除重复行
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"[INFO] 删除重复后行数: {before} -> {df.shape[0]}")

    # 4. 缺失值处理（本数据集基本无缺失，这里统一填 0）
    missing_total = df.isna().sum().sum()
    print(f"[INFO] 总缺失值个数: {missing_total}")
    df = df.fillna(0)

    # 5. Winsorize：对额度、账单、还款金额做分位数截断
    winsor_cols = [c for c in df.columns if ("amt" in c) or ("limit" in c)]
    print("[INFO] Winsorize 列:", winsor_cols)
    for col in winsor_cols:
        df[col] = winsorize_series(df[col], lower=0.01, upper=0.01)

    # 6. 衍生变量 ====================
    # 账单列 & 还款列
    bill_cols = [c for c in df.columns if "bill_amt" in c]
    pay_cols = [c for c in df.columns if "pay_amt" in c]
    pay_status_cols = [c for c in df.columns if c.startswith("pay_") and "amt" not in c]

    # 平均账单
    df["bill_avg"] = df[bill_cols].mean(axis=1)

    # 授信额度使用率
    if "limit_bal" in df.columns:
        df["utilization"] = df["bill_avg"] / df["limit_bal"]
        df["utilization"] = df["utilization"].clip(0, 5)
    else:
        df["utilization"] = 0

    # 平均还款金额
    df["pay_avg"] = df[pay_cols].mean(axis=1)

    # 还款率（避免除 0）
    df["pay_ratio"] = np.where(df["bill_avg"] > 0, df["pay_avg"] / df["bill_avg"], 0)
    df["pay_ratio"] = df["pay_ratio"].clip(0, 3)

    # 逾期次数（PAY_0~PAY_6 > 0 记为逾期）
    if pay_status_cols:
        df["num_overdue"] = (df[pay_status_cols] > 0).sum(axis=1)
    else:
        df["num_overdue"] = 0

    # 7. 保存结果
    df.to_csv(output_path, index=False)
    print(f"[INFO] 清洗后数据已保存到: {output_path}")
    print("[INFO] 最终数据形状:", df.shape)


if __name__ == "__main__":
    input_file = "data/raw/UCI_Credit_Card.csv"
    output_file = "data/processed/model_data.csv"
    load_and_clean_data(input_file, output_file)
