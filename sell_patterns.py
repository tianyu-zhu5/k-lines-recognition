"""
顶部K线形态识别模块
识别常见的顶部反转形态，用于卖出时机判断
"""

import pandas as pd
from typing import List


def is_shooting_star(candle: pd.Series) -> bool:
    """
    识别射击之星形态

    特征：
    - 小实体（阳线或阴线均可）
    - 长上影线（至少是实体的2倍）
    - 无下影线或很短的下影线

    Args:
        candle: 单根K线数据

    Returns:
        True: 是射击之星，False: 不是
    """
    open_price = candle['open']
    close_price = candle['close']
    high_price = candle['high']
    low_price = candle['low']

    # 实体大小
    body = abs(close_price - open_price)
    # 上影线长度
    upper_shadow = high_price - max(close_price, open_price)
    # 下影线长度
    lower_shadow = min(close_price, open_price) - low_price

    # 条件1：上影线至少是实体的2倍
    if body == 0:
        condition1 = upper_shadow > 0
    else:
        condition1 = upper_shadow >= 2 * body

    # 条件2：下影线很短（小于实体）
    condition2 = lower_shadow < body if body > 0 else lower_shadow < upper_shadow / 3

    return condition1 and condition2


def is_gravestone_doji(candle: pd.Series, threshold: float = 0.001) -> bool:
    """
    识别墓碑线（墓碑十字星）

    特征：
    - 开盘价和收盘价几乎相同
    - 长上影线
    - 无下影线或极短的下影线

    Args:
        candle: 单根K线数据
        threshold: 开盘收盘价差异阈值（占收盘价的比例）

    Returns:
        True: 是墓碑线，False: 不是
    """
    open_price = candle['open']
    close_price = candle['close']
    high_price = candle['high']
    low_price = candle['low']

    # 实体大小（应该很小）
    body = abs(close_price - open_price)
    # 上影线长度
    upper_shadow = high_price - max(close_price, open_price)
    # 下影线长度
    lower_shadow = min(close_price, open_price) - low_price

    # 条件1：实体很小（开盘收盘价接近）
    condition1 = body / close_price < threshold if close_price > 0 else False

    # 条件2：有明显的上影线
    total_range = high_price - low_price
    condition2 = upper_shadow > total_range * 0.7 if total_range > 0 else False

    # 条件3：下影线很短
    condition3 = lower_shadow < total_range * 0.1 if total_range > 0 else False

    return condition1 and condition2 and condition3


def is_evening_star(df: pd.DataFrame, idx: int) -> bool:
    """
    识别黄昏之星形态

    特征（三根K线组合）：
    - 第一根：大阳线
    - 第二根：小实体（星线），可以是阳线或阴线，向上跳空
    - 第三根：大阴线，收盘价深入第一根阳线实体内部

    Args:
        df: 股票数据DataFrame
        idx: 第三根K线的索引位置（黄昏之星的最后一根）

    Returns:
        True: 是黄昏之星，False: 不是
    """
    # 需要至少3根K线
    if idx < 2:
        return False

    # 三根K线
    first = df.iloc[idx - 2]
    second = df.iloc[idx - 1]
    third = df.iloc[idx]

    # 第一根K线：大阳线
    first_body = abs(first['close'] - first['open'])
    first_is_bullish = first['close'] > first['open']
    first_range = first['high'] - first['low']

    # 第二根K线：小实体
    second_body = abs(second['close'] - second['open'])

    # 第三根K线：大阴线
    third_body = abs(third['close'] - third['open'])
    third_is_bearish = third['close'] < third['open']
    third_range = third['high'] - third['low']

    # 条件1：第一根是大阳线（实体占比大）
    condition1 = first_is_bullish and first_body > first_range * 0.6

    # 条件2：第二根实体小（小于第一根的1/3）
    condition2 = second_body < first_body / 3

    # 条件3：第二根向上跳空（低点高于第一根的最高点或接近）
    condition3 = second['low'] >= first['high'] * 0.99

    # 条件4：第三根是大阴线
    condition4 = third_is_bearish and third_body > third_range * 0.6

    # 条件5：第三根收盘价深入第一根实体（至少50%）
    first_mid = (first['open'] + first['close']) / 2
    condition5 = third['close'] < first_mid

    return condition1 and condition2 and condition3 and condition4 and condition5


def is_dark_cloud_cover(df: pd.DataFrame, idx: int) -> bool:
    """
    识别乌云盖顶形态

    特征（两根K线组合）：
    - 第一根：大阳线
    - 第二根：开盘价高于第一根最高价，收盘价深入第一根实体（至少50%）

    Args:
        df: 股票数据DataFrame
        idx: 第二根K线的索引位置

    Returns:
        True: 是乌云盖顶，False: 不是
    """
    # 需要至少2根K线
    if idx < 1:
        return False

    # 两根K线
    first = df.iloc[idx - 1]
    second = df.iloc[idx]

    # 第一根：阳线
    first_is_bullish = first['close'] > first['open']
    first_body = first['close'] - first['open']

    # 第二根：阴线
    second_is_bearish = second['close'] < second['open']

    # 条件1：第一根是阳线
    if not first_is_bullish:
        return False

    # 条件2：第二根是阴线
    if not second_is_bearish:
        return False

    # 条件3：第二根开盘价高于第一根最高价（或接近）
    condition3 = second['open'] >= first['high'] * 0.99

    # 条件4：第二根收盘价深入第一根实体（至少50%）
    first_mid = (first['open'] + first['close']) / 2
    condition4 = second['close'] < first_mid

    return condition3 and condition4


def is_bearish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    """
    识别看跌吞没形态

    特征（两根K线组合）：
    - 第一根：阳线
    - 第二根：大阴线，完全吞没第一根K线的实体

    Args:
        df: 股票数据DataFrame
        idx: 第二根K线的索引位置

    Returns:
        True: 是看跌吞没，False: 不是
    """
    # 需要至少2根K线
    if idx < 1:
        return False

    # 两根K线
    first = df.iloc[idx - 1]
    second = df.iloc[idx]

    # 第一根：阳线
    first_is_bullish = first['close'] > first['open']

    # 第二根：阴线
    second_is_bearish = second['close'] < second['open']

    # 条件1：第一根是阳线，第二根是阴线
    if not (first_is_bullish and second_is_bearish):
        return False

    # 条件2：第二根完全吞没第一根的实体
    second_open_higher = second['open'] > first['close']
    second_close_lower = second['close'] < first['open']

    return second_open_higher and second_close_lower


def detect_top_patterns(df: pd.DataFrame, idx: int) -> List[str]:
    """
    检测指定位置的所有顶部形态

    Args:
        df: 股票数据DataFrame
        idx: 要检测的K线索引位置

    Returns:
        检测到的顶部形态名称列表
    """
    patterns = []

    if idx >= len(df):
        return patterns

    candle = df.iloc[idx]

    # 单K线形态
    if is_shooting_star(candle):
        patterns.append("射击之星")

    if is_gravestone_doji(candle):
        patterns.append("墓碑线")

    # 组合形态
    if is_evening_star(df, idx):
        patterns.append("黄昏之星")

    if is_dark_cloud_cover(df, idx):
        patterns.append("乌云盖顶")

    if is_bearish_engulfing(df, idx):
        patterns.append("看跌吞没")

    return patterns


def has_any_top_pattern(df: pd.DataFrame, idx: int) -> bool:
    """
    检查是否存在任何顶部形态

    Args:
        df: 股票数据DataFrame
        idx: 要检测的K线索引位置

    Returns:
        True: 存在顶部形态，False: 不存在
    """
    patterns = detect_top_patterns(df, idx)
    return len(patterns) > 0
