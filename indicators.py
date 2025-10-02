"""
技术指标计算模块
提供常用技术指标的计算函数，用于策略过滤
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_drawdown(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    计算从近期高点的回撤幅度

    Args:
        df: 股票数据DataFrame，需包含'high'列
        window: 回看窗口期（交易日）

    Returns:
        回撤幅度序列（0-1之间，例如0.15表示回撤15%）
    """
    # 计算窗口期内的最高价
    rolling_max = df['high'].rolling(window=window, min_periods=1).max()

    # 计算当前收盘价相对最高价的回撤
    current_close = df['close']
    drawdown = (rolling_max - current_close) / rolling_max

    # 处理除零情况
    drawdown = drawdown.fillna(0)

    return drawdown


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算RSI（相对强弱指标）

    Args:
        df: 股票数据DataFrame，需包含'close'列
        period: RSI计算周期

    Returns:
        RSI值序列（0-100之间）
    """
    # 计算价格变动
    delta = df['close'].diff()

    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 计算平均涨幅和跌幅（使用指数移动平均）
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # 计算RS和RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # 处理除零和NaN情况
    rsi = rsi.fillna(50)  # 无法计算时默认为中性值50

    return rsi


def calculate_ma(df: pd.DataFrame, period: int) -> pd.Series:
    """
    计算简单移动平均线（SMA）

    Args:
        df: 股票数据DataFrame，需包含'close'列
        period: 均线周期

    Returns:
        移动平均线序列
    """
    return df['close'].rolling(window=period, min_periods=1).mean()


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    计算成交量移动平均线

    Args:
        df: 股票数据DataFrame，需包含'volume'列
        period: 均线周期

    Returns:
        成交量移动平均线序列
    """
    return df['volume'].rolling(window=period, min_periods=1).mean()


def is_reversal_candle(candle: pd.Series) -> bool:
    """
    判断是否为反转K线（阳线或锤子线）

    Args:
        candle: 单根K线数据，需包含open, close, high, low字段

    Returns:
        True: 是反转K线，False: 不是
    """
    open_price = candle['open']
    close_price = candle['close']
    high_price = candle['high']
    low_price = candle['low']

    # 条件1: 阳线（收盘价 > 开盘价）
    is_bullish = close_price > open_price

    # 条件2: 锤子线（下影线长度 > 实体的2倍）
    body = abs(close_price - open_price)
    lower_shadow = min(close_price, open_price) - low_price
    is_hammer = lower_shadow > 2 * body if body > 0 else False

    return is_bullish or is_hammer


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算平均真实波幅（ATR）

    Args:
        df: 股票数据DataFrame，需包含'high', 'low', 'close'列
        period: ATR计算周期

    Returns:
        ATR值序列
    """
    # 计算真实波幅的三个组成部分
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    # 真实波幅 = 三者中的最大值
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR = 真实波幅的移动平均
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr


def check_ma_support(df: pd.DataFrame, candle_idx: int,
                     ma_period: int = 60, tolerance: float = 0.03) -> bool:
    """
    检查K线是否接近均线支撑位

    Args:
        df: 股票数据DataFrame
        candle_idx: K线索引位置
        ma_period: 均线周期
        tolerance: 容差范围（例如0.03表示±3%）

    Returns:
        True: 接近均线支撑，False: 不接近
    """
    if candle_idx < ma_period:
        return False

    # 计算均线
    ma = calculate_ma(df[:candle_idx + 1], ma_period)
    ma_value = ma.iloc[-1]

    # 当前收盘价
    close_price = df.iloc[candle_idx]['close']

    # 检查是否在容差范围内
    lower_bound = ma_value * (1 - tolerance)
    upper_bound = ma_value * (1 + tolerance)

    return lower_bound <= close_price <= upper_bound


def check_volume_shrink(df: pd.DataFrame, candle_idx: int,
                        ma_period: int = 20) -> bool:
    """
    检查成交量是否缩量

    Args:
        df: 股票数据DataFrame
        candle_idx: K线索引位置
        ma_period: 成交量均线周期

    Returns:
        True: 缩量，False: 不缩量
    """
    if candle_idx < ma_period:
        return False

    # 计算成交量均线
    volume_ma = calculate_volume_ma(df[:candle_idx + 1], ma_period)
    avg_volume = volume_ma.iloc[-1]

    # 当前成交量
    current_volume = df.iloc[candle_idx]['volume']

    # 当前成交量小于平均量即为缩量
    return current_volume < avg_volume


def check_volume_expand(df: pd.DataFrame, candle_idx: int,
                        ma_period: int = 5, threshold: float = 1.5) -> bool:
    """
    检查成交量是否放量

    Args:
        df: 股票数据DataFrame
        candle_idx: K线索引位置
        ma_period: 成交量均线周期
        threshold: 放量倍数阈值（例如1.5表示超过均量的1.5倍）

    Returns:
        True: 放量，False: 不放量
    """
    if candle_idx < ma_period:
        return False

    # 计算成交量均线（使用前N天数据，不包括当天）
    volume_ma = calculate_volume_ma(df[:candle_idx], ma_period)
    avg_volume = volume_ma.iloc[-1]

    # 当前成交量
    current_volume = df.iloc[candle_idx]['volume']

    # 当前成交量超过平均量的threshold倍即为放量
    return current_volume > threshold * avg_volume


# ==================== 卖出信号相关指标 ====================

def check_rsi_overbought(df: pd.DataFrame, candle_idx: int,
                         threshold: float = 70, period: int = 14) -> bool:
    """
    检查RSI是否超买

    Args:
        df: 股票数据DataFrame
        candle_idx: K线索引位置
        threshold: RSI超买阈值（默认70）
        period: RSI计算周期

    Returns:
        True: 超买，False: 未超买
    """
    if candle_idx < period:
        return False

    # 计算RSI
    rsi = calculate_rsi(df[:candle_idx + 1], period)
    current_rsi = rsi.iloc[-1]

    # RSI超过阈值即为超买
    return current_rsi > threshold


def calculate_trailing_stop_price(df: pd.DataFrame, buy_price: float,
                                   buy_idx: int, current_idx: int,
                                   trail_pct: float = 0.10) -> float:
    """
    计算移动止损价格

    Args:
        df: 股票数据DataFrame
        buy_price: 买入价格
        buy_idx: 买入位置索引
        current_idx: 当前位置索引
        trail_pct: 回撤比例（默认10%，即从最高点回撤10%触发）

    Returns:
        移动止损价格
    """
    # 计算买入后的最高价
    if current_idx <= buy_idx:
        return buy_price * (1 - trail_pct)

    highest_price = df.loc[buy_idx:current_idx, 'high'].max()

    # 移动止损价格 = 最高价 * (1 - 回撤比例)
    trailing_stop = highest_price * (1 - trail_pct)

    # 移动止损价格不能低于买入价
    return max(trailing_stop, buy_price * (1 - trail_pct))


def check_peak_reached(df: pd.DataFrame, candle_idx: int,
                      lookback: int = 5) -> bool:
    """
    检查是否达到阶段性高点（用于判断是否应该卖出）

    Args:
        df: 股票数据DataFrame
        candle_idx: 当前K线索引
        lookback: 回看周期

    Returns:
        True: 达到阶段性高点，False: 未达到
    """
    if candle_idx < lookback or candle_idx >= len(df) - 1:
        return False

    current_high = df.iloc[candle_idx]['high']

    # 检查前N根K线的最高价
    prev_highs = df.iloc[candle_idx - lookback:candle_idx]['high']

    # 检查后1根K线的最高价（需要有后续数据）
    if candle_idx + 1 < len(df):
        next_high = df.iloc[candle_idx + 1]['high']

        # 当前高点高于前N根和后1根，说明是阶段性高点
        return current_high >= prev_highs.max() and current_high > next_high

    return False


def check_volume_divergence(df: pd.DataFrame, candle_idx: int,
                            lookback: int = 5) -> bool:
    """
    检查价量背离（价格新高但成交量萎缩）

    Args:
        df: 股票数据DataFrame
        candle_idx: 当前K线索引
        lookback: 回看周期

    Returns:
        True: 存在背离，False: 不存在背离
    """
    if candle_idx < lookback:
        return False

    # 获取最近N根K线
    recent_data = df.iloc[candle_idx - lookback:candle_idx + 1]

    # 当前价格是否创新高
    current_high = df.iloc[candle_idx]['high']
    is_new_high = current_high >= recent_data['high'].max()

    if not is_new_high:
        return False

    # 当前成交量是否低于平均水平
    current_volume = df.iloc[candle_idx]['volume']
    avg_volume = recent_data['volume'].mean()

    # 价格新高但成交量低于平均，存在背离
    return current_volume < avg_volume * 0.8  # 低于平均量的80%
