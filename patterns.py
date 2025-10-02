"""
K线形态识别库
实现各种经典K线形态的量化识别算法
采用装饰器模式，方便扩展新的形态识别器
"""

import pandas as pd
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """形态识别结果"""
    date: str           # 形态出现的日期
    index: int          # 在DataFrame中的索引
    pattern_name: str   # 形态名称
    meaning: str        # 技术含义
    price: float        # 标注位置的价格
    category: str       # 形态类别
    signal_type: str    # 信号类型（看涨/看跌）
    strength: str       # 信号强度
    color: str          # 显示颜色


class PatternRecognizer:
    """形态识别器基类"""

    def __init__(self):
        self.recognizers: List[Dict] = []

    def register(self, name: str, meaning: str, category: str = "反转",
                 signal_type: str = "看涨", strength: str = "中等"):
        """
        装饰器：注册形态识别器

        Args:
            name: 形态名称
            meaning: 技术含义
            category: 形态类别（底部反转/持续上涨/顶部反转/持续下跌/支撑）
            signal_type: 信号类型（看涨/看跌/中性）
            strength: 信号强度（强/中等/弱）
        """
        def decorator(func: Callable):
            # 根据类别和信号类型确定颜色
            color = self._get_color(category, signal_type, strength)

            self.recognizers.append({
                "name": name,
                "meaning": meaning,
                "category": category,
                "signal_type": signal_type,
                "strength": strength,
                "color": color,
                "function": func
            })
            return func
        return decorator

    def _get_color(self, category: str, signal_type: str, strength: str) -> str:
        """根据类别和信号类型确定颜色"""
        if "底部反转" in category or (signal_type == "看涨" and strength == "强"):
            return "#26a69a"  # 深绿色 - 强烈看涨
        elif "持续上涨" in category or signal_type == "看涨":
            return "#66bb6a"  # 浅绿色 - 持续看涨
        elif "顶部反转" in category or (signal_type == "看跌" and strength == "强"):
            return "#ef5350"  # 红色 - 强烈看跌
        elif "持续下跌" in category or signal_type == "看跌":
            return "#e57373"  # 浅红色 - 持续看跌
        elif "支撑" in category:
            return "#42a5f5"  # 蓝色 - 支撑
        else:
            return "#ffa726"  # 橙色 - 其他

    def recognize_all(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        识别所有注册的形态

        Args:
            df: 股票数据DataFrame

        Returns:
            识别结果列表
        """
        all_results = []

        for recognizer in self.recognizers:
            try:
                indices = recognizer["function"](df)
                for idx in indices:
                    result = PatternResult(
                        date=df.iloc[idx]['date'].strftime('%Y-%m-%d'),
                        index=idx,
                        pattern_name=recognizer["name"],
                        meaning=recognizer["meaning"],
                        price=df.iloc[idx]['high'],  # 标注在最高价位置
                        category=recognizer["category"],
                        signal_type=recognizer["signal_type"],
                        strength=recognizer["strength"],
                        color=recognizer["color"]
                    )
                    all_results.append(result)
                    logger.debug(f"识别到 {recognizer['name']} 于 {result.date}")
            except Exception as e:
                logger.error(f"识别 {recognizer['name']} 时出错: {e}")

        return sorted(all_results, key=lambda x: x.index)

    def get_recognizer_list(self) -> List[Dict]:
        """获取所有注册的识别器信息"""
        return [
            {
                "name": r["name"],
                "meaning": r["meaning"],
                "category": r["category"],
                "signal_type": r["signal_type"],
                "strength": r["strength"],
                "color": r["color"]
            }
            for r in self.recognizers
        ]


# 创建全局识别器实例
recognizer = PatternRecognizer()


# ==================== 辅助函数 ====================

def get_change_pct(open_price: float, close_price: float) -> float:
    """
    计算涨跌幅

    Args:
        open_price: 开盘价
        close_price: 收盘价

    Returns:
        涨跌幅（百分比）
    """
    if open_price == 0:
        return 0
    return (close_price - open_price) / open_price * 100


def is_big_bullish(open_price: float, close_price: float) -> bool:
    """判断是否为大阳线（涨幅 > 6%）"""
    return get_change_pct(open_price, close_price) > 6


def is_mid_bullish(open_price: float, close_price: float) -> bool:
    """判断是否为中阳线（涨幅在 3% 和 6% 之间）"""
    pct = get_change_pct(open_price, close_price)
    return 3 < pct <= 6


def is_small_bullish(open_price: float, close_price: float) -> bool:
    """判断是否为小阳线（涨幅在 1% 和 3% 之间）"""
    pct = get_change_pct(open_price, close_price)
    return 1 < pct <= 3


def is_big_mid_bullish(open_price: float, close_price: float) -> bool:
    """判断是否为大/中阳线（涨幅 > 3%）"""
    return get_change_pct(open_price, close_price) > 3


def is_big_bearish(open_price: float, close_price: float) -> bool:
    """判断是否为大阴线（跌幅 > 6%）"""
    return get_change_pct(open_price, close_price) < -6


def is_mid_bearish(open_price: float, close_price: float) -> bool:
    """判断是否为中阴线（跌幅在 3% 和 6% 之间）"""
    pct = get_change_pct(open_price, close_price)
    return -6 <= pct < -3


def is_small_bearish(open_price: float, close_price: float) -> bool:
    """判断是否为小阴线（跌幅在 1% 和 3% 之间）"""
    pct = get_change_pct(open_price, close_price)
    return -3 <= pct < -1


def is_big_mid_bearish(open_price: float, close_price: float) -> bool:
    """判断是否为大/中阴线（跌幅 > 3%）"""
    return get_change_pct(open_price, close_price) < -3


# ==================== 形态识别算法实现 ====================

@recognizer.register(name="早晨之星", meaning="看涨反转信号", category="底部反转",
                     signal_type="看涨", strength="强")
def recognize_morning_star(df: pd.DataFrame) -> List[int]:
    """
    早晨之星形态识别

    形态特征：
    1. 第一根是阴线
    2. 第二根是星线（实体小）且向下跳空
    3. 第三根是阳线且深入第一根实体

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(2, len(df)):
        # 获取三天数据
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2, h2, l2 = day2['close'], day2['open'], day2['high'], day2['low']
        c3, o3 = day3['close'], day3['open']

        body2 = abs(o2 - c2)
        range2 = h2 - l2

        # 条件1: 第一根是阴线
        cond1 = c1 < o1

        # 条件2: 第二根是星线且向下跳空
        cond2_star = (range2 > 0) and (body2 <= 0.2 * range2)
        cond2_gap = max(o2, c2) < c1
        cond2 = cond2_star and cond2_gap

        # 条件3: 第三根是阳线且深入第一根实体
        cond3_bullish = c3 > o3
        cond3_penetrate = c3 > c1 + 0.5 * (o1 - c1)
        cond3 = cond3_bullish and cond3_penetrate

        if cond1 and cond2 and cond3:
            results.append(i)

    return results


@recognizer.register(name="早晨十字星", meaning="强烈看涨反转信号", category="底部反转",
                     signal_type="看涨", strength="强")
def recognize_morning_doji_star(df: pd.DataFrame) -> List[int]:
    """
    早晨十字星形态识别

    与早晨之星类似，但第二根必须是十字星（实体更小）

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(2, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2, h2, l2 = day2['close'], day2['open'], day2['high'], day2['low']
        c3, o3 = day3['close'], day3['open']

        body2 = abs(o2 - c2)
        range2 = h2 - l2

        # 条件1: 第一根是阴线
        cond1 = c1 < o1

        # 条件2: 第二根是十字星（实体≤5%总长度）且向下跳空
        cond2_doji = (range2 > 0) and (body2 <= 0.05 * range2)
        cond2_gap = max(o2, c2) < c1
        cond2 = cond2_doji and cond2_gap

        # 条件3: 第三根是阳线且深入第一根实体
        cond3_bullish = c3 > o3
        cond3_penetrate = c3 > c1 + 0.5 * (o1 - c1)
        cond3 = cond3_bullish and cond3_penetrate

        if cond1 and cond2 and cond3:
            results.append(i)

    return results


@recognizer.register(name="好友反攻", meaning="看涨信号", category="底部反转",
                     signal_type="看涨", strength="中等")
def recognize_friendly_counterattack(df: pd.DataFrame) -> List[int]:
    """
    好友反攻形态识别

    形态特征：
    1. 第一根是中/大阴线
    2. 第二根向下跳空低开，但收盘价与第一根收盘价相近

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件0: 下跌趋势
        cond0 = is_downtrend

        # 条件1: 第一根是大/中阴线
        cond1 = is_big_mid_bearish(o1, c1)

        # 条件2: 第二根是阳线且跳空低开
        cond2_bullish = c2 > o2
        cond2_gap = o2 < c1
        cond2 = cond2_bullish and cond2_gap

        # 条件3: 收盘价追平（误差≤0.1%）
        if c1 > 0:
            cond3 = abs(c2 - c1) / c1 <= 0.001
        else:
            cond3 = False

        if cond0 and cond1 and cond2 and cond3:
            results.append(i)

    return results


@recognizer.register(name="锤子线", meaning="看涨反转信号", category="底部反转",
                     signal_type="看涨", strength="中等")
def recognize_hammer(df: pd.DataFrame) -> List[int]:
    """
    锤子线形态识别

    形态特征：
    1. 出现在下跌趋势中
    2. 实体小，下影线长（至少是实体的2倍）
    3. 上影线很短或没有

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(5, len(df)):  # 需要至少5天判断趋势
        day = df.iloc[i]

        o, h, l, c = day['open'], day['high'], day['low'], day['close']
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        # 判断是否下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 下影线长（至少是实体的2倍）
        cond2 = (body > 0) and (lower_shadow >= 2 * body)

        # 条件3: 上影线短（小于实体）
        cond3 = upper_shadow <= body

        # 条件4: 实体在K线上半部分
        cond4 = (total_range > 0) and (lower_shadow >= 0.6 * total_range)

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="倒锤子", meaning="看涨反转信号", category="底部反转",
                     signal_type="看涨", strength="中等")
def recognize_inverted_hammer(df: pd.DataFrame) -> List[int]:
    """
    倒锤子形态识别

    形态特征：
    1. 出现在下跌趋势中
    2. 实体小，上影线长（至少是实体的2倍）
    3. 下影线很短或没有

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(5, len(df)):
        day = df.iloc[i]

        o, h, l, c = day['open'], day['high'], day['low'], day['close']
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        # 判断是否下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 上影线长（至少是实体的2倍）
        cond2 = (body > 0) and (upper_shadow >= 2 * body)

        # 条件3: 下影线短（小于实体）
        cond3 = lower_shadow <= body

        # 条件4: 实体在K线下半部分
        cond4 = (total_range > 0) and (upper_shadow >= 0.6 * total_range)

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="曙光初现", meaning="看涨反转信号", category="底部反转",
                     signal_type="看涨", strength="强")
def recognize_piercing_pattern(df: pd.DataFrame) -> List[int]:
    """
    曙光初现形态识别

    形态特征：
    1. 处于下跌趋势中
    2. 第一根是大/中阴线
    3. 第二根低开，但收盘价深入第一根实体50%以上

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 第一根是大/中阴线
        cond2 = is_big_mid_bearish(o1, c1)

        # 条件3: 第二根低开
        cond3 = o2 < c1

        # 条件4: 第二根是阳线且深入第一根实体50%以上
        cond4_yang = c2 > o2
        penetration = c2 > c1 + 0.5 * (o1 - c1) and c2 < o1
        cond4 = cond4_yang and penetration

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="旭日东升", meaning="强烈看涨反转信号", category="底部反转",
                     signal_type="看涨", strength="强")
def recognize_rising_sun(df: pd.DataFrame) -> List[int]:
    """
    旭日东升形态识别（吞没形态）

    形态特征：
    1. 处于下跌趋势中
    2. 第一根是大/中阴线
    3. 第二根阳线完全吞没第一根阴线

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 第一根是大/中阴线
        cond2 = is_big_mid_bearish(o1, c1)

        # 条件3: 第二根是阳线
        cond3 = c2 > o2

        # 条件4: 完全吞没（阳线开盘价低于阴线收盘价，收盘价高于阴线开盘价）
        cond4 = o2 < c1 and c2 > o1

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="红三兵", meaning="持续上涨信号", category="持续上涨",
                     signal_type="看涨", strength="强")
def recognize_three_white_soldiers(df: pd.DataFrame) -> List[int]:
    """
    红三兵形态识别

    形态特征：
    1. 连续三根阳线
    2. 每根开盘在前一根实体内
    3. 收盘价依次创新高

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(2, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']
        c3, o3 = day3['close'], day3['open']

        # 条件1: 三根都是阳线
        cond1 = (c1 > o1) and (c2 > o2) and (c3 > o3)

        # 条件2: 开盘在前一根实体内
        cond2 = (o2 >= c1 * 0.95 and o2 <= o1) and (o3 >= c2 * 0.95 and o3 <= o2)

        # 条件3: 收盘价依次创新高
        cond3 = c3 > c2 > c1

        # 条件4: 实体相对较大
        avg_body = df['body'].rolling(window=10, min_periods=1).mean().iloc[i]
        cond4 = (day1['body'] > avg_body * 0.5 and
                 day2['body'] > avg_body * 0.5 and
                 day3['body'] > avg_body * 0.5)

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="平底", meaning="支撑信号", category="支撑",
                     signal_type="中性", strength="中等")
def recognize_tweezers_bottom(df: pd.DataFrame) -> List[int]:
    """
    平底形态识别

    形态特征：
    1. 处于下跌趋势中
    2. 两根或多根K线的最低价在同一水平

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 最低价接近（误差≤0.05%）
        low_diff = abs(day2['low'] - day1['low']) / day1['low']
        cond2 = low_diff <= 0.0005

        if cond1 and cond2:
            results.append(i)

    return results


@recognizer.register(name="射击之星", meaning="看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="强")
def recognize_shooting_star(df: pd.DataFrame) -> List[int]:
    """
    射击之星形态识别（倒T字线在上涨趋势中）

    形态特征：
    1. 出现在上涨趋势中
    2. 上影线很长（至少是实体的2倍）
    3. 实体很小
    4. 下影线很短或没有

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(5, len(df)):
        day = df.iloc[i]

        o, h, l, c = day['open'], day['high'], day['low'], day['close']
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        # 判断是否上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 上影线长（至少是实体的2倍）
        cond2 = (body > 0) and (upper_shadow >= 2 * body)

        # 条件3: 下影线短（小于实体）
        cond3 = lower_shadow <= body

        # 条件4: 实体在K线下半部分
        cond4 = (total_range > 0) and (upper_shadow >= 0.6 * total_range)

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="黑三鸦", meaning="持续下跌信号", category="持续下跌",
                     signal_type="看跌", strength="强")
def recognize_three_black_crows(df: pd.DataFrame) -> List[int]:
    """
    黑三鸦形态识别

    形态特征：
    1. 连续三根阴线
    2. 每根开盘在前一根实体内
    3. 收盘价依次创新低

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(2, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']
        c3, o3 = day3['close'], day3['open']

        # 条件1: 三根都是阴线
        cond1 = (c1 < o1) and (c2 < o2) and (c3 < o3)

        # 条件2: 开盘在前一根实体内
        cond2 = (o2 <= o1 and o2 >= c1 * 0.95) and (o3 <= o2 and o3 >= c2 * 0.95)

        # 条件3: 收盘价依次创新低
        cond3 = c3 < c2 < c1

        # 条件4: 实体相对较大
        avg_body = df['body'].rolling(window=10, min_periods=1).mean().iloc[i]
        cond4 = (day1['body'] > avg_body * 0.5 and
                 day2['body'] > avg_body * 0.5 and
                 day3['body'] > avg_body * 0.5)

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="跳空上扬", meaning="持续上涨信号", category="持续上涨",
                     signal_type="看涨", strength="强")
def recognize_rising_window(df: pd.DataFrame) -> List[int]:
    """
    跳空上扬形态识别

    形态特征：
    1. 出现在上涨趋势中
    2. 向上跳空
    3. 跳空后继续上涨

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根是阳线
        cond2 = day1['close'] > day1['open']

        # 条件3: 向上跳空
        cond3 = day2['low'] > day1['high']

        # 条件4: 第二根是阳线
        cond4 = day2['close'] > day2['open']

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="黄昏之星", meaning="看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="强")
def recognize_evening_star(df: pd.DataFrame) -> List[int]:
    """
    黄昏之星形态识别（早晨之星的反向）

    形态特征：
    1. 第一根是阳线
    2. 第二根是星线（实体小）且向上跳空
    3. 第三根是阴线且深入第一根实体

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(7, len(df)):
        # 获取三天数据
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2, h2, l2 = day2['close'], day2['open'], day2['high'], day2['low']
        c3, o3 = day3['close'], day3['open']

        body2 = abs(o2 - c2)
        range2 = h2 - l2

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件0: 上涨趋势
        cond0 = is_uptrend

        # 条件1: 第一根是阳线
        cond1 = c1 > o1

        # 条件2: 第二根是星线且向上跳空
        cond2_star = (range2 > 0) and (body2 <= 0.2 * range2)
        cond2_gap = min(o2, c2) > c1
        cond2 = cond2_star and cond2_gap

        # 条件3: 第三根是阴线且深入第一根实体
        cond3_bearish = c3 < o3
        cond3_penetrate = c3 < c1 - 0.5 * (c1 - o1)
        cond3 = cond3_bearish and cond3_penetrate

        if cond0 and cond1 and cond2 and cond3:
            results.append(i)

    return results


@recognizer.register(name="黄昏十字星", meaning="强烈看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="强")
def recognize_evening_doji_star(df: pd.DataFrame) -> List[int]:
    """
    黄昏十字星形态识别

    与黄昏之星类似，但第二根必须是十字星（实体更小）

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(7, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2, h2, l2 = day2['close'], day2['open'], day2['high'], day2['low']
        c3, o3 = day3['close'], day3['open']

        body2 = abs(o2 - c2)
        range2 = h2 - l2

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['open'] > df.iloc[i-3]['open']

        # 条件0: 上涨趋势
        cond0 = is_uptrend

        # 条件1: 第一根是阳线
        cond1 = c1 > o1

        # 条件2: 第二根是十字星（实体≤5%总长度）且向上跳空
        cond2_doji = (range2 > 0) and (body2 <= 0.05 * range2)
        cond2_gap = min(o2, c2) > c1
        cond2 = cond2_doji and cond2_gap

        # 条件3: 第三根是阴线且深入第一根实体
        cond3_bearish = c3 < o3
        cond3_penetrate = c3 < c1 - 0.5 * (c1 - o1)
        cond3 = cond3_bearish and cond3_penetrate

        if cond0 and cond1 and cond2 and cond3:
            results.append(i)

    return results


@recognizer.register(name="乌云盖顶", meaning="看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="强")
def recognize_dark_cloud_cover(df: pd.DataFrame) -> List[int]:
    """
    乌云盖顶形态识别

    形态特征：
    1. 处于上涨趋势中
    2. 第一根是大/中阳线
    3. 第二根跳高开盘
    4. 第二根是阴线且收盘价深入第一根实体50%以上

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        c1, o1, h1 = day1['close'], day1['open'], day1['high']
        c2, o2 = day2['close'], day2['open']

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根是大/中阳线
        cond2 = is_big_mid_bullish(o1, c1)

        # 条件3: 第二根跳高开盘
        cond3 = o2 > h1

        # 条件4: 第二根是阴线且深入第一根实体50%以上
        cond4_bearish = c2 < o2
        penetration = c2 < (o1 + c1) / 2 and c2 > o1
        cond4 = cond4_bearish and penetration

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="倾盆大雨", meaning="强烈看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="强")
def recognize_pouring_rain(df: pd.DataFrame) -> List[int]:
    """
    倾盆大雨形态识别

    形态特征：
    1. 处于上涨趋势中
    2. 第一根是大/中阳线
    3. 第二根低开
    4. 第二根是阴线且收盘价低于第一根开盘价

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根是大/中阳线
        cond2 = is_big_mid_bullish(o1, c1)

        # 条件3: 第二根低开
        cond3 = o2 < o1

        # 条件4: 第二根是阴线且收盘价低于第一根开盘价
        cond4_bearish = c2 < o2
        cond4_low = c2 < o1
        cond4 = cond4_bearish and cond4_low

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="吊颈线", meaning="看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="中等")
def recognize_hanging_man(df: pd.DataFrame) -> List[int]:
    """
    吊颈线形态识别（锤子线在上涨趋势中）

    形态特征：
    1. 出现在上涨趋势中
    2. 实体小，下影线长（至少是实体的2倍）
    3. 上影线很短或没有

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(5, len(df)):
        day = df.iloc[i]

        o, h, l, c = day['open'], day['high'], day['low'], day['close']
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        # 判断是否上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 下影线长（至少是实体的2倍）
        cond2 = (body > 0) and (lower_shadow >= 2 * body)

        # 条件3: 上影线短（小于实体）
        cond3 = upper_shadow <= body

        # 条件4: 实体在K线上半部分
        cond4 = (total_range > 0) and (lower_shadow >= 0.6 * total_range)

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="平顶", meaning="看跌反转信号", category="顶部反转",
                     signal_type="看跌", strength="中等")
def recognize_tweezers_top(df: pd.DataFrame) -> List[int]:
    """
    平顶形态识别

    形态特征：
    1. 处于上涨趋势中
    2. 两根或多根K线的最高价在同一水平

    Returns:
        形态确认日的索引列表
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 最高价接近（误差≤0.05%）
        high_diff = abs(day2['high'] - day1['high']) / day1['high']
        cond2 = high_diff <= 0.0005

        if cond1 and cond2:
            results.append(i)

    return results


@recognizer.register(name="连续跳空三阴线", meaning="强烈见底信号", category="底部反转",
                     signal_type="看涨", strength="强")
def recognize_three_gap_down(df: pd.DataFrame) -> List[int]:
    """
    连续跳空三阴线形态识别

    形态特征：
    1. 处于下跌趋势中
    2. 连续三根阴线
    3. 每根都向下跳空

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(7, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 三根都是阴线
        cond2 = (day1['close'] < day1['open']) and \
                (day2['close'] < day2['open']) and \
                (day3['close'] < day3['open'])

        # 条件3: 每根都向下跳空
        cond3 = (day2['high'] < day1['low']) and \
                (day3['high'] < day2['low'])

        if cond1 and cond2 and cond3:
            results.append(i)

    return results


@recognizer.register(name="低档五阳线", meaning="见底信号", category="底部反转",
                     signal_type="看涨", strength="中等")
def recognize_five_bulls_bottom(df: pd.DataFrame) -> List[int]:
    """
    低档五阳线形态识别

    形态特征：
    1. 处于下跌行情中
    2. 连续拉出5根或以上的小阳线（涨幅1%-3%）

    Returns:
        形态确认日的索引列表（第五根K线）
    """
    results = []

    for i in range(9, len(df)):
        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-7]['close']

        if not is_downtrend:
            continue

        # 检查连续5根是否都是小阳线
        is_five_bulls = True
        for j in range(5):
            day = df.iloc[i-j]
            # 必须是小阳线（涨幅1%-3%）
            if not is_small_bullish(day['open'], day['close']):
                is_five_bulls = False
                break

        if is_five_bulls:
            results.append(i)

    return results


@recognizer.register(name="两红夹一黑", meaning="反转信号", category="底部反转",
                     signal_type="看涨", strength="中等")
def recognize_two_bulls_sandwich_bear_bottom(df: pd.DataFrame) -> List[int]:
    """
    两红夹一黑形态识别（跌势中）

    形态特征：
    1. 处于跌势中
    2. 第一根阳线
    3. 第二根阴线（实体较短）
    4. 第三根阳线
    5. 中间阴线被两边阳线包裹

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(7, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 第一根和第三根是阳线
        cond2 = (day1['close'] > day1['open']) and (day3['close'] > day3['open'])

        # 条件3: 第二根是阴线
        cond3 = day2['close'] < day2['open']

        # 条件4: 中间阴线实体较短，被两边阳线包裹
        cond4 = (day2['body'] < day1['body'] * 0.5) and \
                (day2['body'] < day3['body'] * 0.5) and \
                (day2['open'] < max(day1['close'], day3['close'])) and \
                (day2['close'] > min(day1['open'], day3['open']))

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="高档五阴线", meaning="见顶信号", category="顶部反转",
                     signal_type="看跌", strength="中等")
def recognize_five_bears_top(df: pd.DataFrame) -> List[int]:
    """
    高档五阴线形态识别

    形态特征：
    1. 处于涨势中
    2. 在有力度的阳线后
    3. 连续出现5根或以上的小阴线（跌幅1%-3%）

    Returns:
        形态确认日的索引列表（第五根K线）
    """
    results = []

    for i in range(9, len(df)):
        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-7]['close']

        if not is_uptrend:
            continue

        # 检查连续5根是否都是小阴线
        is_five_bears = True
        for j in range(5):
            day = df.iloc[i-j]
            # 必须是小阴线（跌幅1%-3%）
            if not is_small_bearish(day['open'], day['close']):
                is_five_bears = False
                break

        if is_five_bears:
            results.append(i)

    return results


@recognizer.register(name="两黑夹一红", meaning="见顶信号", category="顶部反转",
                     signal_type="看跌", strength="中等")
def recognize_two_bears_sandwich_bull_top(df: pd.DataFrame) -> List[int]:
    """
    两黑夹一红形态识别（涨势中）

    形态特征：
    1. 处于涨势中
    2. 第一根阴线
    3. 第二根阳线（实体较短）
    4. 第三根阴线
    5. 中间阳线被两边阴线包裹

    Returns:
        形态确认日的索引列表（第三根K线）
    """
    results = []

    for i in range(7, len(df)):
        day1 = df.iloc[i-2]
        day2 = df.iloc[i-1]
        day3 = df.iloc[i]

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根和第三根是阴线
        cond2 = (day1['close'] < day1['open']) and (day3['close'] < day3['open'])

        # 条件3: 第二根是阳线
        cond3 = day2['close'] > day2['open']

        # 条件4: 中间阳线实体较短，被两边阴线包裹
        cond4 = (day2['body'] < day1['body'] * 0.5) and \
                (day2['body'] < day3['body'] * 0.5) and \
                (day2['close'] < max(day1['open'], day3['open'])) and \
                (day2['open'] > min(day1['close'], day3['close']))

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="穿头破脚(看涨)", meaning="买进信号", category="底部反转",
                     signal_type="看涨", strength="强")
def recognize_engulfing_bullish(df: pd.DataFrame) -> List[int]:
    """
    穿头破脚形态识别（跌势中）

    形态特征：
    1. 处于下跌趋势中
    2. 第一根是阴线
    3. 第二根是阳线
    4. 第二根阳线实体完全包容第一根阴线实体

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 第一根是阴线
        cond2 = day1['close'] < day1['open']

        # 条件3: 第二根是阳线
        cond3 = day2['close'] > day2['open']

        # 条件4: 阳线实体完全包容阴线实体
        cond4 = (day2['open'] < day1['close']) and (day2['close'] > day1['open'])

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="穿头破脚(看跌)", meaning="卖出信号", category="顶部反转",
                     signal_type="看跌", strength="强")
def recognize_engulfing_bearish(df: pd.DataFrame) -> List[int]:
    """
    穿头破脚形态识别（涨势中）

    形态特征：
    1. 处于上涨趋势中
    2. 第一根是阳线
    3. 第二根是阴线
    4. 第二根阴线实体完全包容第一根阳线实体

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根是阳线
        cond2 = day1['close'] > day1['open']

        # 条件3: 第二根是阴线
        cond3 = day2['close'] < day2['open']

        # 条件4: 阴线实体完全包容阳线实体
        cond4 = (day2['close'] < day1['open']) and (day2['open'] > day1['close'])

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="身怀六甲(看涨)", meaning="买进信号", category="底部反转",
                     signal_type="看涨", strength="中等")
def recognize_harami_bullish(df: pd.DataFrame) -> List[int]:
    """
    身怀六甲形态识别（跌势中）

    形态特征：
    1. 处于下跌趋势中
    2. 第一根为大/中阴线，能完全包容第二根
    3. 第二根K线为小阴/阳线或十字星

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        o1, c1 = day1['open'], day1['close']
        o2, c2 = day2['open'], day2['close']

        # 判断下跌趋势
        is_downtrend = df.iloc[i]['close'] < df.iloc[i-5]['close']

        # 条件1: 下跌趋势
        cond1 = is_downtrend

        # 条件2: 第一根是大/中阴线
        cond2 = is_big_mid_bearish(o1, c1)

        # 条件3: 第一根实体包容第二根（第二根开盘和收盘都在第一根实体内）
        cond3 = (min(o2, c2) >= min(o1, c1)) and \
                (max(o2, c2) <= max(o1, c1))

        # 条件4: 第二根实体较小
        cond4 = day2['body'] < day1['body'] * 0.5

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="身怀六甲(看跌)", meaning="卖出信号", category="顶部反转",
                     signal_type="看跌", strength="中等")
def recognize_harami_bearish(df: pd.DataFrame) -> List[int]:
    """
    身怀六甲形态识别（涨势中）

    形态特征：
    1. 处于上涨趋势中
    2. 第一根为大/中阳线，能完全包容第二根
    3. 第二根K线为小阴/阳线或十字星

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        o1, c1 = day1['open'], day1['close']
        o2, c2 = day2['open'], day2['close']

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根是大/中阳线
        cond2 = is_big_mid_bullish(o1, c1)

        # 条件3: 第一根实体包容第二根（第二根开盘和收盘都在第一根实体内）
        cond3 = (min(o2, c2) >= min(o1, c1)) and \
                (max(o2, c2) <= max(o1, c1))

        # 条件4: 第二根实体较小
        cond4 = day2['body'] < day1['body'] * 0.5

        if cond1 and cond2 and cond3 and cond4:
            results.append(i)

    return results


@recognizer.register(name="淡友反攻", meaning="见顶信号", category="顶部反转",
                     signal_type="看跌", strength="中等")
def recognize_bearish_counterattack(df: pd.DataFrame) -> List[int]:
    """
    淡友反攻形态识别（好友反攻的反向）

    形态特征：
    1. 处于上涨趋势中
    2. 第一根是大阳线
    3. 第二根跳高开盘
    4. 第二根是中/大阴线，收盘价与第一根收盘价相近

    Returns:
        形态确认日的索引列表（第二根K线）
    """
    results = []

    for i in range(6, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        c1, o1 = day1['close'], day1['open']
        c2, o2 = day2['close'], day2['open']

        # 判断上涨趋势
        is_uptrend = df.iloc[i]['close'] > df.iloc[i-5]['close']

        # 条件1: 上涨趋势
        cond1 = is_uptrend

        # 条件2: 第一根是大阳线
        cond2 = is_big_bullish(o1, c1)

        # 条件3: 第二根跳高开盘
        cond3 = o2 > c1

        # 条件4: 第二根是中/大阴线
        cond4 = is_big_mid_bearish(o2, c2)

        # 条件5: 收盘价追平（误差≤0.1%）
        if c1 > 0:
            cond5 = abs(c2 - c1) / c1 <= 0.001
        else:
            cond5 = False

        if cond1 and cond2 and cond3 and cond4 and cond5:
            results.append(i)

    return results


# ==================== 工具函数 ====================

def get_all_patterns() -> List[Dict]:
    """获取所有已注册的形态信息"""
    return recognizer.get_recognizer_list()


def recognize_patterns(df: pd.DataFrame, pattern_names: List[str] = None,
                      categories: List[str] = None) -> List[PatternResult]:
    """
    识别指定的形态

    Args:
        df: 股票数据DataFrame
        pattern_names: 要识别的形态名称列表，None表示识别所有
        categories: 要识别的形态类别列表，None表示所有类别

    Returns:
        识别结果列表
    """
    if pattern_names is None and categories is None:
        return recognizer.recognize_all(df)

    # 过滤指定的识别器
    all_results = []
    for rec in recognizer.recognizers:
        # 检查是否匹配名称或类别
        name_match = pattern_names is None or rec["name"] in pattern_names
        category_match = categories is None or rec["category"] in categories

        if name_match and category_match:
            try:
                indices = rec["function"](df)
                for idx in indices:
                    result = PatternResult(
                        date=df.iloc[idx]['date'].strftime('%Y-%m-%d'),
                        index=idx,
                        pattern_name=rec["name"],
                        meaning=rec["meaning"],
                        price=df.iloc[idx]['high'],
                        category=rec["category"],
                        signal_type=rec["signal_type"],
                        strength=rec["strength"],
                        color=rec["color"]
                    )
                    all_results.append(result)
            except Exception as e:
                logger.error(f"识别 {rec['name']} 时出错: {e}")

    return sorted(all_results, key=lambda x: x.index)


if __name__ == "__main__":
    # 测试代码
    print("已注册的形态识别器:")
    for pattern in get_all_patterns():
        print(f"  - {pattern['name']}: {pattern['meaning']} ({pattern['category']})")
