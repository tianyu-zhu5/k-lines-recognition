"""
回测引擎
实现K线形态信号的回测功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from data_loader import StockDataLoader
from patterns import recognize_patterns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    stock_code: str          # 股票代码
    buy_date: str            # 买入日期
    buy_price: float         # 买入价格
    shares: int              # 买入股数
    amount: float            # 买入金额
    sell_date: Optional[str] = None   # 卖出日期
    sell_price: Optional[float] = None  # 卖出价格
    sell_reason: Optional[str] = None   # 卖出原因
    profit: Optional[float] = None      # 盈亏
    profit_pct: Optional[float] = None  # 盈亏比例
    pattern_name: str = ""   # 触发的形态名称


@dataclass
class FilterConfig:
    """过滤器配置（仅对黑三鸦生效）"""
    # 累计跌幅过滤
    enable_drawdown_filter: bool = False      # 是否启用累计跌幅过滤
    min_drawdown_pct: float = 0.15            # 最小回撤比例（15%）
    drawdown_window: int = 20                 # 回看窗口期（交易日）

    # RSI超卖过滤
    enable_rsi_filter: bool = False           # 是否启用RSI过滤
    rsi_threshold: float = 30                 # RSI阈值（小于此值才买入）
    rsi_period: int = 14                      # RSI计算周期

    # 反转K线确认过滤
    enable_reversal_filter: bool = False      # 是否启用反转K线确认


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000      # 初始资金
    per_trade_amount: float = 10000      # 每次买入金额
    stop_loss_pct: float = 0.05          # 止损比例
    stop_profit_pct: float = 0.05        # 止盈比例
    hold_days: Optional[int] = 10        # 持仓天数（None表示不限制）
    start_date: Optional[str] = None     # 回测开始日期
    end_date: Optional[str] = None       # 回测结束日期
    pattern_names: List[str] = None      # 要回测的形态名称列表
    filter_config: FilterConfig = field(default_factory=FilterConfig)  # 过滤器配置


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig, data_loader: StockDataLoader):
        self.config = config
        self.data_loader = data_loader
        self.cash = config.initial_capital
        self.positions: List[Trade] = []  # 持仓
        self.closed_trades: List[Trade] = []  # 已平仓交易
        self.daily_values: List[Dict] = []  # 每日账户价值

    def load_stock_pool(self, pool_file: str) -> List[str]:
        """
        加载股票池

        Args:
            pool_file: 股票池文件路径

        Returns:
            股票代码列表
        """
        stocks = []
        try:
            with open(pool_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if line and not line.startswith('#'):
                        # 提取股票代码（格式：000001.SZ）
                        stocks.append(line)
            logger.info(f"从 {pool_file} 加载了 {len(stocks)} 只股票")
        except Exception as e:
            logger.error(f"加载股票池失败: {e}")
        return stocks

    def run_backtest(self, stock_codes: List[str]) -> Dict:
        """
        运行回测

        Args:
            stock_codes: 股票代码列表

        Returns:
            回测结果字典
        """
        logger.info(f"开始回测，股票数量: {len(stock_codes)}")

        # 重置状态
        self.cash = self.config.initial_capital
        self.positions = []
        self.closed_trades = []
        self.daily_values = []

        # 收集所有股票的信号
        all_signals = self._collect_signals(stock_codes)

        if not all_signals:
            logger.warning("未发现任何交易信号")
            return self._generate_report()

        # 按日期排序信号
        all_signals.sort(key=lambda x: x['date'])

        # 模拟交易
        self._simulate_trading(all_signals, stock_codes)

        # 生成报告
        return self._generate_report()

    def _apply_filters(self, df: pd.DataFrame, signal_idx: int, pattern_name: str) -> bool:
        """
        应用过滤器，判断信号是否有效

        Args:
            df: 股票数据DataFrame
            signal_idx: 信号位置索引
            pattern_name: 形态名称

        Returns:
            True: 通过过滤，False: 被过滤
        """
        # 只对黑三鸦应用过滤器
        if pattern_name != "黑三鸦":
            return True

        from indicators import (
            calculate_drawdown,
            calculate_rsi,
            is_reversal_candle
        )

        config = self.config.filter_config

        # 过滤器1: 累计跌幅过滤
        if config.enable_drawdown_filter:
            # 计算到信号位置为止的回撤
            drawdown = calculate_drawdown(df[:signal_idx + 1], config.drawdown_window)
            if drawdown.iloc[-1] < config.min_drawdown_pct:
                logger.debug(f"过滤：回撤不足 {drawdown.iloc[-1]*100:.1f}% < {config.min_drawdown_pct*100:.1f}%")
                return False

        # 过滤器2: RSI超卖过滤
        if config.enable_rsi_filter:
            rsi = calculate_rsi(df[:signal_idx + 1], config.rsi_period)
            if rsi.iloc[-1] >= config.rsi_threshold:
                logger.debug(f"过滤：RSI未超卖 {rsi.iloc[-1]:.1f} >= {config.rsi_threshold:.1f}")
                return False

        # 过滤器3: 反转K线确认过滤
        if config.enable_reversal_filter:
            # 检查黑三鸦后第一根K线（即买入前一天，signal_idx+1）
            # 因为黑三鸦买入延迟是2天，所以signal_idx+1就是买入前一天
            if signal_idx + 1 < len(df):
                next_candle = df.iloc[signal_idx + 1]
                if not is_reversal_candle(next_candle):
                    logger.debug(f"过滤：无反转K线确认")
                    return False

        return True  # 通过所有过滤器

    def _collect_signals(self, stock_codes: List[str]) -> List[Dict]:
        """收集所有股票的交易信号"""
        all_signals = []

        for stock_code in stock_codes:
            try:
                # 转换股票代码格式：000001.SZ -> 000001_SZ
                stock_code_converted = stock_code.replace('.', '_')

                # 加载股票数据
                df = self.data_loader.load_stock_data(stock_code_converted)
                if df is None or len(df) == 0:
                    continue

                # 过滤日期范围
                if self.config.start_date:
                    df = df[df['date'] >= pd.to_datetime(self.config.start_date)]
                if self.config.end_date:
                    df = df[df['date'] <= pd.to_datetime(self.config.end_date)]

                if len(df) == 0:
                    continue

                # 识别形态
                patterns = recognize_patterns(df, pattern_names=self.config.pattern_names)

                # 黑三鸦冷却期：记录黑三鸦信号日期及其后2个交易日，跳过这些日期的黑三鸦
                black_crow_cooldown_dates = set()
                for pattern in patterns:
                    if pattern.pattern_name == "黑三鸦":
                        signal_idx = pattern.index
                        # 记录信号日期的后两个交易日
                        if signal_idx + 1 < len(df):
                            black_crow_cooldown_dates.add(df.iloc[signal_idx + 1]['date'])
                        if signal_idx + 2 < len(df):
                            black_crow_cooldown_dates.add(df.iloc[signal_idx + 2]['date'])

                # 记录信号
                for pattern in patterns:
                    signal_date = pattern.date
                    signal_idx = pattern.index

                    # 如果是黑三鸦在冷却期内，跳过
                    if pattern.pattern_name == "黑三鸦" and pd.to_datetime(signal_date) in black_crow_cooldown_dates:
                        continue

                    # 【新增】应用过滤器
                    if not self._apply_filters(df, signal_idx, pattern.pattern_name):
                        continue  # 被过滤，跳过此信号

                    # 确定买入延迟天数：黑三鸦为2天，其他形态为1天
                    buy_delay = 2 if pattern.pattern_name == "黑三鸦" else 1

                    # 检查是否有足够的未来交易日
                    if signal_idx + buy_delay < len(df):
                        buy_date = df.iloc[signal_idx + buy_delay]['date']
                        buy_price = df.iloc[signal_idx + buy_delay]['open']

                        all_signals.append({
                            'stock_code': stock_code,
                            'signal_date': signal_date,
                            'buy_date': buy_date,
                            'buy_price': buy_price,
                            'pattern_name': pattern.pattern_name,
                            'date': buy_date  # 用于排序
                        })

            except Exception as e:
                logger.error(f"处理股票 {stock_code} 时出错: {e}")
                continue

        logger.info(f"收集到 {len(all_signals)} 个交易信号")
        return all_signals

    def _simulate_trading(self, signals: List[Dict], stock_codes: List[str]):
        """模拟交易过程"""

        # 获取每只股票的完整数据
        stock_data = {}
        for stock_code in stock_codes:
            # 转换股票代码格式：000001.SZ -> 000001_SZ
            stock_code_converted = stock_code.replace('.', '_')
            df = self.data_loader.load_stock_data(stock_code_converted)
            if df is not None:
                stock_data[stock_code] = df

        # 收集所有交易日期（从所有股票数据中获取）
        all_trading_dates = set()
        for stock_code, df in stock_data.items():
            # 过滤到回测时间范围内的数据
            df_filtered = df.copy()
            if self.config.start_date:
                df_filtered = df_filtered[df_filtered['date'] >= pd.to_datetime(self.config.start_date)]
            if self.config.end_date:
                df_filtered = df_filtered[df_filtered['date'] <= pd.to_datetime(self.config.end_date)]

            all_trading_dates.update(df_filtered['date'].tolist())

        # 排序所有交易日期
        all_trading_dates = sorted(all_trading_dates)

        # 逐日模拟（遍历所有交易日，而不仅仅是有买入信号的日期）
        for current_date in all_trading_dates:
            # 1. 处理卖出（先卖后买）
            self._process_sells(current_date, stock_data)

            # 2. 处理买入
            today_signals = [s for s in signals if s['buy_date'] == current_date]
            for signal in today_signals:
                self._process_buy(signal)

            # 3. 记录每日账户价值
            self._record_daily_value(current_date, stock_data)

        # 最后清仓所有持仓
        self._close_all_positions(stock_data)

    def _process_buy(self, signal: Dict):
        """处理买入"""
        stock_code = signal['stock_code']
        buy_date = signal['buy_date']
        buy_price = signal['buy_price']
        pattern_name = signal['pattern_name']

        # 计算买入股数（取整百股）
        shares = int(self.config.per_trade_amount / buy_price / 100) * 100
        if shares == 0:
            return

        # 计算实际花费
        actual_amount = shares * buy_price

        # 检查资金是否足够
        if actual_amount > self.cash:
            logger.debug(f"资金不足，无法买入 {stock_code}")
            return

        # 扣除现金
        self.cash -= actual_amount

        # 创建持仓记录
        trade = Trade(
            stock_code=stock_code,
            buy_date=str(buy_date.date()) if isinstance(buy_date, pd.Timestamp) else buy_date,
            buy_price=buy_price,
            shares=shares,
            amount=actual_amount,
            pattern_name=pattern_name
        )
        self.positions.append(trade)

        logger.debug(f"买入: {stock_code} {shares}股 @{buy_price:.2f} on {trade.buy_date}")

    def _process_sells(self, current_date, stock_data: Dict):
        """处理卖出"""
        positions_to_remove = []

        for i, position in enumerate(self.positions):
            stock_code = position.stock_code

            # 获取股票数据
            if stock_code not in stock_data:
                continue

            df = stock_data[stock_code]

            # 找到当前日期的数据
            current_row = df[df['date'] == current_date]
            if len(current_row) == 0:
                continue

            current_row = current_row.iloc[0]

            # 检查卖出条件
            sell_price = None
            sell_reason = None

            # 1. 检查跳空止损/止盈
            open_price = current_row['open']
            if open_price >= position.buy_price * (1 + self.config.stop_profit_pct):
                sell_price = open_price
                sell_reason = f"跳空止盈({self.config.stop_profit_pct*100:.1f}%)"
            elif open_price <= position.buy_price * (1 - self.config.stop_loss_pct):
                sell_price = open_price
                sell_reason = f"跳空止损({self.config.stop_loss_pct*100:.1f}%)"

            # 2. 检查盘中止损（优先）
            if sell_price is None:
                low_price = current_row['low']
                if low_price <= position.buy_price * (1 - self.config.stop_loss_pct):
                    sell_price = position.buy_price * (1 - self.config.stop_loss_pct)
                    sell_reason = f"止损({self.config.stop_loss_pct*100:.1f}%)"

            # 3. 检查盘中止盈
            if sell_price is None:
                high_price = current_row['high']
                if high_price >= position.buy_price * (1 + self.config.stop_profit_pct):
                    sell_price = position.buy_price * (1 + self.config.stop_profit_pct)
                    sell_reason = f"止盈({self.config.stop_profit_pct*100:.1f}%)"

            # 4. 检查持仓天数（交易日天数）
            if sell_price is None and self.config.hold_days is not None:
                # 计算交易日天数
                buy_date = pd.to_datetime(position.buy_date)
                # 找到买入日期和当前日期在DataFrame中的位置
                buy_idx = df[df['date'] == buy_date].index
                current_idx = df[df['date'] == current_date].index

                if len(buy_idx) > 0 and len(current_idx) > 0:
                    # 计算交易日天数（不包括买入当天）
                    trading_days_held = current_idx[0] - buy_idx[0]
                    if trading_days_held >= self.config.hold_days:
                        sell_price = current_row['close']
                        sell_reason = f"持仓{self.config.hold_days}天到期"

            # 执行卖出
            if sell_price is not None:
                self._execute_sell(position, current_date, sell_price, sell_reason)
                positions_to_remove.append(i)

        # 移除已卖出的持仓
        for i in reversed(positions_to_remove):
            self.positions.pop(i)

    def _execute_sell(self, position: Trade, sell_date, sell_price: float, sell_reason: str):
        """执行卖出"""
        # 计算收益
        sell_amount = position.shares * sell_price
        profit = sell_amount - position.amount
        profit_pct = profit / position.amount

        # 更新持仓记录
        position.sell_date = str(sell_date.date()) if isinstance(sell_date, pd.Timestamp) else sell_date
        position.sell_price = sell_price
        position.sell_reason = sell_reason
        position.profit = profit
        position.profit_pct = profit_pct

        # 增加现金
        self.cash += sell_amount

        # 添加到已平仓列表
        self.closed_trades.append(position)

        logger.debug(f"卖出: {position.stock_code} {position.shares}股 @{sell_price:.2f} "
                    f"收益: {profit:.2f} ({profit_pct*100:.2f}%) 原因: {sell_reason}")

    def _record_daily_value(self, current_date, stock_data: Dict):
        """记录每日账户价值"""
        # 计算持仓市值
        position_value = 0
        for position in self.positions:
            stock_code = position.stock_code
            if stock_code in stock_data:
                df = stock_data[stock_code]
                current_row = df[df['date'] == current_date]
                if len(current_row) > 0:
                    close_price = current_row.iloc[0]['close']
                    position_value += position.shares * close_price

        total_value = self.cash + position_value

        self.daily_values.append({
            'date': current_date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value
        })

    def _close_all_positions(self, stock_data: Dict):
        """清仓所有持仓"""
        for position in self.positions:
            stock_code = position.stock_code
            if stock_code in stock_data:
                df = stock_data[stock_code]
                # 使用最后一个交易日的收盘价
                last_row = df.iloc[-1]
                self._execute_sell(position, last_row['date'], last_row['close'], "回测结束清仓")

        self.positions = []

    def _generate_report(self) -> Dict:
        """生成回测报告"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_return': 0,
                'avg_profit': 0,
                'avg_profit_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'max_profit_trade': None,
                'max_loss_trade': None,
                'final_capital': self.config.initial_capital,
                'trades': [],
                'daily_values': []
            }

        # 基本统计
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.profit > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_profit = sum(t.profit for t in self.closed_trades)
        total_return = total_profit / self.config.initial_capital

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 计算夏普比率（简化版）
        sharpe_ratio = self._calculate_sharpe_ratio()

        # 平均收益
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        avg_profit_pct = np.mean([t.profit_pct for t in self.closed_trades]) if total_trades > 0 else 0

        # 最大盈利和亏损
        max_profit_trade = max(self.closed_trades, key=lambda x: x.profit) if self.closed_trades else None
        max_loss_trade = min(self.closed_trades, key=lambda x: x.profit) if self.closed_trades else None

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': total_trades - len(winning_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'avg_profit': avg_profit,
            'avg_profit_pct': avg_profit_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'max_profit_trade': max_profit_trade,
            'max_loss_trade': max_loss_trade,
            'final_capital': self.cash,
            'trades': self.closed_trades,
            'daily_values': self.daily_values
        }

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.daily_values:
            return 0

        values = [d['total_value'] for d in self.daily_values]
        max_dd = 0
        peak = values[0]

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率（简化版，假设无风险利率为0）"""
        if not self.daily_values or len(self.daily_values) < 2:
            return 0

        # 计算日收益率
        returns = []
        for i in range(1, len(self.daily_values)):
            prev_value = self.daily_values[i-1]['total_value']
            curr_value = self.daily_values[i]['total_value']
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)

        if not returns:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        # 年化夏普比率（假设250个交易日）
        sharpe = (mean_return / std_return) * np.sqrt(250)
        return sharpe
