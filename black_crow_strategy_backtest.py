"""
黑三鸦策略独立回测脚本
专门用于黑三鸦底部反转策略的回测，参数配置与通用框架隔离
"""

import json
import pandas as pd
from datetime import datetime
from backtest import BacktestEngine, BacktestConfig, FilterConfig
from data_loader import StockDataLoader


def load_strategy_config(config_file: str = "strategies/black_crow_strategy_config.json") -> dict:
    """
    加载策略配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_black_crow_backtest(save_results: bool = True):
    """
    运行黑三鸦策略回测

    Args:
        save_results: 是否保存回测结果
    """
    print("=" * 80)
    print("黑三鸦底部反转策略 - 独立回测")
    print("=" * 80)

    # 加载策略配置
    print("\n📋 加载策略配置...")
    strategy_config = load_strategy_config()

    print(f"策略名称: {strategy_config['strategy_name']}")
    print(f"策略版本: {strategy_config['strategy_version']}")
    print(f"创建日期: {strategy_config['create_date']}")
    print(f"回测区间: {strategy_config['backtest_period']}")
    print(f"股票池: {strategy_config['stock_pool']}")

    # 创建回测配置
    backtest_cfg = strategy_config['backtest_config']
    filter_cfg = strategy_config['filter_config']

    filter_config = FilterConfig(
        enable_drawdown_filter=filter_cfg['enable_drawdown_filter'],
        min_drawdown_pct=filter_cfg['min_drawdown_pct'],
        drawdown_window=filter_cfg['drawdown_window'],

        enable_rsi_filter=filter_cfg['enable_rsi_filter'],
        rsi_threshold=filter_cfg['rsi_threshold'],
        rsi_period=filter_cfg['rsi_period'],

        enable_reversal_filter=filter_cfg['enable_reversal_filter']
    )

    config = BacktestConfig(
        initial_capital=backtest_cfg['initial_capital'],
        per_trade_amount=backtest_cfg['per_trade_amount'],
        stop_loss_pct=backtest_cfg['stop_loss_pct'],
        stop_profit_pct=backtest_cfg['stop_profit_pct'],
        hold_days=backtest_cfg['hold_days'],
        start_date=backtest_cfg['start_date'],
        end_date=backtest_cfg['end_date'],
        pattern_names=backtest_cfg['pattern_names'],
        filter_config=filter_config
    )

    # 打印配置信息
    print("\n⚙️ 回测参数:")
    print(f"  初始资金: ¥{config.initial_capital:,.0f}")
    print(f"  每次买入: ¥{config.per_trade_amount:,.0f}")
    print(f"  止损比例: {config.stop_loss_pct*100:.1f}%")
    print(f"  止盈比例: {config.stop_profit_pct*100:.1f}%")
    print(f"  持仓天数: {config.hold_days}天")
    print(f"  回测形态: {', '.join(config.pattern_names)}")

    print("\n🔍 过滤器配置:")
    print(f"  累计跌幅过滤: {'✓ 启用' if filter_config.enable_drawdown_filter else '✗ 禁用'}")
    if filter_config.enable_drawdown_filter:
        print(f"    - 最小回撤: {filter_config.min_drawdown_pct*100:.0f}%")
        print(f"    - 回看窗口: {filter_config.drawdown_window}天")

    print(f"  RSI超卖过滤: {'✓ 启用' if filter_config.enable_rsi_filter else '✗ 禁用'}")
    if filter_config.enable_rsi_filter:
        print(f"    - RSI阈值: {filter_config.rsi_threshold:.0f}")
        print(f"    - RSI周期: {filter_config.rsi_period}天")

    print(f"  反转K线确认: {'✓ 启用' if filter_config.enable_reversal_filter else '✗ 禁用'}")

    # 加载股票池
    print("\n📊 加载股票池...")
    data_loader = StockDataLoader(data_dir="data/daily")
    engine = BacktestEngine(config, data_loader)
    stock_codes = engine.load_stock_pool(strategy_config['stock_pool'])

    if not stock_codes:
        print("❌ 股票池为空，回测终止")
        return

    print(f"成功加载 {len(stock_codes)} 只股票")

    # 运行回测
    print("\n🚀 开始回测...")
    print("-" * 80)

    results = engine.run_backtest(stock_codes)

    # 显示回测结果
    print("\n" + "=" * 80)
    print("📈 回测结果")
    print("=" * 80)

    print(f"\n总收益: ¥{results['total_profit']:.2f}")
    print(f"总收益率: {results['total_return']*100:.2f}%")
    print(f"胜率: {results['win_rate']*100:.1f}% ({results['winning_trades']}/{results['total_trades']})")
    print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")

    print(f"\n总交易次数: {results['total_trades']}")
    print(f"盈利次数: {results['winning_trades']}")
    print(f"亏损次数: {results['losing_trades']}")
    print(f"平均收益: ¥{results['avg_profit']:.2f}")
    print(f"平均收益率: {results['avg_profit_pct']*100:.2f}%")

    if results['max_profit_trade']:
        print(f"\n最大盈利交易:")
        t = results['max_profit_trade']
        print(f"  {t.stock_code} {t.pattern_name}")
        print(f"  买入: {t.buy_date} @ ¥{t.buy_price:.2f}")
        print(f"  卖出: {t.sell_date} @ ¥{t.sell_price:.2f}")
        print(f"  盈利: ¥{t.profit:.2f} ({t.profit_pct*100:.2f}%)")

    if results['max_loss_trade']:
        print(f"\n最大亏损交易:")
        t = results['max_loss_trade']
        print(f"  {t.stock_code} {t.pattern_name}")
        print(f"  买入: {t.buy_date} @ ¥{t.buy_price:.2f}")
        print(f"  卖出: {t.sell_date} @ ¥{t.sell_price:.2f}")
        print(f"  亏损: ¥{t.profit:.2f} ({t.profit_pct*100:.2f}%)")

    # 保存结果
    if save_results and results['trades']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"results/black_crow_backtest_{timestamp}.csv"

        trades_data = []
        for trade in results['trades']:
            trades_data.append({
                "股票代码": trade.stock_code,
                "形态": trade.pattern_name,
                "买入日期": trade.buy_date,
                "买入价": f"{trade.buy_price:.2f}",
                "股数": trade.shares,
                "买入金额": f"{trade.amount:.2f}",
                "卖出日期": trade.sell_date,
                "卖出价": f"{trade.sell_price:.2f}",
                "卖出原因": trade.sell_reason,
                "盈亏": f"{trade.profit:.2f}",
                "收益率": f"{trade.profit_pct*100:.2f}%"
            })

        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n💾 回测结果已保存至: {output_file}")

    print("\n" + "=" * 80)
    print("回测完成！")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # 运行黑三鸦策略回测
    results = run_black_crow_backtest(save_results=True)

    # 可以在这里添加更多的分析和可视化
    # 例如：生成资金曲线图、交易分布图等
