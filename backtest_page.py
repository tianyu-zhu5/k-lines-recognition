"""
回测系统页面
提供回测参数配置、运行回测、展示结果的UI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_echarts import st_echarts
from datetime import datetime, timedelta
import os
from typing import List, Dict

from backtest import BacktestEngine, BacktestConfig, Trade
from data_loader import StockDataLoader
from patterns import get_all_patterns


def create_trade_chart(df: pd.DataFrame, trades: List[Trade], stock_code: str) -> dict:
    """
    创建带有买卖点标注的K线图

    Args:
        df: 股票数据DataFrame
        trades: 该股票的交易记录列表
        stock_code: 股票代码

    Returns:
        ECharts配置字典
    """
    # 准备K线数据
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    kline_data = df[['open', 'close', 'low', 'high']].values.tolist()

    # 计算涨跌幅和振幅
    change_pct = ((df['close'] - df['open']) / df['open'] * 100).round(2).tolist()
    change_value = (df['close'] - df['open']).round(2).tolist()
    amplitude = ((df['high'] - df['low']) / df['open'] * 100).round(2).tolist()

    # 准备买卖点标注
    buy_points = []
    sell_points = []

    for trade in trades:
        # 买入点
        buy_points.append({
            "name": "买入",
            "coord": [trade.buy_date, trade.buy_price],
            "value": f"买入\\n{trade.buy_price:.2f}",
            "symbol": "arrow",
            "symbolRotate": 180,  # 箭头向上
            "symbolSize": 15,
            "itemStyle": {
                "color": "#00C853"  # 绿色
            },
            "label": {
                "show": True,
                "formatter": f"买入\\n{trade.buy_price:.2f}",
                "fontSize": 10,
                "color": "#fff",
                "backgroundColor": "#00C853",
                "padding": [2, 4],
                "borderRadius": 3,
                "position": "bottom"
            }
        })

        # 卖出点
        sell_color = "#D32F2F" if trade.profit < 0 else "#FF6F00"  # 亏损红色，盈利橙色
        sell_points.append({
            "name": trade.sell_reason,
            "coord": [trade.sell_date, trade.sell_price],
            "value": f"{trade.sell_reason}\\n{trade.sell_price:.2f}",
            "symbol": "arrow",
            "symbolRotate": 0,  # 箭头向下
            "symbolSize": 15,
            "itemStyle": {
                "color": sell_color
            },
            "label": {
                "show": True,
                "formatter": f"{trade.sell_reason}\\n{trade.sell_price:.2f}\\n{trade.profit_pct*100:.1f}%",
                "fontSize": 10,
                "color": "#fff",
                "backgroundColor": sell_color,
                "padding": [2, 4],
                "borderRadius": 3,
                "position": "top"
            }
        })

    # 合并所有标注点
    all_points = buy_points + sell_points

    # ECharts配置
    option = {
        "title": {
            "text": f"{stock_code} 交易明细",
            "left": "center",
            "top": 10,
            "textStyle": {
                "fontSize": 18,
                "fontWeight": "bold"
            }
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "cross"
            }
        },
        "grid": {
            "left": "3%",
            "right": "3%",
            "bottom": "15%",
            "top": "15%",
            "containLabel": True
        },
        "xAxis": {
            "type": "category",
            "data": dates,
            "scale": True,
            "boundaryGap": False,
            "axisLine": {
                "lineStyle": {
                    "color": "#8392A5"
                }
            },
            "splitLine": {
                "show": True,
                "lineStyle": {
                    "color": "#f0f0f0"
                }
            }
        },
        "yAxis": [
            {
                "scale": True,
                "splitArea": {
                    "show": True
                },
                "axisLine": {
                    "lineStyle": {
                        "color": "#8392A5"
                    }
                }
            },
            {
                "scale": True,
                "show": False
            }
        ],
        "dataZoom": [
            {
                "type": "inside",
                "start": 0,
                "end": 100,
                "minValueSpan": 10
            },
            {
                "type": "slider",
                "show": True,
                "start": 0,
                "end": 100,
                "bottom": "5%"
            }
        ],
        "series": [
            {
                "name": "日K",
                "type": "candlestick",
                "data": kline_data,
                "itemStyle": {
                    "color": "#ef5350",
                    "color0": "#26a69a",
                    "borderColor": "#ef5350",
                    "borderColor0": "#26a69a"
                },
                "markPoint": {
                    "data": all_points,
                    "tooltip": {
                        "formatter": "{b}: {c}"
                    }
                }
            },
            {
                "name": "涨跌额",
                "type": "line",
                "data": change_value,
                "yAxisIndex": 1,
                "showSymbol": False,
                "lineStyle": {
                    "opacity": 0
                }
            },
            {
                "name": "涨跌幅(%)",
                "type": "line",
                "data": change_pct,
                "yAxisIndex": 1,
                "showSymbol": False,
                "lineStyle": {
                    "opacity": 0
                }
            },
            {
                "name": "振幅(%)",
                "type": "line",
                "data": amplitude,
                "yAxisIndex": 1,
                "showSymbol": False,
                "lineStyle": {
                    "opacity": 0
                }
            }
        ]
    }

    return option


def render_backtest_page():
    """渲染回测页面"""

    st.title("📊 K线形态回测系统")
    st.markdown("---")

    # 初始化 session_state
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'backtest_config' not in st.session_state:
        st.session_state.backtest_config = None

    # 初始化数据加载器
    data_loader = StockDataLoader(data_dir="data/daily")

    # 侧边栏：回测参数配置
    with st.sidebar:
        st.header("⚙️ 回测参数配置")

        # 1. 股票池选择
        st.subheader("1️⃣ 股票池")
        pool_option = st.radio(
            "选择股票池",
            ["单个股票", "股票池文件"],
            help="选择回测单个股票还是股票池"
        )

        stock_codes = []
        if pool_option == "单个股票":
            single_stock = st.text_input(
                "股票代码",
                value="000001.SZ",
                help="输入股票代码，如: 000001.SZ"
            )
            if single_stock:
                stock_codes = [single_stock]
        else:
            pool_file = st.text_input(
                "股票池文件路径",
                value="hs300.txt",
                help="相对于项目根目录的路径"
            )
            if pool_file and os.path.exists(pool_file):
                st.success(f"✅ 文件存在")
            elif pool_file:
                st.error(f"❌ 文件不存在: {pool_file}")

        st.markdown("---")

        # 2. 时间范围
        st.subheader("2️⃣ 时间范围")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=365),
                help="回测开始日期"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now(),
                help="回测结束日期"
            )

        st.markdown("---")

        # 3. 形态选择
        st.subheader("3️⃣ 形态选择")
        all_patterns = get_all_patterns()

        # 按类别分组
        pattern_by_category = {}
        for p in all_patterns:
            cat = p['category']
            if cat not in pattern_by_category:
                pattern_by_category[cat] = []
            pattern_by_category[cat].append(p['name'])

        # 分类别选择
        selected_patterns = []
        for category, patterns in pattern_by_category.items():
            with st.expander(f"📁 {category}", expanded=False):
                for pattern_name in patterns:
                    if st.checkbox(pattern_name, key=f"pattern_{pattern_name}"):
                        selected_patterns.append(pattern_name)

        # 快捷选择
        col1, col2 = st.columns(2)
        with col1:
            if st.button("全选", use_container_width=True):
                selected_patterns = [p['name'] for p in all_patterns]
        with col2:
            if st.button("清空", use_container_width=True):
                selected_patterns = []

        st.markdown("---")

        # 4. 买入条件
        st.subheader("4️⃣ 买入条件")
        per_trade_amount = st.number_input(
            "每次买入金额（元）",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="每次买入的固定金额"
        )

        st.info("📌 信号触发后，第二个交易日按开盘价买入")

        st.markdown("---")

        # 5. 卖出条件
        st.subheader("5️⃣ 卖出条件")

        stop_loss_pct = st.slider(
            "止损比例（%）",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="基于买入价的止损比例"
        ) / 100

        stop_profit_pct = st.slider(
            "止盈比例（%）",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="基于买入价的止盈比例"
        ) / 100

        use_hold_days = st.checkbox("启用持仓天数限制", value=True)
        hold_days = None
        if use_hold_days:
            hold_days = st.number_input(
                "持仓天数",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                help="达到天数后按收盘价清仓"
            )

        st.info("📌 止盈止损按盘中价格触发\n📌 同时触发时优先止损")

        st.markdown("---")

        # 6. 资金设置
        st.subheader("6️⃣ 资金设置")
        initial_capital = st.number_input(
            "初始资金（元）",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="回测初始资金"
        )

        st.markdown("---")

        # 运行回测按钮
        run_backtest = st.button(
            "🚀 开始回测",
            type="primary",
            use_container_width=True
        )

        # 清除结果按钮
        if st.session_state.backtest_results is not None:
            if st.button(
                "🗑️ 清除结果",
                use_container_width=True
            ):
                st.session_state.backtest_results = None
                st.session_state.backtest_config = None
                st.rerun()

    # 主区域：显示回测结果
    if run_backtest:
        # 验证参数
        if pool_option == "股票池文件" and not os.path.exists(pool_file):
            st.error(f"❌ 股票池文件不存在: {pool_file}")
            return

        if not selected_patterns:
            st.warning("⚠️ 请至少选择一个形态")
            return

        if not stock_codes and pool_option == "股票池文件":
            # 加载股票池
            engine = BacktestEngine(BacktestConfig(), data_loader)
            stock_codes = engine.load_stock_pool(pool_file)

            if not stock_codes:
                st.error("❌ 股票池为空")
                return

        # 创建回测配置
        config = BacktestConfig(
            initial_capital=initial_capital,
            per_trade_amount=per_trade_amount,
            stop_loss_pct=stop_loss_pct,
            stop_profit_pct=stop_profit_pct,
            hold_days=hold_days,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            pattern_names=selected_patterns
        )

        # 运行回测
        with st.spinner("正在运行回测，请稍候..."):
            engine = BacktestEngine(config, data_loader)
            results = engine.run_backtest(stock_codes)

        # 保存结果到 session_state
        st.session_state.backtest_results = results
        st.session_state.backtest_config = config

    # 显示回测结果（如果存在）
    if st.session_state.backtest_results is not None:
        render_backtest_results(st.session_state.backtest_results, st.session_state.backtest_config)
    elif not run_backtest:
        # 显示欢迎信息
        st.info("""
        ### 👋 欢迎使用K线形态回测系统！

        **使用步骤：**
        1. 在左侧选择股票池（单个股票或股票池文件）
        2. 设置回测时间范围
        3. 选择要回测的K线形态
        4. 配置买入卖出条件
        5. 点击"🚀 开始回测"按钮

        **功能特点：**
        - ✅ 支持多形态组合回测
        - ✅ 灵活的止盈止损设置
        - ✅ 详细的绩效指标分析
        - ✅ 可视化交易记录展示
        """)


def render_backtest_results(results: dict, config: BacktestConfig):
    """渲染回测结果"""

    st.success("✅ 回测完成！")

    # 1. 绩效概览
    st.header("📈 绩效概览")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "总收益",
            f"¥{results['total_profit']:.2f}",
            f"{results['total_return']*100:.2f}%"
        )
    with col2:
        st.metric(
            "胜率",
            f"{results['win_rate']*100:.1f}%",
            f"{results['winning_trades']}/{results['total_trades']}"
        )
    with col3:
        st.metric(
            "最大回撤",
            f"{results['max_drawdown']*100:.2f}%"
        )
    with col4:
        st.metric(
            "夏普比率",
            f"{results['sharpe_ratio']:.2f}"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总交易次数", results['total_trades'])
    with col2:
        st.metric("盈利次数", results['winning_trades'])
    with col3:
        st.metric("亏损次数", results['losing_trades'])
    with col4:
        st.metric(
            "平均收益",
            f"¥{results['avg_profit']:.2f}",
            f"{results['avg_profit_pct']*100:.2f}%"
        )

    st.markdown("---")

    # 2. 资金曲线
    st.header("💰 资金曲线")

    if results['daily_values']:
        df_daily = pd.DataFrame(results['daily_values'])

        fig = go.Figure()

        # 总资产曲线
        fig.add_trace(go.Scatter(
            x=df_daily['date'],
            y=df_daily['total_value'],
            mode='lines',
            name='总资产',
            line=dict(color='#2E86DE', width=2)
        ))

        # 初始资金基准线
        fig.add_trace(go.Scatter(
            x=[df_daily['date'].iloc[0], df_daily['date'].iloc[-1]],
            y=[config.initial_capital, config.initial_capital],
            mode='lines',
            name='初始资金',
            line=dict(color='gray', dash='dash', width=1)
        ))

        fig.update_layout(
            title="账户资金变化",
            xaxis_title="日期",
            yaxis_title="资金（元）",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. 交易记录
    st.header("📋 交易记录")

    if results['trades']:
        # 创建交易记录DataFrame
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

        # 显示统计
        col1, col2 = st.columns(2)
        with col1:
            if results['max_profit_trade']:
                st.success(f"""
                **🎯 最大盈利交易**
                - 股票: {results['max_profit_trade'].stock_code}
                - 盈利: ¥{results['max_profit_trade'].profit:.2f} ({results['max_profit_trade'].profit_pct*100:.2f}%)
                """)
        with col2:
            if results['max_loss_trade']:
                st.error(f"""
                **📉 最大亏损交易**
                - 股票: {results['max_loss_trade'].stock_code}
                - 亏损: ¥{results['max_loss_trade'].profit:.2f} ({results['max_loss_trade'].profit_pct*100:.2f}%)
                """)

        # 显示交易表格
        st.dataframe(df_trades, use_container_width=True, hide_index=True)

        # 下载CSV
        csv = df_trades.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载交易记录CSV",
            data=csv,
            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("没有交易记录")

    st.markdown("---")

    # 4. 收益分布
    st.header("📊 收益分布")

    if results['trades']:
        profit_pcts = [t.profit_pct * 100 for t in results['trades']]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=profit_pcts,
            nbinsx=30,
            name='收益率分布',
            marker_color='#2E86DE'
        ))

        fig.update_layout(
            title="交易收益率分布",
            xaxis_title="收益率（%）",
            yaxis_title="交易次数",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 5. 个股K线图
    st.header("📈 个股交易明细")

    if results['trades']:
        # 获取所有有交易的股票代码（去重）
        traded_stocks = list(set([t.stock_code for t in results['trades']]))
        traded_stocks.sort()

        if traded_stocks:
            # 股票选择器
            selected_stock = st.selectbox(
                "选择股票查看K线图",
                options=traded_stocks,
                index=0,
                help="选择要查看交易明细的股票",
                key="backtest_stock_selector"
            )

            if selected_stock:
                # 获取该股票的所有交易
                stock_trades = [t for t in results['trades'] if t.stock_code == selected_stock]

                # 显示该股票的交易统计
                stock_profit = sum([t.profit for t in stock_trades])
                stock_win_rate = len([t for t in stock_trades if t.profit > 0]) / len(stock_trades) if stock_trades else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("交易次数", len(stock_trades))
                with col2:
                    st.metric("总盈亏", f"¥{stock_profit:.2f}")
                with col3:
                    st.metric("胜率", f"{stock_win_rate*100:.1f}%")

                # 加载股票数据
                data_loader = StockDataLoader(data_dir="data/daily")
                stock_code_converted = selected_stock.replace('.', '_')
                df = data_loader.load_stock_data(stock_code_converted)

                if df is not None and len(df) > 0:
                    # 过滤数据到回测时间范围
                    if config.start_date:
                        df = df[df['date'] >= pd.to_datetime(config.start_date)]
                    if config.end_date:
                        df = df[df['date'] <= pd.to_datetime(config.end_date)]

                    # 创建并显示K线图
                    option = create_trade_chart(df, stock_trades, selected_stock)
                    st_echarts(option, height="600px", key=f"trade_chart_{selected_stock}")
                else:
                    st.error(f"无法加载股票 {selected_stock} 的数据")
        else:
            st.info("没有交易记录")
    else:
        st.info("没有交易记录")
