"""
K线形态识别系统 - 主应用程序
基于 Streamlit + ECharts 实现股票K线形态识别与可视化
"""

import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
from typing import List
import logging

# 导入自定义模块
from data_loader import StockDataLoader
from patterns import recognize_patterns, get_all_patterns, PatternResult
from stock_info import StockInfo

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="K线形态分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 页面选择 ====================

# 在侧边栏顶部添加页面选择
with st.sidebar:
    page = st.radio(
        "导航",
        ["📊 形态识别", "🔬 回测系统"],
        label_visibility="collapsed"
    )
    st.markdown("---")

# 根据选择显示不同页面
if page == "🔬 回测系统":
    from backtest_page import render_backtest_page
    render_backtest_page()
    st.stop()  # 停止执行后面的形态识别页面代码


# ==================== 初始化 ====================

@st.cache_resource
def init_data_loader():
    """初始化数据加载器（缓存）"""
    return StockDataLoader(data_dir="data/daily")


@st.cache_resource
def init_stock_info():
    """初始化股票信息管理器（缓存）"""
    return StockInfo(data_dir="data/daily")


# ==================== 工具函数 ====================

def create_echarts_option(df: pd.DataFrame, patterns: List[PatternResult],
                         stock_name: str) -> dict:
    """
    创建ECharts配置

    Args:
        df: 股票数据DataFrame
        patterns: 形态识别结果列表
        stock_name: 股票名称

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

    # 准备标注数据
    mark_points = []
    for pattern in patterns:
        mark_points.append({
            "name": pattern.pattern_name,
            "coord": [pattern.date, pattern.price],
            "value": f"{pattern.meaning} ({pattern.category})",
            "symbol": "pin",
            "symbolSize": 50,
            "itemStyle": {
                "color": pattern.color
            },
            "label": {
                "show": True,
                "formatter": pattern.pattern_name,
                "fontSize": 10,
                "color": "#fff",
                "backgroundColor": pattern.color,
                "padding": [2, 4],
                "borderRadius": 3
            }
        })

    # ECharts配置
    option = {
        "title": {
            "text": f"{stock_name} K线图",
            "left": "center",
            "top": 10,
            "textStyle": {
                "fontSize": 20,
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
                "start": 50,
                "end": 100,
                "minValueSpan": 10
            },
            {
                "type": "slider",
                "show": True,
                "start": 50,
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
                    "color": "#ef5350",      # 阳线颜色
                    "color0": "#26a69a",     # 阴线颜色
                    "borderColor": "#ef5350",
                    "borderColor0": "#26a69a"
                },
                "markPoint": {
                    "data": mark_points,
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


def render_pattern_statistics(patterns: List[PatternResult]):
    """
    渲染形态统计信息

    Args:
        patterns: 形态识别结果列表
    """
    if not patterns:
        st.info("未识别到任何形态")
        return

    # 统计各形态数量
    pattern_counts = {}
    for p in patterns:
        pattern_counts[p.pattern_name] = pattern_counts.get(p.pattern_name, 0) + 1

    # 按类别统计
    category_counts = {}
    for p in patterns:
        category_counts[p.category] = category_counts.get(p.category, 0) + 1

    st.subheader("📊 形态统计")

    # 按类别显示统计
    st.write("**按类别统计:**")
    cols = st.columns(min(len(category_counts), 5))
    for idx, (cat, count) in enumerate(category_counts.items()):
        with cols[idx % 5]:
            st.metric(label=cat, value=f"{count} 次")

    st.markdown("---")

    # 创建列显示各形态统计
    st.write("**具体形态统计:**")
    cols = st.columns(min(len(pattern_counts), 5))
    for idx, (name, count) in enumerate(pattern_counts.items()):
        with cols[idx % 5]:
            st.metric(label=name, value=f"{count} 次")

    # 显示详细列表
    with st.expander("查看详细识别结果", expanded=False):
        pattern_df = pd.DataFrame([
            {
                "日期": p.date,
                "形态": p.pattern_name,
                "类别": p.category,
                "信号类型": p.signal_type,
                "强度": p.strength,
                "技术含义": p.meaning,
                "价格": f"{p.price:.2f}"
            }
            for p in patterns
        ])
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)


# ==================== 主应用 ====================

def main():
    """主应用函数"""

    # 初始化
    data_loader = init_data_loader()
    stock_info = init_stock_info()

    # 侧边栏
    with st.sidebar:
        st.title("📈 K线形态识别系统")
        st.markdown("---")

        # 股票搜索
        st.subheader("🔍 股票搜索")
        search_keyword = st.text_input(
            "输入股票代码或名称",
            placeholder="例如: 000001, 600519, GZMT",
            help="支持代码、名称、拼音首字母搜索"
        )

        # 搜索结果
        if search_keyword:
            search_results = stock_info.search(search_keyword, limit=20)
        else:
            search_results = stock_info.get_all_stocks()[:20]

        # 股票选择
        if search_results:
            selected_display = st.selectbox(
                "选择股票",
                options=[s['display'] for s in search_results],
                index=0
            )
            # 找到对应的股票代码
            selected_stock = next(
                (s for s in search_results if s['display'] == selected_display),
                None
            )
        else:
            st.warning("未找到匹配的股票")
            selected_stock = None

        st.markdown("---")

        # 显示天数选择
        st.subheader("⚙️ 显示设置")
        n_days = st.slider(
            "显示天数",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="选择显示的K线天数"
        )

        # 形态选择
        st.subheader("🎯 识别形态")
        all_patterns = get_all_patterns()

        # 获取所有类别
        all_categories = sorted(list(set([p['category'] for p in all_patterns])))

        # 类别筛选
        category_selection = st.multiselect(
            "选择形态类别",
            options=all_categories,
            default=all_categories,
            help="按类别筛选形态"
        )

        # 根据选择的类别过滤形态
        filtered_patterns = [p for p in all_patterns if p['category'] in category_selection]

        # 形态选择
        pattern_selection = st.multiselect(
            "选择具体形态",
            options=[p['name'] for p in filtered_patterns],
            default=[p['name'] for p in filtered_patterns],
            help="可以多选或全选"
        )

        # 显示颜色图例
        with st.expander("📊 颜色图例", expanded=False):
            st.markdown("""
            - 🟢 **深绿色** - 底部反转/强烈看涨
            - 🟩 **浅绿色** - 持续上涨
            - 🔴 **红色** - 顶部反转/强烈看跌
            - 🟥 **浅红色** - 持续下跌
            - 🔵 **蓝色** - 支撑信号
            - 🟠 **橙色** - 其他信号
            """)

        st.markdown("---")

        # 刷新按钮
        if st.button("🔄 刷新股票列表", use_container_width=True):
            stock_info.refresh()
            st.rerun()

        # 关于
        with st.expander("ℹ️ 关于"):
            st.markdown("""
            **K线形态识别系统**

            - 版本: 1.0.0
            - 技术栈: Streamlit + ECharts
            - 功能: 自动识别经典K线形态

            **已支持形态:**
            """)
            for cat in all_categories:
                st.markdown(f"**{cat}:**")
                cat_patterns = [p for p in all_patterns if p['category'] == cat]
                for p in cat_patterns:
                    st.markdown(f"  - {p['name']}: {p['meaning']}")

    # 主内容区
    if selected_stock:
        stock_code = selected_stock['code']
        stock_name = selected_stock['display']

        # 加载数据
        with st.spinner(f"正在加载 {stock_name} 的数据..."):
            df = data_loader.get_latest_n_days(stock_code, n=n_days)

        if df is None or len(df) == 0:
            st.error(f"无法加载 {stock_name} 的数据，请检查数据文件是否存在")
            return

        # 识别形态
        with st.spinner("正在识别K线形态..."):
            if pattern_selection and category_selection:
                patterns = recognize_patterns(df, pattern_names=pattern_selection,
                                            categories=category_selection)
            else:
                patterns = []

        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数据天数", f"{len(df)} 天")
        with col2:
            st.metric("最新价格", f"{df.iloc[-1]['close']:.2f}")
        with col3:
            st.metric("识别形态", f"{len(patterns)} 个")
        with col4:
            price_change = df.iloc[-1]['close'] - df.iloc[-2]['close']
            change_pct = (price_change / df.iloc[-2]['close']) * 100
            st.metric(
                "日涨跌幅",
                f"{change_pct:.2f}%",
                delta=f"{price_change:.2f}"
            )

        st.markdown("---")

        # 渲染K线图
        option = create_echarts_option(df, patterns, stock_name)
        st_echarts(option, height="600px", key=f"chart_{stock_code}")

        st.markdown("---")

        # 渲染形态统计
        render_pattern_statistics(patterns)

    else:
        # 欢迎页面
        st.title("📈 K线形态识别系统")
        st.markdown("""
        ### 欢迎使用K线形态识别系统！

        本系统可以自动识别股票K线图中的经典形态，包括：
        - 早晨之星
        - 早晨十字星
        - 好友反攻
        - 锤子线
        - 倒锤子

        #### 使用方法：
        1. 在左侧搜索框输入股票代码或名称
        2. 选择要查看的股票
        3. 调整显示天数和识别形态
        4. 查看K线图和形态标注

        #### 功能特点：
        - ✅ 支持代码、名称、拼音搜索
        - ✅ 交互式K线图（缩放、拖动）
        - ✅ 自动形态识别和标注
        - ✅ 形态统计分析
        - ✅ 模块化设计，易于扩展

        👈 请从左侧开始搜索股票
        """)

        # 显示系统信息
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 系统统计")
            total_stocks = len(stock_info.get_all_stocks())
            total_patterns = len(get_all_patterns())
            st.write(f"- 股票数量: **{total_stocks}** 支")
            st.write(f"- 支持形态: **{total_patterns}** 种")

        with col2:
            st.subheader("🎯 已支持的形态")
            all_cats = sorted(list(set([p['category'] for p in get_all_patterns()])))
            for cat in all_cats:
                st.write(f"**{cat}:**")
                cat_patterns = [p for p in get_all_patterns() if p['category'] == cat]
                for p in cat_patterns:
                    # 使用颜色emoji
                    if "底部反转" in p['category']:
                        emoji = "🟢"
                    elif "持续上涨" in p['category']:
                        emoji = "🟩"
                    elif "顶部反转" in p['category']:
                        emoji = "🔴"
                    elif "持续下跌" in p['category']:
                        emoji = "🟥"
                    elif "支撑" in p['category']:
                        emoji = "🔵"
                    else:
                        emoji = "🟠"
                    st.write(f"  {emoji} **{p['name']}**: {p['meaning']}")


if __name__ == "__main__":
    main()
