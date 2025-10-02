"""
数据加载模块
负责从CSV文件加载股票日线数据，并进行清洗和预处理
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """股票数据加载器"""

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化数据加载器

        Args:
            data_dir: 日线数据目录路径
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def load_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        加载指定股票的日线数据

        Args:
            stock_code: 股票代码，格式如 "000001_SZ" 或 "600519_SH"

        Returns:
            处理后的DataFrame，包含列: date, open, high, low, close, volume
            如果文件不存在返回None
        """
        # 确保股票代码格式正确
        if '_' not in stock_code:
            logger.error(f"股票代码格式错误: {stock_code}")
            return None

        file_path = self.data_dir / f"{stock_code}.csv"

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None

        try:
            # 读取CSV文件，第一列是索引列
            df = pd.read_csv(file_path, index_col=0)

            # 数据清洗和预处理
            df = self._preprocess_data(df)

            logger.info(f"成功加载 {stock_code} 数据，共 {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"加载 {stock_code} 数据失败: {e}")
            return None

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理

        Args:
            df: 原始数据DataFrame

        Returns:
            处理后的DataFrame
        """
        # 索引列是日期，将其转为date列
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})

        # 选择需要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols].copy()

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

        # 移除成交量为0的数据（可能是非交易日）
        df = df[df['volume'] > 0].copy()

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # 添加辅助列
        df['body'] = abs(df['close'] - df['open'])  # K线实体长度
        df['range'] = df['high'] - df['low']        # K线总长度

        return df

    def get_latest_n_days(self, stock_code: str, n: int = 500) -> Optional[pd.DataFrame]:
        """
        获取最近N天的数据

        Args:
            stock_code: 股票代码
            n: 天数

        Returns:
            最近N天的数据
        """
        df = self.load_stock_data(stock_code)
        if df is None or len(df) == 0:
            return None

        return df.tail(n).reset_index(drop=True)

    def get_available_stocks(self) -> list:
        """
        获取所有可用的股票代码

        Returns:
            股票代码列表
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        stock_codes = [f.stem for f in csv_files]
        return sorted(stock_codes)


# 工具函数
def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    计算简单移动平均线

    Args:
        series: 价格序列
        period: 周期

    Returns:
        SMA序列
    """
    return series.rolling(window=period, min_periods=1).mean()


def is_downtrend(df: pd.DataFrame, index: int, lookback: int = 5) -> bool:
    """
    判断是否处于下跌趋势

    Args:
        df: 数据DataFrame
        index: 当前索引
        lookback: 回看天数

    Returns:
        是否下跌趋势
    """
    if index < lookback:
        return False

    current_close = df.loc[index, 'close']
    past_close = df.loc[index - lookback, 'close']

    return current_close < past_close


if __name__ == "__main__":
    # 测试代码
    loader = StockDataLoader()

    # 获取所有股票
    stocks = loader.get_available_stocks()
    print(f"共有 {len(stocks)} 支股票")
    print(f"前5支: {stocks[:5]}")

    # 加载示例数据
    if stocks:
        test_stock = stocks[0]
        df = loader.load_stock_data(test_stock)
        if df is not None:
            print(f"\n{test_stock} 数据预览:")
            print(df.head())
            print(f"\n数据形状: {df.shape}")
