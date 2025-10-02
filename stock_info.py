"""
股票信息管理模块
负责股票代码、名称的管理和搜索功能
支持按代码、名称、拼音首字母搜索
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from pypinyin import lazy_pinyin
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockInfo:
    """股票信息管理器"""

    def __init__(self, data_dir: str = "data/daily", cache_file: str = "stock_list_cache.json"):
        """
        初始化股票信息管理器

        Args:
            data_dir: 数据目录
            cache_file: 缓存文件路径
        """
        self.data_dir = Path(data_dir)
        self.cache_file = Path(cache_file)
        self.stock_list: List[Dict] = []

        # 加载或生成股票列表
        self._load_or_generate_stock_list()

    def _load_or_generate_stock_list(self):
        """加载缓存或重新生成股票列表"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.stock_list = json.load(f)
                logger.info(f"从缓存加载 {len(self.stock_list)} 支股票信息")
                return
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，将重新生成")

        # 重新生成股票列表
        self._generate_stock_list()

    def _generate_stock_list(self):
        """从CSV文件生成股票列表"""
        if not self.data_dir.exists():
            logger.error(f"数据目录不存在: {self.data_dir}")
            return

        csv_files = list(self.data_dir.glob("*.csv"))
        self.stock_list = []

        for csv_file in csv_files:
            stock_code = csv_file.stem  # 例如: 000001_SZ

            # 解析股票代码
            parts = stock_code.split('_')
            if len(parts) == 2:
                code_num, market = parts

                # 生成股票名称（这里使用代码作为名称，实际应用中可以从其他数据源获取真实名称）
                stock_name = self._get_stock_name(code_num, market)

                # 生成拼音首字母
                pinyin_abbr = ''.join([py[0].upper() for py in lazy_pinyin(stock_name)])

                stock_info = {
                    "code": stock_code,
                    "code_num": code_num,
                    "market": market,
                    "name": stock_name,
                    "pinyin": pinyin_abbr,
                    "display": f"{code_num}.{market} {stock_name}"
                }

                self.stock_list.append(stock_info)

        # 按代码排序
        self.stock_list.sort(key=lambda x: x['code'])

        # 保存到缓存
        self._save_cache()

        logger.info(f"生成 {len(self.stock_list)} 支股票信息")

    def _get_stock_name(self, code_num: str, market: str) -> str:
        """
        获取股票名称（示例实现）
        实际应用中应该从tushare等数据源获取真实名称

        Args:
            code_num: 股票代码数字部分
            market: 市场代码（SH/SZ）

        Returns:
            股票名称
        """
        # 这里使用简单的命名规则，实际应该从数据库或API获取
        # 可以后续扩展对接tushare、akshare等数据源
        return f"股票{code_num}"

    def _save_cache(self):
        """保存股票列表到缓存文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.stock_list, f, ensure_ascii=False, indent=2)
            logger.info(f"股票信息已缓存到 {self.cache_file}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def search(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        搜索股票

        Args:
            keyword: 搜索关键词（代码、名称或拼音首字母）
            limit: 返回结果数量限制

        Returns:
            匹配的股票信息列表
        """
        if not keyword:
            return self.stock_list[:limit]

        keyword_upper = keyword.upper()
        results = []

        for stock in self.stock_list:
            # 匹配代码数字部分
            if keyword in stock['code_num']:
                results.append(stock)
            # 匹配完整代码
            elif keyword_upper in stock['code'].upper():
                results.append(stock)
            # 匹配名称
            elif keyword in stock['name']:
                results.append(stock)
            # 匹配拼音首字母
            elif keyword_upper in stock['pinyin']:
                results.append(stock)

        return results[:limit]

    def get_by_code(self, stock_code: str) -> Optional[Dict]:
        """
        根据代码获取股票信息

        Args:
            stock_code: 股票代码

        Returns:
            股票信息字典，不存在返回None
        """
        for stock in self.stock_list:
            if stock['code'] == stock_code:
                return stock
        return None

    def get_all_stocks(self) -> List[Dict]:
        """获取所有股票信息"""
        return self.stock_list

    def refresh(self):
        """刷新股票列表（重新扫描数据目录）"""
        logger.info("刷新股票列表...")
        self._generate_stock_list()


class StockNameLoader:
    """
    股票名称加载器（可选扩展）
    用于从外部数据源（如tushare）加载真实的股票名称
    """

    @staticmethod
    def load_from_tushare(token: str) -> Dict[str, str]:
        """
        从tushare加载股票名称（需要安装tushare并有token）

        Args:
            token: tushare API token

        Returns:
            股票代码到名称的映射字典
        """
        try:
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()

            # 获取股票列表
            df = pro.stock_basic(exchange='', list_status='L',
                                fields='ts_code,symbol,name,area,industry,market')

            # 转换格式：将 000001.SZ 转为 000001_SZ
            name_dict = {}
            for _, row in df.iterrows():
                ts_code = row['ts_code']  # 例如: 000001.SZ
                code, market = ts_code.split('.')
                stock_code = f"{code}_{market}"
                name_dict[stock_code] = row['name']

            logger.info(f"从tushare加载了 {len(name_dict)} 支股票名称")
            return name_dict

        except ImportError:
            logger.error("tushare未安装，请使用: pip install tushare")
            return {}
        except Exception as e:
            logger.error(f"从tushare加载股票名称失败: {e}")
            return {}

    @staticmethod
    def update_stock_list_with_names(stock_info: 'StockInfo', name_dict: Dict[str, str]):
        """
        使用真实名称更新股票列表

        Args:
            stock_info: StockInfo实例
            name_dict: 代码到名称的映射
        """
        updated_count = 0
        for stock in stock_info.stock_list:
            if stock['code'] in name_dict:
                old_name = stock['name']
                new_name = name_dict[stock['code']]
                stock['name'] = new_name
                stock['display'] = f"{stock['code_num']}.{stock['market']} {new_name}"

                # 更新拼音首字母
                stock['pinyin'] = ''.join([py[0].upper() for py in lazy_pinyin(new_name)])

                updated_count += 1

        logger.info(f"更新了 {updated_count} 支股票的名称")

        # 保存到缓存
        stock_info._save_cache()


if __name__ == "__main__":
    # 测试代码
    stock_info = StockInfo()

    print(f"\n总共 {len(stock_info.stock_list)} 支股票")

    # 测试搜索
    print("\n搜索 '000001':")
    results = stock_info.search('000001')
    for r in results[:5]:
        print(f"  {r['display']}")

    print("\n搜索 '600':")
    results = stock_info.search('600')
    for r in results[:5]:
        print(f"  {r['display']}")

    # 演示如何使用tushare更新名称（需要token）
    print("\n如需使用真实股票名称，请:")
    print("1. 注册tushare账号获取token")
    print("2. 运行以下代码:")
    print("   name_dict = StockNameLoader.load_from_tushare('your_token')")
    print("   StockNameLoader.update_stock_list_with_names(stock_info, name_dict)")
