# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from cmath import nan
from functools import reduce
from math import sqrt
import numpy as np
import pandas as pd
#from conda.gateways.connection.adapters.ftp import data_callback_factory
from pandas import DataFrame
from datetime import datetime
import pandas_ta as pta

from typing import Optional, Union
from scipy.signal import argrelextrema

from finta import TA as fta

from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                stoploss_from_open, stoploss_from_absolute,
                                IntParameter, IStrategy, informative, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import logging
logger = logging.getLogger(__name__)

# custom indicators
# ##############################################################################################################################################################################################
"""
TT: Test DR: DryRun LR: LiveRun

策略：
* FFT_TT_Wedge_03

基线：
* FFT_TT_Wedge_03

优化：
* 增加出场条件：如果出现相反趋势，则退出。

结果：
* 持仓时间变短。
* 2022年变好很多。

优化前：2024年
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃                 ┃        ┃              ┃                 ┃              ┃                  ┃  Win  Draw  Loss ┃                  ┃
┃        Strategy ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃     Avg Duration ┃             Win% ┃         Drawdown ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ FFT_TT_Wedge_03 │    155 │         3.77 │        1660.120 │       166.01 │ 8 days, 12:33:00 │   50     0   105 │     205.011 USDT │
│                 │        │              │                 │              │                  │             32.3 │           15.78% │
└─────────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────────┴──────────────────┴──────────────────┘
优化后：2024年
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃                 ┃        ┃              ┃                 ┃              ┃                 ┃  Win  Draw  Loss ┃                   ┃
┃        Strategy ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃             Win% ┃          Drawdown ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ FFT_TT_Wedge_03 │    211 │         1.96 │        1175.132 │       117.51 │ 2 days, 2:04:00 │  102     0   109 │      174.019 USDT │
│                 │        │              │                 │              │                 │             48.3 │            12.10% │
└─────────────────┴────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴──────────────────┴───────────────────┘

"""

# ##############################################################################################################################################################################################

## FIXME
def get_recent_indicator(df, window, feature, ft_lst):
    """查找窗口window内，最近feature=1的K线，取出ft_lst"""
    # 创建索引序列（flag=1时记录索引）
    s = df.index.to_series().where(df[feature] == 1)
    # 后移1位排除当前K线
    s_shifted = s.shift(1)
    # 滚动查找最近窗口内的最大索引
    max_indices = s_shifted.rolling(window, min_periods=1).max()
    # 映射收盘价并返回
    rst = []
    for ft in ft_lst:
        _ft = df[ft].reindex(max_indices).values
        rst.append(_ft)
    return rst


class FFT_TT_Wedge_03(IStrategy):
    ## FFT: 方方土 DR：DryRun TrendPullback：系统类型
    INTERFACE_VERSION = 1

    # Optimal timeframe for the strategy.
    timeframe = '4h'

    # Can this strategy go short?
    can_short = True

    minimal_roi = {
        "0": 1
    }

    # 启用自定义仓位管理
    use_custom_stake_amount = True

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # 强制取消未完成订单
    cancel_unfilled_order = True

    position_adjustment_enable = False

    stoploss = -0.1

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # 策略配置
    use_custom_stoploss = True
    ignore_roi_if_entry_signal = False

    # 自定义退出逻辑
    use_custom_exit = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    #leverage here
    leverage_optimize = False
    leverage_num = IntParameter(low=1, high=5, default=1, space='buy', optimize=leverage_optimize)
    # leverage_num = 1

    protect_optimize = False
    # cooldown_lookback = IntParameter(1, 40, default=4, space="protection", optimize=protect_optimize)
    max_drawdown_lookback = IntParameter(1, 50, default=2, space="protection", optimize=protect_optimize)
    max_drawdown_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    max_drawdown_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)
    max_allowed_drawdown = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space="protection",
                                            optimize=protect_optimize)
    stoploss_guard_lookback = IntParameter(1, 50, default=8, space="protection", optimize=protect_optimize)
    stoploss_guard_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    stoploss_guard_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)

    ##################  策略参数  #########################
    optimize_label = True

    ## 局部极值：窗口 
    # 4
    swing_order = IntParameter(2, 4, default=2, space="buy", optimize=optimize_label)
    # 8 5
    uni_trend_window = IntParameter(5, 10, default=5, space="buy", optimize=optimize_label)
    uni_trend_thd = IntParameter(3, 7, default=3, space="buy", optimize=optimize_label)
    #uni_trend_window = _trend_window
    #uni_trend_thd = _trend_thd
    # 9 5
    #uni_trend_window = IntParameter(5, 10, default=5, space="buy", optimize=optimize_label)
    #uni_trend_thd = IntParameter(3, 7, default=3, space="buy", optimize=optimize_label)
    #uni_trend_window = _trend_window
    #uni_trend_thd = _trend_thd

    # 5
    bull_push_window = IntParameter(5, 10, default=5, space="buy", optimize=optimize_label)
    # 10
    bear_push_window = IntParameter(5, 10, default=5, space="buy", optimize=optimize_label)

    # 8
    bull_push_pullback_window = CategoricalParameter([8, 10, 12], default=10, space="buy", optimize=optimize_label)
    # 12
    bear_push_pullback_window = CategoricalParameter([8, 10, 12], default=10, space="buy", optimize=optimize_label)

    # 25
    bull_pp2_window = CategoricalParameter([20, 25, 30], default=20, space="buy", optimize=optimize_label)
    # 30
    bear_pp2_window = CategoricalParameter([20, 25, 30], default=20, space="buy", optimize=optimize_label)
    # 20
    bull_pp3_window = CategoricalParameter([20, 25, 30], default=20, space="buy", optimize=optimize_label)
    # 20
    bear_pp3_window = CategoricalParameter([20, 25, 30], default=20, space="buy", optimize=optimize_label)
    # 10
    wedge_top_1_window = CategoricalParameter([5, 10, 15], default=10, space="buy", optimize=optimize_label)
    # 5
    wedge_bottom_1_window = CategoricalParameter([5, 10, 15], default=10, space="buy", optimize=optimize_label)

    # 止盈参数
    # 3.0
    take_profit_long_factor = DecimalParameter(1.0, 2.0, default=1.0, decimals=1, space="sell", optimize=optimize_label)
    # 2.9
    take_profit_short_factor = DecimalParameter(1.0, 2.0, default=1.0, decimals=1, space="sell", optimize=optimize_label)

    plot_config = {
        'main_plot': {
            'ema_20': {'color': '#03A89E'},
            'swing_high': {'color': '#75fc40'},
            'swing_low': {'color': '#54e5f8'},
            'swing_high_imp': {'color': '#FF9912'},
            'swing_low_imp': {'color': '#DA70D6'},

            'bull_p_high': {'color': '#FF9912'},
            'bull_pp_high': {'color': '#DA70D6'},
            'bull_pp_low': {'color': '#03A89E'},
            'bull_pp_high1': {'color': '#A0522D'},
            'bull_pp_low1': {'color': '#6B8E23'},
            'bull_pp_high2': {'color': '#A020F0'},
            'bull_pp_low2': {'color': '#698B22'},
            'bull_pp_high3': {'color': '#03A89E'},

            'bear_p_low': {'color': '#FF9912'},
            'bear_pp_high': {'color': '#DA70D6'},
            'bear_pp_low': {'color': '#03A89E'},
            'bear_pp_high1': {'color': '#A0522D'},
            'bear_pp_low1': {'color': '#A020F0'},
            'bear_pp_high2': {'color': '#A020F0'},
            'bear_pp_low2': {'color': '#698B22'},
            'bear_pp_low3': {'color': '#03A89E'},
            # 'bull_pp_pullback_low': {'color': '#7FFF00'},
            # 'bull_pp_low_1': {'color': '#F4A460'},
            # 'bull_pp_low_2': {'color': '#40E0D0'},
            # 'bull_pp_low_3': {'color': '#DA70D6'},

        },
        'subplots': {
            'conds': {
                'local_high': {'color': '#A0522D'},
                'local_low': {'color': '#DA70D6'},

                'high_K': {'color': '#6B8E23'},
                'higher_trend': {'color': '#75fc40'},
                'bull_push': {'color': '#228B22'},
                'bull_push_pullback': {'color': '#A020F0'},
                'bull_pp_2': {'color': '#698B22'},
                'bull_pp_3': {'color': '#54e5f8'},
                'wedge_top_overlap': {'color': '#E3170D'},

                'low_K': {'color': '#6B8E23'},
                'lower_trend': {'color': '#75fc40'},
                'bear_push': {'color': '#228B22'},
                'bear_push_pullback': {'color': '#A020F0'},
                'bear_pp_2': {'color': '#698B22'},
                'bear_pp_3': {'color': '#54e5f8'},
                'wedge_bottom_overlap': {'color': '#E3170D'},
                # 'short_pullback': {'color': '#A020F0'},
            }
        }
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if self.uni_trend_thd.value > self.uni_trend_window.value:
            raise ValueError("参数错误！")

    @property
    def protections(self):
        return [
            # # 设置冷却期为 18 根 K 线，即在交易完成后的 18 根 K 线内不允许再次开仓
            # {
            #     "method": "CooldownPeriod", # 在一次交易完成后，设置一个冷却期，在此期间不允许再次开仓，以避免过度交易和频繁操作
            #     "stop_duration_candles": 5 # 冷却期的时长为 5 根 K 线所代表的时
            # },
            # 如果回撤超过设定的阈值，策略将暂停交易一段时间
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value, # 回撤计算的回看周期
                "trade_limit": self.max_drawdown_trade_limit.value, # 回看周期内允许的最大交易次数
                "stop_duration_candles": self.max_drawdown_stop_duration.value, # 暂停交易的持续时间
                "max_allowed_drawdown": self.max_allowed_drawdown.value # 允许的最大回撤比例
            },
            # 如果止损触发次数超过设定的阈值，策略将暂停交易一段时间
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value, # 止损触发的回看周期
                "trade_limit": self.stoploss_guard_trade_limit.value, # 回看周期内允许的最大止损触发次数
                "stop_duration_candles": self.stoploss_guard_stop_duration.value, # 暂停交易的持续时间
                "only_per_pair": False # 设置为 False ，则对所有交易对生效
            }
        ]


    def calc_basic_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)

        # 局部极值点
        dataframe['prev_high'] = dataframe['high'].shift(self.swing_order.value)
        dataframe['swing_high'] = dataframe.iloc[argrelextrema(dataframe['prev_high'].values, np.greater, order=self.swing_order.value)[0]]['prev_high']
        dataframe['local_high'] = np.where(dataframe['swing_high'].isna(), 0, 1)

        dataframe['prev_low'] = dataframe['low'].shift(self.swing_order.value)
        dataframe['swing_low'] = dataframe.iloc[argrelextrema(dataframe['prev_low'].values, np.less, order=self.swing_order.value)[0]]['prev_low']
        dataframe['local_low'] = np.where(dataframe['swing_low'].isna(), 0, 1)

        dataframe['swing_high'] = dataframe['swing_high'].ffill()
        dataframe['swing_low'] = dataframe['swing_low'].ffill()

        # 重要的高点：该高点之后，低点被突破。
        dataframe['swing_high_imp'] = np.where(
            (dataframe['swing_low'] < dataframe['swing_low'].shift(1))
            , dataframe['swing_high'], np.nan
        )
        dataframe['swing_high_imp'] = dataframe['swing_high_imp'].ffill()

        # 重要的低点：该低点之后，高点被突破。
        dataframe['swing_low_imp'] = np.where(
            (dataframe['swing_high'] > dataframe['swing_high'].shift(1))
            , dataframe['swing_low'], np.nan
        )
        dataframe['swing_low_imp'] = dataframe['swing_low_imp'].ffill()

        return dataframe

    def calc_wedge_top_indicators(self, dataframe: DataFrame) -> DataFrame:
        ########
        # HH & HL
        dataframe['high_K'] = np.where(
            (dataframe['high'] > dataframe['high'].shift(1))
            & (dataframe['low'] > dataframe['low'].shift(1))
            , 1, 0
        )

        dataframe['higher_trend'] = np.where(
            (dataframe['high_K'].rolling(self.uni_trend_window.value).sum() >= self.uni_trend_thd.value)
            , 1, 0
        )

        ## 上推
        bull_push = (
                (dataframe['local_high'] == 1)  # 条件1：当前K线local_high==1
                & (  # 合并以下两个子条件
                    # 生成10个条件组成的数组（每个k对应一个条件）
                    np.any(
                        [
                            (dataframe['higher_trend'].shift(k) == 1)  # 子条件1：k周期前存在higher_trend信号
                            & (  # 子条件2：中间无local_low信号
                                    (k == 1)  # k=1时没有中间K线
                                    | (  # k>1时检查中间区间
                                            dataframe['local_low']
                                            .shift(1)
                                            .rolling(window=k-1, min_periods=0)
                                            .sum() == 0  # 窗口内无local_low
                                    )
                            )
                            for k in range(1, self.bull_push_window.value)  # 遍历1-10周期
                        ],
                        axis=0  # 在条件数组的列方向做OR运算
                    )
                )
        )

        dataframe['bull_push'] = np.where(bull_push, 1, 0)
        dataframe['bull_p_high'] = np.where(bull_push, dataframe['swing_high'], np.nan)
        dataframe['bull_p_high'] = dataframe['bull_p_high'].ffill()

        ## 上推&回调
        bull_push_pullback = (
                (dataframe['local_low'] == 1)
                & (dataframe['bull_push'].rolling(self.bull_push_pullback_window.value).max() == 1)
        )

        dataframe['bull_push_pullback'] = np.where(bull_push_pullback, 1, 0)
        dataframe['bull_pp_high'] = np.where(bull_push_pullback, dataframe['bull_p_high'], np.nan)
        dataframe['bull_pp_high'] = dataframe['bull_pp_high'].ffill()
        dataframe['bull_pp_low'] = np.where(bull_push_pullback, dataframe['swing_low'], np.nan)
        dataframe['bull_pp_low'] = dataframe['bull_pp_low'].ffill()

        [bull_pp_high1, bull_pp_low1] = get_recent_indicator(
            dataframe,
            self.bull_pp2_window.value,
            'bull_push_pullback',
            ['bull_pp_high', 'bull_pp_low']
        )
        dataframe['bull_pp_high1'] = bull_pp_high1
        dataframe['bull_pp_low1'] = bull_pp_low1

        bull_pp_2_conds = (
                (dataframe['bull_push_pullback'] == 1)
                & (dataframe['bull_pp_high1'].notna())
                & (dataframe['bull_pp_high1'] != dataframe['bull_pp_high'])
        )
        dataframe['bull_pp_2'] = np.where(bull_pp_2_conds, 1, 0)
        dataframe['bull_pp_high2'] = np.where(bull_pp_2_conds, dataframe['bull_pp_high'], np.nan)
        dataframe['bull_pp_low2'] = np.where(bull_pp_2_conds, dataframe['bull_pp_low'], np.nan)

        [bull_pp_high1, bull_pp_low1, bull_pp_high2, bull_pp_low2] = get_recent_indicator(
            dataframe,
            self.bull_pp3_window.value,
            'bull_pp_2',
            ['bull_pp_high1', 'bull_pp_low1', 'bull_pp_high2', 'bull_pp_low2']
        )

        bull_pp_3_conds = (
                (dataframe['bull_push'] == 1)
                & (dataframe['bull_pp_2'].rolling(self.bull_pp3_window.value).max() == 1)
        )

        dataframe['bull_pp_3'] = np.where(bull_pp_3_conds, 1, 0)
        dataframe['bull_pp_high3'] = np.where(bull_pp_3_conds, dataframe['bull_p_high'], np.nan)
        dataframe['bull_pp_high1'] = np.where(bull_pp_3_conds, bull_pp_high1, np.nan)
        dataframe['bull_pp_low1'] = np.where(bull_pp_3_conds, bull_pp_low1, np.nan)
        dataframe['bull_pp_high2'] = np.where(bull_pp_3_conds, bull_pp_high2, np.nan)
        dataframe['bull_pp_low2'] = np.where(bull_pp_3_conds, bull_pp_low2, np.nan)

        # 填充空值
        dataframe['bull_pp_high1'] = dataframe['bull_pp_high1'].ffill()
        dataframe['bull_pp_high2'] = dataframe['bull_pp_high2'].ffill()
        dataframe['bull_pp_high3'] = dataframe['bull_pp_high3'].ffill()

        ## 识别三推楔形，有重叠区域。
        wedge_top_overlap = (
                (dataframe['bull_pp_3'] == 1)
                # 上推2的回调和上推1，有重叠区域
                & (dataframe['bull_pp_low2'] < dataframe['bull_pp_high1'])
        )
        dataframe['wedge_top_overlap'] = np.where(wedge_top_overlap, 1, 0)

        ## 入场条件1，有重叠，价格低于上推2的高点
        dataframe['wedge_top_1'] = np.where(
            (dataframe['close'] < dataframe['bull_pp_high2'])
            & (dataframe['wedge_top_overlap'].rolling(self.wedge_top_1_window.value).max() == 1)
            , 1, 0
        )
        return dataframe

    def calc_wedge_bottom_indicators(self, dataframe: DataFrame) -> DataFrame:
        ########
        # LH & LL
        dataframe['low_K'] = np.where(
            (dataframe['high'] < dataframe['high'].shift(1))
            & (dataframe['low'] < dataframe['low'].shift(1))
            , 1, 0
        )

        dataframe['lower_trend'] = np.where(
            (dataframe['low_K'].rolling(self.uni_trend_window.value).sum() >= self.uni_trend_thd.value)
            , 1, 0
        )

        ## 下推
        # 1、当前K线是局部低点local_low==1
        # 2、最近窗口内，存在lower_trend==1的K线
        # 3、lower_trend=1 和 local_low==1的K线之间，没有local_high=1的K线
        bear_push = (
                (dataframe['local_low'] == 1)  # 条件1：当前K线local_low==1
                & (  # 合并以下两个子条件
                    # 生成10个条件组成的数组（每个k对应一个条件）
                    np.any(
                        [
                            (dataframe['lower_trend'].shift(k) == 1)  # 子条件1：k周期前存在lower_trend信号
                            & (  # 子条件2：中间无local_high信号
                                    (k == 1)  # k=1时没有中间K线
                                    | (  # k>1时检查中间区间
                                            dataframe['local_high']
                                            .shift(1)  # 定位到lower_trend信号的下一个K线
                                            .rolling(window=k - 1, min_periods=0)
                                            .sum() == 0  # 窗口内无local_high
                                    )
                            )
                            for k in range(1, self.bear_push_window.value)  # 遍历1-10周期
                        ],
                        axis=0  # 在条件数组的列方向做OR运算
                    )
                )
        )

        dataframe['bear_push'] = np.where(bear_push, 1, 0)
        dataframe['bear_p_low'] = np.where(bear_push, dataframe['swing_low'], np.nan)
        dataframe['bear_p_low'] = dataframe['bear_p_low'].ffill()

        ## 下推&回调
        bear_push_pullback = (
                (dataframe['local_high'] == 1)
                & (  # 合并以下两个子条件
                    np.any(
                        [
                            (dataframe['bear_push'].shift(k) == 1)
                            & (
                                    (k == 1)
                                    | (
                                            dataframe['local_low']
                                            .shift(1)
                                            .rolling(window=k - 1, min_periods=0)
                                            .sum() == 0
                                    )
                            )
                            for k in range(1, self.bear_push_pullback_window.value)
                        ],
                        axis=0  # 在条件数组的列方向做OR运算
                    )
                )
        )
        dataframe['bear_push_pullback'] = np.where(bear_push_pullback, 1, 0)
        dataframe['bear_pp_low'] = np.where(bear_push_pullback, dataframe['bear_p_low'], np.nan)
        dataframe['bear_pp_low'] = dataframe['bear_pp_low'].ffill()
        dataframe['bear_pp_high'] = np.where(bear_push_pullback, dataframe['swing_high'], np.nan)
        dataframe['bear_pp_high'] = dataframe['bear_pp_high'].ffill()

        [bear_pp_high1, bear_pp_low1] = get_recent_indicator(
            dataframe,
            self.bear_pp2_window.value,
            'bear_push_pullback',
            ['bear_pp_high', 'bear_pp_low']
        )
        dataframe['bear_pp_high1'] = bear_pp_high1
        dataframe['bear_pp_low1'] = bear_pp_low1

        bear_pp_2_conds = (
                (dataframe['bear_push_pullback'] == 1)
                & (dataframe['bear_pp_low1'].notna())
                & (dataframe['bear_pp_low1'] != dataframe['bear_pp_low'])
        )
        dataframe['bear_pp_2'] = np.where(bear_pp_2_conds, 1, 0)
        dataframe['bear_pp_high2'] = np.where(bear_pp_2_conds, dataframe['bear_pp_high'], np.nan)
        dataframe['bear_pp_low2'] = np.where(bear_pp_2_conds, dataframe['bear_pp_low'], np.nan)

        [bear_pp_high1, bear_pp_low1, bear_pp_high2, bear_pp_low2] = get_recent_indicator(
            dataframe,
            self.bear_pp3_window.value,
            'bear_pp_2',
            ['bear_pp_high1', 'bear_pp_low1', 'bear_pp_high2', 'bear_pp_low2']
        )

        bear_pp_3_conds = (
                (dataframe['bear_push'] == 1)
                & (dataframe['bear_pp_2'].rolling(self.bear_pp3_window.value).max() == 1)
        )

        dataframe['bear_pp_3'] = np.where(bear_pp_3_conds, 1, 0)
        dataframe['bear_pp_low3'] = np.where(bear_pp_3_conds, dataframe['bear_p_low'], np.nan)

        dataframe['bear_pp_high1'] = np.where(bear_pp_3_conds, bear_pp_high1, np.nan)
        dataframe['bear_pp_low1'] = np.where(bear_pp_3_conds, bear_pp_low1, np.nan)
        dataframe['bear_pp_high2'] = np.where(bear_pp_3_conds, bear_pp_high2, np.nan)
        dataframe['bear_pp_low2'] = np.where(bear_pp_3_conds, bear_pp_low2, np.nan)

        ## 识别三推楔形，有重叠区域。
        wedge_bottom_overlap = (
                (dataframe['bear_pp_3'] == 1)
                # 下推2的回调和下推1，有重叠区域
                & (dataframe['bear_pp_high2'] > dataframe['bear_pp_low1'])
        )
        dataframe['wedge_bottom_overlap'] = np.where(wedge_bottom_overlap, 1, 0)

        dataframe['bear_pp_low1'] = dataframe['bear_pp_low1'].ffill()
        dataframe['bear_pp_low2'] = dataframe['bear_pp_low2'].ffill()
        dataframe['bear_pp_low3'] = dataframe['bear_pp_low3'].ffill()

        ## 入场条件1，有重叠，价格高于下推2的低点
        dataframe['wedge_bottom_1'] = np.where(
            (dataframe['close'] > dataframe['bear_pp_low2'])
            & (dataframe['wedge_bottom_overlap'].rolling(self.wedge_bottom_1_window.value).max() == 1)
            , 1, 0
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        ################### 定义基础指标 ##########################

        # 基础指标
        dataframe = self.calc_basic_indicators(dataframe)

        # 三推楔形顶
        dataframe = self.calc_wedge_top_indicators(dataframe)

        # 三推楔形底
        dataframe = self.calc_wedge_bottom_indicators(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ########################### long below #####################################

        dataframe.loc[
            (
                (dataframe['wedge_bottom_1'] == 1)
                & (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long')

        ########################### short below #####################################

        dataframe.loc[
            (
                (dataframe["wedge_top_1"] == 1)
                & (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'entry_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 此处主要用于兼容性设置，实际退出逻辑由自定义方法处理

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        # ##############################  exit long below ############################
        exit_long_1 = (
            (dataframe['lower_trend'] == 1)
        )

        conditions_long.append(exit_long_1)
        dataframe.loc[exit_long_1, 'exit_tag'] += 'exit_long_lower_trend'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        # ##############################  exit short below ############################
        exit_short_1 = (
            (dataframe['higher_trend'] == 1)
        )
        conditions_short.append(exit_short_1)
        dataframe.loc[exit_short_1, 'exit_tag'] += 'exit_short_higher_trend'

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'exit_short'] = 1

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
                            current_time: datetime, entry_tag: Optional[str],
                            **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False
        entry_candle = dataframe.iloc[-1].squeeze()
        if 'long' in entry_tag:
            if rate < entry_candle['bear_pp_low3']:
                return False
            # if rate > entry_candle['higher_trend_max']:
            #     return False

        if 'short' in entry_tag:
            if rate > entry_candle['bull_pp_high3']:
                return False
            # if rate < entry_candle['lower_trend_min']:
            #     return False
        return True

    def custom_stoploss(self, current_time: datetime, current_rate: float,
                         current_profit: float, trade: 'Trade', **kwargs) -> float:
        # 获取交易开仓时的K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # 确保时间列为 UTC 时区
        dataframe = dataframe.assign(
            date_utc=pd.to_datetime(dataframe["date"], utc=True)
        )

        # 找到入场时间对应的 K 线索引
        open_date = trade.open_date_utc
        open_rate = trade.open_rate

        # 找到开仓时的K线索引位置
        dataframe = dataframe.loc[dataframe['date'] < open_date]
        if len(dataframe) == 0:
            return self.stoploss

        entry_candle = dataframe.iloc[-1]

        long_stoploss_price = entry_candle['bear_pp_low3']
        short_stoploss_price = entry_candle['bull_pp_high3']

        # # 获取后续所有已闭合K线
        # # FIXME
        # subsequent_data = dataframe[dataframe.index > entry_candle.name]
        # if not subsequent_data.empty:
        #     # 计算后续K线中的最低swing_high_imp
        #     short_min_imp = subsequent_data['swing_high_imp'].min()
        #     short_stoploss_price = min(short_stoploss_price, short_min_imp)

        # print("=============custom_stoploss", open_date)
        # print(trade)
        # print("entry: %s" %entry_candle["date"])
        # print(long_stoploss_price)
        #
        # 计算动态止损
        if trade.is_short:
            # open_rate，交易的入场价
            if open_rate < short_stoploss_price:
                stoploss_price = short_stoploss_price
                stoploss_ratio = (stoploss_price / current_rate) - 1
                return stoploss_ratio
        else:
            # open_rate，交易的入场价
            if open_rate > long_stoploss_price:
                stoploss_price = long_stoploss_price
                stoploss_ratio= (stoploss_price / current_rate) - 1
                return stoploss_ratio
        #
        # ## 动态止盈
        # if current_profit >= 0.2:
        #     return -0.01
        # if current_profit >= 0.15:
        #     return -0.05
        # if current_profit >= 0.1:
        #     return -0.05

        return self.stoploss

    def custom_exit(self, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        # 获取交易开仓时的K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # 确保时间列为 UTC 时区
        dataframe = dataframe.assign(date_utc=pd.to_datetime(dataframe["date"], utc=True))

        # 找到入场时间对应的 K 线索引
        open_date = trade.open_date_utc

        # 找到开仓时的K线索引位置
        dataframe = dataframe.loc[dataframe['date'] < open_date]
        if len(dataframe) == 0:
            return None

        entry_candle = dataframe.iloc[-1]

        # # 止损位
        long_stoploss_price = entry_candle['bear_pp_low3']
        short_stoploss_price = entry_candle['bull_pp_high3']

        # 获取入场价格
        entry_price = trade.open_rate

        # print("================== custom_exit", open_date)
        # print(trade)
        # print("entry_candle: %s" %(entry_candle["date"]))
        # 计算止盈价格
        if trade.is_short:
            # 默认10%的margin, 实际根据止损价，计算margin
            margin_price = (short_stoploss_price - entry_price) if (entry_price < short_stoploss_price) else (entry_price * abs(self.stoploss))
            # 根据margin计算止盈价
            take_profit_price = entry_price - self.take_profit_short_factor.value * margin_price
            # take_profit_price = entry_price + self.take_profit_long_factor.value * (entry_price - lowest_price)
            # print("exit Long, E: %s, S: %s, M: %s, T: %s" % (entry_price, higher_trend_stoploss_price, margin_price, take_profit_price))

            if current_rate <= take_profit_price:
                return 'take_profit_short'
        else:
            # 默认10%的margin, 实际根据止损价，计算margin
            margin_price = (entry_price - long_stoploss_price) if (entry_price > long_stoploss_price) else (entry_price * abs(self.stoploss))
            # 根据margin计算止盈价
            take_profit_price = entry_price + self.take_profit_long_factor.value * margin_price
            #take_profit_price = entry_price + self.take_profit_long_factor.value * (entry_price - lowest_price)
            #print("exit Long, E: %s, S: %s, M: %s, T: %s" % (entry_price, higher_trend_stoploss_price, margin_price, take_profit_price))

            if current_rate >= take_profit_price:
                #print("exit Long H: %s, L: %s, E: %s, T: %s" % (highest_price, lowest_price, entry_price, take_profit_price))
                #print("done!!!!! %s" %current_rate)
                return 'take_profit_long'

        return None

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: float, max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:
        # 获取当前总资产（以基础货币计算）
        total_value = self.wallets.get_total(self.config['stake_currency'])

        # 计算每个仓位的大小
        stake = 1.0 * total_value / self.config['max_open_trades']

        # 确保不低于最小交易金额
        if stake < min_stake:
            return 0  # 无法开仓

        # 确保不超过最大限额
        return min(stake, max_stake)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        return self.leverage_num.value
