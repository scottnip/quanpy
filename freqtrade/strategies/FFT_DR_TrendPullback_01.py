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

T: Test
DR: DryRun
LR: LiveRun

基线：
* PaFftStra_T04

优化：
* 动态仓位：5个trade，每个trade占比20%

结果：
* 略好一丢丢

"""

# ##############################################################################################################################################################################################

class FFT_DR_TrendPullback_01(IStrategy):
    ## FFT: 方方土 DR：DryRun TrendPullback：系统类型
    INTERFACE_VERSION = 1

    # Optimal timeframe for the strategy.
    timeframe = '1h'

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

    optimize_label = True

    ## 判断上升趋势：窗口，阈值
    higher_trend_window = IntParameter(5, 10, default=10, space="buy", optimize=optimize_label)
    higher_trend_num = IntParameter(5, 10, default=8, space="buy", optimize=optimize_label)

    ## 判断下降趋势：窗口，阈值
    lower_trend_window = IntParameter(5, 10, default=10, space="buy", optimize=optimize_label)
    lower_trend_num = IntParameter(5, 10, default=8, space="buy", optimize=optimize_label)

    ## 上升趋势下的回调：窗口
    long_pullback_window = IntParameter(2, 5, default=3, space="buy", optimize=optimize_label)
    ## 回调+上升：窗口
    pullback_high_roll_window = IntParameter(1, 5, default=3, space="buy", optimize=optimize_label)
    ## 上升趋势+回调：窗口
    long_pb_trend_pullback_window = IntParameter(1, 5, default=3, space="buy", optimize=optimize_label)
    long_pb_trend_higher_trend_window = IntParameter(1, 10, default=8, space="buy", optimize=optimize_label)

    ## 下降趋势下的回调：窗口
    short_pullback_window = IntParameter(2, 5, default=3, space="buy", optimize=optimize_label)
    ## 回调+下降：窗口
    pullback_low_roll_window = IntParameter(1, 5, default=3, space="buy", optimize=optimize_label)
    ## 下降趋势+回调：窗口
    short_pb_trend_pullback_window = IntParameter(1, 5, default=3, space="buy", optimize=optimize_label)
    short_pb_trend_higher_trend_window = IntParameter(1, 10, default=8, space="buy", optimize=optimize_label)

    # 好的信号K，实体的占比，收盘价的位置
    good_signal_entity_ratio = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=optimize_label)
    good_signal_close_ratio = DecimalParameter(0.5, 0.9, default=0.5, decimals=1, space="buy", optimize=optimize_label)

    # 穿越ema 参数
    crossed_ema_window = IntParameter(1, 10, default=5, space="buy", optimize=optimize_label)

    ## 止损参数
    higher_trend_stoploss_ratio = DecimalParameter(0.5, 0.9, default=0.5, decimals=1, space="sell", optimize=optimize_label)
    lower_trend_stoploss_ratio = DecimalParameter(0.5, 0.9, default=0.5, decimals=1, space="sell", optimize=optimize_label)

    ## 止盈参数
    take_profit_long_factor = DecimalParameter(1.0, 2.0, default=2.0, decimals=1, space="sell", optimize=optimize_label)
    take_profit_short_factor = DecimalParameter(1.0, 2.0, default=2.0, decimals=1, space="sell", optimize=optimize_label)

    plot_config = {
        'main_plot': {
            'higher_trend_max': {'color': '#75fc40'},
            'higher_trend_min': {'color': '#54e5f8'},
            'lower_trend_max': {'color': '#FF9912'},
            'lower_trend_min': {'color': '#A0522D'},
            'ema_20': {'color': '#03A89E'},
            'long_pullback_max': {'color': '#6B8E23'},
            'long_pullback_min': {'color': '#7FFF00'},
            'short_pullback_max': {'color': '#DA70D6'},
            'short_pullback_min': {'color': '#03A89E'},
            'higher_trend_stoploss_price': {'color': '#A020F0'},
            'lower_trend_stoploss_price': {'color': '#698B22'},
        },
        'subplots': {
            'conds': {
                'long_pb_trend': {'color': '#A0522D'},
                'good_long_signal': {'color': '#DA70D6'},
                'pullback_high': {'color': '#6B8E23'},
                'higher_trend': {'color': '#75fc40'},
                'short_pb_trend': {'color': '#228B22'},
                'good_short_signal': {'color': '#A020F0'},
                'pullback_low': {'color': '#698B22'},
                'lower_trend': {'color': '#54e5f8'},
                'long_pullback': {'color': '#7FFF00'},
                'short_pullback': {'color': '#A020F0'},
            },

            # 'dmi': {
                # 'dmi_trend': {'color': '#75fc40'},
                # 'dmi_cross': {'color': '#54e5f8'},
                # 'dmi_cross_sum5': {'color': '#DA70D6'},
                # 'dmi_cross_sum3': {'color': '#7FFF00'},
                # 'dmi_plus': {'color': '#A0522D'},
                # 'dmi_minus': {'color': '#A020F0'},
                # 'dmi_plus_minus_ratio': {'color': '#00EEEE'},
                # 'dmi_minus_plus_ratio': {'color': '#228B22'},
                # 'dmi_plus_slope': {'color': '#7D26CD'},
                # 'dmi_plus_minus_ratio_mean': {'color': '#6B8E23'},
                # 'dmi_minus_plus_ratio_mean': {'color': '#A020F0'},
                # 'dmi_width': {'color': '#03A89E'},
                # 'dmi_width_mean': {'color': '#6B8E23'},
                # 'dmi_width_ratio': {'color': '#698B22'}
            # },
        }
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if self.higher_trend_num.value > self.higher_trend_window.value:
            raise ValueError("参数 higher_trend_num 必须 <= higher_trend_window")
        if self.higher_trend_num.value < 0.5 * self.higher_trend_window.value:
            raise ValueError("参数 higher_trend_num 大于 0.5 * higher_trend_window ")

        if self.lower_trend_num.value > self.lower_trend_window.value:
            raise ValueError("参数 lower_trend_num 必须 <= lower_trend_window")
        if self.lower_trend_num.value < 0.5 * self.lower_trend_window.value:
            raise ValueError("参数 lower_trend_num 大于 0.5 * lower_trend_window ")

    @property
    def protections(self):
        return [
            # # 设置冷却期为 18 根 K 线，即在交易完成后的 18 根 K 线内不允许再次开仓
            # ## FIXME 12 add cooldown
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        ################### 定义基础指标 ##########################
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)

        ################### 定义 其他指标 ###################################
        crossed_above_ema = (qtpylib.crossed_above(dataframe['close'], dataframe['ema_20']))
        dataframe['cross_above_ema'] = crossed_above_ema.astype(int)

        crossed_below_ema = (qtpylib.crossed_below(dataframe['close'], dataframe['ema_20']))
        dataframe['cross_below_ema'] = crossed_below_ema.astype(int)

        ################### 定义 上涨趋势+回调 指标 ##########################
        dataframe['prev_high'] = dataframe['high'].shift(1)
        dataframe['prev_low'] = dataframe['low'].shift(1)

        ## 好的做多信号K线
        ## 1、阳线 2、实体部分超过一半 3、收盘价在high和low的头部1/3处
        good_long_signal = (
                (dataframe['close'] > dataframe['open'])
                & (dataframe['close'] >= (
                dataframe['open'] + (dataframe['high'] - dataframe['low']) * self.good_signal_entity_ratio.value))
                & (dataframe['close'] >= (
                dataframe['low'] + (dataframe['high'] - dataframe['low']) * self.good_signal_close_ratio.value))
        )
        dataframe['good_long_signal'] = good_long_signal.astype(int)

        ## 好的做空信号K线
        ## 1、阴线 2、实体部分超过一半 3、收盘价在high和low的尾部
        good_short_signal = (
                (dataframe['close'] < dataframe['open'])
                & (dataframe['open'] >= (
                dataframe['close'] + (dataframe['high'] - dataframe['low']) * self.good_signal_entity_ratio.value))
                & (dataframe['close'] <= (
                dataframe['high'] - (dataframe['high'] - dataframe['low']) * self.good_signal_close_ratio.value))
        )
        dataframe['good_short_signal'] = good_short_signal.astype(int)

        dataframe['higher_high'] = np.where(
            (dataframe['high'] > dataframe['prev_high'])
            & (dataframe['low'] > dataframe['prev_low'])
            , 1, 0)

        dataframe['higher_trend'] = (
            dataframe['higher_high']
            .rolling(self.higher_trend_window.value)
            .sum()
            .ge(self.higher_trend_num.value)
            .astype(int)
            .fillna(0))

        dataframe['lower_low'] = np.where(
            (dataframe['high'] < dataframe['prev_high'])
            & (dataframe['low'] < dataframe['prev_low'])
            , 1, 0)

        dataframe['lower_trend'] = (
            dataframe['lower_low']
            .rolling(self.lower_trend_window.value)
            .sum()
            .ge(self.lower_trend_num.value)
            .astype(int)
            .fillna(0))

        ## 多头趋势中的回调
        dataframe['long_pullback'] = (
            dataframe['lower_low']
            .rolling(self.long_pullback_window.value)
            .sum()
            .eq(self.long_pullback_window.value)
            .astype(int).fillna(0)
        )

        ## 空头趋势中的回调
        dataframe['short_pullback'] = (
            dataframe['higher_high']
            .rolling(self.short_pullback_window.value)
            .sum()
            .eq(self.short_pullback_window.value)
            .astype(int).fillna(0)
        )

        ## 上升趋势，数K线，高1，2，3
        ## 1、当前high超过前一根K线的high，2、最近3根K线中有回调。
        dataframe['pullback_high'] = np.where(
            (dataframe['high'] > dataframe['high'].shift(1))
            & (dataframe['long_pullback'].rolling(self.pullback_high_roll_window.value).max().eq(1))
            , 1, 0
        )
        ## 下降趋势，数K线，高1，2，3
        ## 1、低点更低（恢复下降趋势），2、最近3根K线中有回调。
        dataframe['pullback_low'] = np.where(
            (dataframe['low'] < dataframe['low'].shift(1))
            & (dataframe['short_pullback'].rolling(self.pullback_low_roll_window.value).max().eq(1))
            , 1, 0
        )

        ## 上涨趋势下的回调上涨策略
        ## 1、上涨趋势 2、回调上涨 3、前一根K线是好的信号K 4、K线是阳线
        dataframe['long_pb_trend'] = np.where(
                # 当前K是好的信号K
                (dataframe['good_long_signal'] == 1)
                & (dataframe['close'] > dataframe['open'])
                # 前long_pb_trend_pullback_window内，有回调
                & (dataframe['pullback_high'].rolling(self.long_pb_trend_pullback_window.value).max() == 1)
                # 在long_pb_trend_pullback_window之前的，XX窗口内，有上涨趋势
                & (dataframe['higher_trend'].shift(self.long_pb_trend_pullback_window.value).rolling(self.long_pb_trend_higher_trend_window.value).max() == 1)
                , 1, 0
        )

        ## 下降趋势下的回调下降策略
        ## 1、下降趋势 2、回调下降 3、前一根K线是好的信号K 4、K线是阴线
        dataframe['short_pb_trend'] = np.where(
                (dataframe['good_short_signal'].shift(1) == 1)
                & (dataframe['close'] < dataframe['open'])
                & (dataframe['pullback_low'].rolling(self.short_pb_trend_pullback_window.value).max() == 1)
                & (dataframe['lower_trend'].shift(self.short_pb_trend_pullback_window.value).rolling(self.short_pb_trend_higher_trend_window.value).max() == 1)
                , 1, 0
        )

        # 上涨趋势，时间窗口内，最大值和最小值
        dataframe['highest_price_higher_trend'] = dataframe['high'].rolling(self.higher_trend_window.value).max()
        dataframe['lowest_price_higher_trend'] = dataframe['low'].rolling(self.higher_trend_window.value).min()

        dataframe['highest_price_long_pullback'] = dataframe['high'].rolling(self.pullback_high_roll_window.value).max()
        dataframe['lowest_price_long_pullback'] = dataframe['low'].rolling(self.pullback_high_roll_window.value).min()

        # 下降趋势，时间窗口内，最大值和最小值
        dataframe['highest_price_lower_trend'] = dataframe['high'].rolling(self.lower_trend_window.value).max()
        dataframe['lowest_price_lower_trend'] = dataframe['low'].rolling(self.lower_trend_window.value).min()

        dataframe['highest_price_short_pullback'] = dataframe['high'].rolling(self.pullback_low_roll_window.value).max()
        dataframe['lowest_price_short_pullback'] = dataframe['low'].rolling(self.pullback_low_roll_window.value).min()

        # 上涨趋势的，上下界
        dataframe['higher_trend_max'] = np.where(dataframe['higher_trend'] == 1, dataframe['highest_price_higher_trend'], np.nan)
        dataframe['higher_trend_max'] = dataframe['higher_trend_max'].ffill()
        dataframe['higher_trend_min'] = np.where(dataframe['higher_trend'] == 1, dataframe['lowest_price_higher_trend'], np.nan)
        dataframe['higher_trend_min'] = dataframe['higher_trend_min'].ffill()

        # 下降趋势的，上下界
        dataframe['lower_trend_max'] = np.where(dataframe['lower_trend'] == 1, dataframe['highest_price_lower_trend'], np.nan)
        dataframe['lower_trend_max'] = dataframe['lower_trend_max'].ffill()
        dataframe['lower_trend_min'] = np.where(dataframe['lower_trend'] == 1, dataframe['lowest_price_lower_trend'], np.nan)
        dataframe['lower_trend_min'] = dataframe['lower_trend_min'].ffill()

        # 上涨趋势之后的pullback的上下界
        dataframe['long_pullback_max'] = np.where(dataframe['long_pullback'] == 1, dataframe['highest_price_long_pullback'], np.nan)
        dataframe['long_pullback_max'] = dataframe['long_pullback_max'].ffill()
        dataframe['long_pullback_min'] = np.where(dataframe['long_pullback'] == 1, dataframe['lowest_price_long_pullback'], np.nan)
        dataframe['long_pullback_min'] = dataframe['long_pullback_min'].ffill()

        # 下降趋势之后的pullback的上下界
        dataframe['short_pullback_max'] = np.where(dataframe['short_pullback'] == 1, dataframe['highest_price_short_pullback'], np.nan)
        dataframe['short_pullback_max'] = dataframe['short_pullback_max'].ffill()
        dataframe['short_pullback_min'] = np.where(dataframe['short_pullback'] == 1, dataframe['lowest_price_short_pullback'], np.nan)
        dataframe['short_pullback_min'] = dataframe['short_pullback_min'].ffill()

        # 上涨趋势的上下界，止损位
        dataframe['higher_trend_stoploss_price'] = self.higher_trend_stoploss_ratio.value * dataframe['long_pullback_min'] + (1 - self.higher_trend_stoploss_ratio.value) * dataframe['higher_trend_min']

        # 下降趋势的止损位
        dataframe['lower_trend_stoploss_price'] = self.lower_trend_stoploss_ratio.value * dataframe['short_pullback_max'] + (1 - self.lower_trend_stoploss_ratio.value) * dataframe['lower_trend_max']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ########################### long below #####################################

        dataframe.loc[
            (
                (dataframe["long_pb_trend"] == 1)
                # 在ema上方
                & (dataframe['close'] > dataframe["ema_20"])
                # 入场K线 不能突破上涨趋势的上界
                & (dataframe['high'] < dataframe['higher_trend_max'])
                # 前面窗口内，不允许下穿ema的情况
                & ((dataframe['cross_below_ema'].rolling(self.crossed_ema_window.value).max()) == 0)
                # 回调不要超过上涨趋势的上界
                & (dataframe['long_pullback_max'] <= dataframe['higher_trend_max'])
                & (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long_pb')

        ########################### short below #####################################

        dataframe.loc[
            (
                (dataframe["short_pb_trend"] == 1)
                # 在ema下方
                & (dataframe['close'] < dataframe["ema_20"])
                # 入场K线 不能突破上涨趋势的上界
                & (dataframe['low'] > dataframe['lower_trend_min'])
                # 前面窗口内，不允许上穿ema的情况
                & ((dataframe['cross_above_ema'].rolling(self.crossed_ema_window.value).max()) == 0)
                # 回调不要超过下降趋势的下界
                & (dataframe['short_pullback_min'] >= dataframe['lower_trend_min'])
                & (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'entry_short_pb')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 此处主要用于兼容性设置，实际退出逻辑由自定义方法处理

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        ##############################  exit long below ############################
        exit_long_1 = (
            (dataframe['cross_below_ema'] == 1)
        )

        conditions_long.append(exit_long_1)
        dataframe.loc[exit_long_1, 'exit_tag'] += 'exit_long_ema'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        ##############################  exit short below ############################
        exit_short_1 = (
            (dataframe['cross_above_ema'] == 1)
        )
        conditions_short.append(exit_short_1)
        dataframe.loc[exit_short_1, 'exit_tag'] += 'exit_short_ema'

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
            if rate < entry_candle['long_pullback_min']:
                return False
            if rate > entry_candle['higher_trend_max']:
                return False

        if 'short' in entry_tag:
            if rate > entry_candle['short_pullback_max']:
                return False
            if rate < entry_candle['lower_trend_min']:
                return False
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

        long_stoploss_price = entry_candle['higher_trend_stoploss_price']
        short_stoploss_price = entry_candle['lower_trend_stoploss_price']

        # print("=============custom_stoploss", open_date)
        # print(trade)
        # print("entry: %s" %entry_candle["date"])
        # print(long_stoploss_price)

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

        # 止损位
        higher_trend_stoploss_price = entry_candle['higher_trend_stoploss_price']
        lower_trend_stoploss_price = entry_candle['lower_trend_stoploss_price']

        # 获取入场价格
        entry_price = trade.open_rate

        # print("================== custom_exit", open_date)
        # print(trade)
        # print("entry_candle: %s" %(entry_candle["date"]))
        # 计算止盈价格
        if trade.is_short:
            # 默认10%的margin, 实际根据止损价，计算margin
            margin_price = (lower_trend_stoploss_price - entry_price) if (entry_price < lower_trend_stoploss_price) else (entry_price * abs(self.stoploss))
            # 根据margin计算止盈价
            take_profit_price = entry_price - self.take_profit_short_factor.value * margin_price
            # take_profit_price = entry_price + self.take_profit_long_factor.value * (entry_price - lowest_price)
            # print("exit Long, E: %s, S: %s, M: %s, T: %s" % (entry_price, higher_trend_stoploss_price, margin_price, take_profit_price))

            if current_rate <= take_profit_price:
                return 'take_profit_short'
        else:
            # 默认10%的margin, 实际根据止损价，计算margin
            margin_price = (entry_price - higher_trend_stoploss_price) if (entry_price > higher_trend_stoploss_price) else (entry_price * abs(self.stoploss))
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
