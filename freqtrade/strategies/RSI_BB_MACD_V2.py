# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from cmath import nan
from functools import reduce
from math import sqrt
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, stoploss_from_open, DecimalParameter,
                                IntParameter, IStrategy, informative, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


# custom indicators
# ##############################################################################################################################################################################################

def trade_signal(dataframe, rsi_tp = 14, bb_tp = 20, suffix=""):
    # Compute indicators
    dataframe[f'RSI_{suffix}'] = ta.RSI(dataframe['close'], timeperiod=int(rsi_tp))
    dataframe[f'upper_band_{suffix}'], dataframe[f'middle_band_{suffix}'], dataframe[f'lower_band_{suffix}'] = ta.BBANDS(dataframe['close'], timeperiod=int(bb_tp))
    dataframe['macd'], dataframe['signal'], _ = ta.MACD(dataframe['close'])

    # LONG Trade conditions
    # 做多趋势
    conditions_long = (
            (dataframe[f'RSI_{suffix}'] > 50) &
            (dataframe['close'] > dataframe[f'middle_band_{suffix}']) &
            (dataframe['close'] < dataframe[f'upper_band_{suffix}']) &
            (dataframe['macd'] > dataframe['signal']) &
            ((dataframe['high'] - dataframe['close']) < (dataframe['close'] - dataframe['open'])) &
            (dataframe['close'] > dataframe['open'])
    )
    # 做空趋势
    conditions_short = (
            (dataframe[f'RSI_{suffix}'] < 50) &
            (dataframe['close'] < dataframe[f'middle_band_{suffix}']) &
            (dataframe['close'] > dataframe[f'lower_band_{suffix}']) &
            (dataframe['macd'] < dataframe['signal']) &
            ((dataframe['close'] - dataframe['low']) < (dataframe['open'] - dataframe['close'])) &
            (dataframe['close'] < dataframe['open'])
    )

    # dataframe['signal'] = 0
    dataframe.loc[conditions_long, 'trend'] = 1
    dataframe.loc[conditions_short, 'trend'] = -1

    return dataframe

# ##############################################################################################################################################################################################

class RSI_BB_MACD_V2(IStrategy):


    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short = True

    # risk_c = DecimalParameter(0.025, 0.01, 0.1, decimals=2, space='buy')

    # Minimal ROI designed for the strategy.
    minimal_roi = {
    #   "0": 0.282,
    #   "138": 0.179,
    #   "310": 0.089,
    #   "877": 0
    # '0': 0.344, '260': 0.225, '486': 0.09, '796': 0
    "0": 0.184,
    "416": 0.14,
    "933": 0.073,
    "1982": 0

    #   '0': 0.279, '154': 0.122, '376': 0.085, '456': 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    ## FIXME
    #stoploss = -0.317
    stoploss = -0.15

    # Trailing stop:
    trailing_stop = True
    # FIXME
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.022
    #trailing_stop_positive = 0.01
    #trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    #leverage here
    leverage_optimize = True
    leverage_num = IntParameter(low=1, high=5, default=5, space='buy', optimize=leverage_optimize)

    # Strategy parameters
    parameters_opt = True

    adx_long_max_1 = DecimalParameter(6.1, 10.0, default=6.5, decimals = 1, space="buy", optimize = parameters_opt)
    adx_long_min_1 = DecimalParameter(4.0, 6.0, default=5.7, decimals = 1, space="buy", optimize = parameters_opt)

    adx_long_max_2 = DecimalParameter(24.9, 60.0, default=50.7, decimals = 1, space="buy", optimize = parameters_opt)
    adx_long_min_2 = DecimalParameter(18.5, 21.0, default=20.9, decimals = 1, space="buy", optimize = parameters_opt)

    adx_short_max_1 = DecimalParameter(14.1, 21.9, default=21.4, decimals = 1, space="buy", optimize = parameters_opt)
    adx_short_min_1 = DecimalParameter(8.7, 14, default=9.9, decimals = 1, space="buy", optimize = parameters_opt)

    adx_short_max_2 = DecimalParameter(30.6, 55.0, default=50.8, decimals = 1, space="buy", optimize = parameters_opt)
    adx_short_min_2 = DecimalParameter(25.0, 30.5, default=30.3, decimals = 1, space="buy", optimize = parameters_opt)

    bb_tp_l = IntParameter(15, 35, default=16, space="buy", optimize= parameters_opt)
    bb_tp_s = IntParameter(15, 35, default=20, space="buy", optimize= parameters_opt)

    rsi_tp_l = IntParameter(10, 25, default=22, space="buy", optimize= parameters_opt)
    rsi_tp_s = IntParameter(10, 25, default=17, space="buy", optimize= parameters_opt)

    volume_check = IntParameter(10, 45, default=38, space="buy", optimize= parameters_opt)
    volume_check_s = IntParameter(15, 45, default=20, space="buy", optimize= parameters_opt)

    atr_long_mul = DecimalParameter(1.1, 6.0, default=3.8, decimals = 1, space="sell", optimize = parameters_opt)
    atr_short_mul = DecimalParameter(1.1, 6.0, default=5.0, decimals = 1, space="sell", optimize = parameters_opt)

    ema_period_l_exit = IntParameter(22, 200, default=91, space="sell", optimize= parameters_opt)
    ema_period_s_exit = IntParameter(22, 200, default=147, space="sell", optimize= parameters_opt)

    volume_check_exit = IntParameter(10, 45, default=19, space="sell", optimize= parameters_opt)
    volume_check_exit_s = IntParameter(15, 45, default=41, space="sell", optimize= parameters_opt)


    protect_optimize = True
    # cooldown_lookback = IntParameter(1, 40, default=4, space="protection", optimize=protect_optimize)
    max_drawdown_lookback = IntParameter(1, 50, default=2, space="protection", optimize=protect_optimize)
    max_drawdown_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    max_drawdown_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)
    max_allowed_drawdown = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space="protection", optimize=protect_optimize)
    stoploss_guard_lookback = IntParameter(1, 50, default=8, space="protection", optimize=protect_optimize)
    stoploss_guard_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    stoploss_guard_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)

    plot_config = {
        'main_plot': {
            'ema_l': {'color': '#c85656'},
            'ema_s': {'color': '#FF8C00'},
            'upper_band_l': {'color': '#FF6EB4'},
            'middle_band_l': {'color': '#FF4500'},
            'upper_band_s': {'color': '#CDB38B'},
            'middle_band_s': {'color': '#CD9B9B'}
        },
        'subplots': {
            'rsi': {
                'RSI_l': {'color': '#BFBFBF'},
                'RSI_s': {'color': '#BDB76B'}

            },
            'atr': {
                'atr': {'color': '#75fc40'}
            },
            'adx': {
                'adx': {'color': '#54e5f8'}
            }
        }
    }

    @property
    def protections(self):
        return [
            # # 设置冷却期为 18 根 K 线，即在交易完成后的 18 根 K 线内不允许再次开仓
            # ##FIXME add cooldown
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

    # ema_long = IntParameter(50, 250, default=100, space="buy")
    # ema_short = IntParameter(50, 250, default=100, space="buy")

    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Optional order type mapping.
    order_types = {
        # FIXME 设置市价单
        #'entry': 'limit',
        #'exit': 'limit',
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        # 做多趋势
        L_optimize_trend_alert  = trade_signal(dataframe=dataframe, rsi_tp= self.rsi_tp_l.value,
                                               bb_tp = self.bb_tp_l.value, suffix="l")
        dataframe['trend_l'] = L_optimize_trend_alert['trend']

        # 做空趋势
        S_optimize_trend_alert  = trade_signal(dataframe=dataframe, rsi_tp= self.rsi_tp_s.value,
                                               bb_tp = self.bb_tp_s.value, suffix="s")
        dataframe['trend_s'] = S_optimize_trend_alert['trend']

        # ADX 用于量化市场趋势的强弱程度，而非趋势方向
        dataframe['adx'] = ta.ADX(dataframe)

        # ATR 计算价格波动的平均范围，反映资产价格的波动强度
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=20)

        # EMA
        dataframe['ema_l'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_period_l_exit.value))
        dataframe['ema_s'] = ta.EMA(dataframe['close'], timeperiod=int(self.ema_period_s_exit.value))

        # Volume Weighted
        # 前一个周期的成交量均值
        dataframe['volume_mean'] = dataframe['volume'].rolling(self.volume_check.value).mean().shift(1)
        dataframe['volume_mean_exit'] = dataframe['volume'].rolling(self.volume_check_exit.value).mean().shift(1)

        dataframe['volume_mean_s'] = dataframe['volume'].rolling(self.volume_check_s.value).mean().shift(1)
        dataframe['volume_mean_exit_s'] = dataframe['volume'].rolling(self.volume_check_exit_s.value).mean().shift(1)

        # BBAND
        dataframe['upper_band_l'], dataframe['middle_band_l'], dataframe['lower_band_l'] = ta.BBANDS(dataframe['close'], timeperiod=int(self.bb_tp_l.value))
        dataframe['upper_band_s'], dataframe['middle_band_s'], dataframe['lower_band_s'] = ta.BBANDS(dataframe['close'], timeperiod=int(self.bb_tp_s.value))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ## FIXME
        # 1. 去掉entry_long_1，entry_short_1，效果变差
        # 2. 增加价格和布林带的对比

        dataframe.loc[
            (
                    # 5.7 ~ 6.5
                    (dataframe['adx'] > self.adx_long_min_1.value) &
                    (dataframe['adx'] < self.adx_long_max_1.value) &
                    (dataframe['close'] > dataframe['middle_band_l']) &  # FIXME
                    (dataframe['close'] < dataframe['upper_band_l']) # FIXME
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long_1')

        dataframe.loc[
            (
                    (dataframe['adx'] > self.adx_long_min_2.value) &  # trend strength confirmation
                    (dataframe['adx'] < self.adx_long_max_2.value) &  # trend strength confirmation
                    (dataframe['trend_l'] == 1) &
                    (dataframe['volume'] > dataframe['volume_mean'])
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long_2')

        # #9.9 ~ 21.4
        dataframe.loc[
            (
                    (dataframe['adx'] > self.adx_short_min_1.value) & # trend strength confirmation
                    (dataframe['adx'] < self.adx_short_max_1.value) &
                    (dataframe['close'] > dataframe['middle_band_s']) & # FIXME
                    (dataframe['close'] < dataframe['upper_band_s']) # FIXME
            ),
            ['enter_short','enter_tag']] = (1, 'entry_short_1')

        # 30.3 ~ 50
        dataframe.loc[
            (
                    (dataframe['adx'] > self.adx_short_min_2.value) &  # trend strength confirmation
                    (dataframe['adx'] < self.adx_short_max_2.value) &  # trend strength confirmation
                    (dataframe['trend_s'] == -1) &
                    (dataframe['volume'] > dataframe['volume_mean_s']) # volume weighted indicator
            ),
            ['enter_short', 'enter_tag']] = (1, 'entry_short_2')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 当持有多仓时，若收盘价低于长期 EMA 减去一定倍数的 ATR，则平多仓
        # 当持有空仓时，若收盘价高于长期 EMA 加上一定倍数的 ATR，则平空仓。

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        exit_long = (
                # (dataframe['close'] < dataframe['low'].shift(self.sell_shift.value)) &
                (dataframe['close'] < (dataframe['ema_l'] - (self.atr_long_mul.value * dataframe['atr']))) &
                (dataframe['volume'] > dataframe['volume_mean_exit'])
        )

        conditions_long.append(exit_long)
        dataframe.loc[exit_long, 'exit_tag'] += 'exit_long'

        exit_short = (
                # (dataframe['close'] > dataframe['high'].shift(self.sell_shift_short.value)) &
                (dataframe['close'] > (dataframe['ema_s'] + (self.atr_short_mul.value * dataframe['atr']))) &
                (dataframe['volume'] > dataframe['volume_mean_exit_s'])
        )

        conditions_short.append(exit_short)
        dataframe.loc[exit_short, 'exit_tag'] += 'exit_short'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value
