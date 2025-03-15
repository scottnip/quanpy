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

def EWO(dataframe, ema_length=5, ema2_length=30):
    df = dataframe.copy()
    # 短期EMA
    ema1 = ta.EMA(df, timeperiod=ema_length)
    # 长期EMA
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

# ##############################################################################################################################################################################################

class RSI_BB_MACD_T5(IStrategy):


    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.184,
        "416": 0.14,
        "933": 0.073,
        "1982": 0
    }

    stoploss = -0.15

    # 禁用固定止损，启用自定义止损
    use_custom_stoploss = False
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.022
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
    #leverage_optimize = False
    #leverage_num = IntParameter(low=1, high=5, default=1, space='buy', optimize=leverage_optimize)
    leverage_num = 1

    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(-20.0, -8.0, default=-10, space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=5, space='buy', optimize=True)

    # Strategy parameters
    parameters_opt = True

    adx_long_max_1 = DecimalParameter(6.1, 10.0, default=6.5, decimals = 1, space="buy", optimize = parameters_opt)
    adx_long_min_1 = DecimalParameter(4.0, 6.0, default=5.7, decimals = 1, space="buy", optimize = parameters_opt)

    adx_long_max_2 = DecimalParameter(24.9, 60.0, default=50.7, decimals = 1, space="buy", optimize = parameters_opt)
    adx_long_min_2 = DecimalParameter(18.5, 21.0, default=20.9, decimals = 1, space="buy", optimize = parameters_opt)

    adx_short_max_2 = DecimalParameter(30.6, 55.0, default=50.8, decimals = 1, space="buy", optimize = parameters_opt)
    adx_short_min_2 = DecimalParameter(25.0, 30.5, default=30.3, decimals = 1, space="buy", optimize = parameters_opt)

    adx_es1_min = DecimalParameter(15.0, 30.0, default=20.0, decimals = 1, space="buy", optimize = parameters_opt)
    adx_es1_max = DecimalParameter(40.0, 60.0, default=50.0, decimals = 1, space="buy", optimize = parameters_opt)

    bb_tp_l = IntParameter(15, 35, default=16, space="buy", optimize= parameters_opt)
    bb_tp_s = IntParameter(15, 35, default=20, space="buy", optimize= parameters_opt)

    rsi_tp_l = IntParameter(10, 25, default=22, space="buy", optimize= parameters_opt)
    rsi_tp_s = IntParameter(10, 25, default=17, space="buy", optimize= parameters_opt)

    rsi_fast_el3_thd = IntParameter(60, 90, default=80, space="buy", optimize= parameters_opt)
    rsi_trend_short_thd2 = IntParameter(20, 35, default=30, space="buy", optimize= parameters_opt)
    rsi_trend_short_thd1 = IntParameter(35, 50, default=40, space="buy", optimize= parameters_opt)
    rsi_trend_strong_thd2 = IntParameter(65, 80, default=70, space="buy", optimize=parameters_opt)
    rsi_trend_strong_thd1 = IntParameter(50, 65, default=60, space="buy", optimize=parameters_opt)

    fastk_el3_thd = IntParameter(60, 90, default=80, space="buy", optimize= parameters_opt)
    fastk_es1_thd = IntParameter(0, 30, default=10, space="buy", optimize= parameters_opt)
    fastk_fastd_el3_ratio = DecimalParameter(0.9, 2.0, default=1.0, decimals = 1, space="buy", optimize = parameters_opt)
    fastk_fastd_es1_ratio = DecimalParameter(0.5, 1.1, default=1.0, decimals = 1, space="buy", optimize = parameters_opt)

    mom_fast_accel_ratio_el3_thd = DecimalParameter(0.0, 4.0, default=0.0, decimals = 1, space="buy", optimize = parameters_opt)
    mom_fast_accel_ratio_es1_thd = DecimalParameter(-4.0, 0.0, default=0.0, decimals = 1, space="buy", optimize = parameters_opt)
    mom_fast_accel_ratio_xs_thd = DecimalParameter(0.0, 4.0, default=0.0, decimals = 1, space="sell", optimize = parameters_opt)
    mom_fast_accel_xs_thd = DecimalParameter(0.0, 1.0, default=0.2, decimals = 1, space="sell", optimize = parameters_opt)

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

    fast_ema = IntParameter(10, 30, default=20, space="buy", optimize=protect_optimize)
    slow_ema = IntParameter(40, 60, default=50, space="buy", optimize=protect_optimize)
    atr_mult_long = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=protect_optimize)
    atr_mult_short = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=protect_optimize)
    atr_multiplier = DecimalParameter(2.5, 3.5, default=2.5, space='sell', optimize=protect_optimize)

    price_volatility_el3_thd = DecimalParameter(0.1, 2.0, default=1.5, decimals=1, space="buy", optimize=protect_optimize)

    plot_config = {
        'main_plot': {
            'fast_ema': {'color': '#c85656'},
            'slow_ema': {'color': '#FF8C00'},
            'sma_15': {'color': '#FF00FF'},

            'upper_band_l': {'color': '#FF6EB4'},
            'middle_band_l': {'color': '#FF4500'},
            'lower_band_l': {'color': '#B2DFEE'},
            'upper_band_s': {'color': '#CDB38B'},
            'middle_band_s': {'color': '#CD9B9B'},
            'lower_band_s': {'color': '#B23AEE'},
            #'price_low': {'color': '#F4A460'}
        },
        'subplots': {
            # 'RSI': {
            #     'RSI_l': {'color': '#BFBFBF'},
            #     'RSI_s': {'color': '#BDB76B'}
            # },
            'rsi': {
                'rsi': {'color': '#FF9912'},
                'rsi_slow': {'color': '#E3CF57'},
                'rsi_fast': {'color': '#FFD700'},
                'rsi_fast_ratio': {'color': '#C76114'},
            },
            'conds': {
                'atr': {'color': '#75fc40'},
                'adx': {'color': '#54e5f8'},
                'cti': {'color': '#DA70D6'},
                'cci': {'color': '#7FFF00'},
                'mfi': {'color': '#082E54'},
                'ewo': {'color': '#C76114'},
                'dmi_plus': {'color': '#A0522D'},
                'dmi_minus': {'color': '#A020F0'},
                'close_upper_band_pct_l': {'color': '#E3170D'}
            },
            'bandwidth': {
                'bandwidth_l': {'color': '#00FFFF'},
                'bandwidth_s': {'color': '#7FFF00'},
                'bandwidth_s_thd': {'color': '#228B22'}

            },
            'trend_strength': {
                'trend_short_strength': {'color': '#00EE76'},
                'trend_long_strength': {'color': '#DDA0DD'}
            },
            'fastk': {
                'fastk': {'color': '#e830dc'},
                'fastd': {'color': '#698B22'}
            },
            'price_change': {
                'price_change_rate': {'color': '#03A89E'},
                'price_volatility': {'color': '#6B8E23'}
            },
            'mom': {
                #'mom': {'color': '#7D26CD'},
                #'mom_accel': {'color': '#7CFC00'},
                #'mom_accel_ratio': {'color': '#7CCD7C'},
                'mom_fast': {'color': '#00FF7F'},
                'mom_fast_accel': {'color': '#00EEEE'},
                'mom_fast_accel_ratio': {'color': '#008B8B'},
                'mom_fast_accel_ratio_accel': {'color': '#A020F0'}
            }
        }
    }

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
        dataframe['atr_trailing'] = dataframe['close'] - (dataframe['atr'] * self.atr_multiplier.value)

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
        dataframe['bandwidth_l'] = (dataframe['upper_band_l'] - dataframe['lower_band_l']) / dataframe['middle_band_l'] * 100
        dataframe['bandwidth_s'] = (dataframe['upper_band_s'] - dataframe['lower_band_s']) / dataframe['middle_band_s'] * 100
        dataframe['bandwidth_s_thd'] = dataframe['bandwidth_s'].rolling(100).quantile(0.2)

        # close价格在BBAND中的位置
        dataframe['close_upper_band_pct_l'] = (dataframe['close'] - dataframe['middle_band_l']) / (dataframe['upper_band_l'] - dataframe['middle_band_l'])

        # rsi
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        # rsi_fast 相对于rsi的变化率。百分比
        dataframe['rsi_fast_ratio'] = (dataframe['rsi_fast'] / dataframe['rsi'] - 1) *100

        # 背离检测（最近5周期最低点）
        dataframe['price_low'] = dataframe['low'].rolling(5).min()
        dataframe['rsi_low'] = dataframe['rsi'].rolling(5).min()

        #价格波动
        dataframe['price_change_rate'] = dataframe['close'].pct_change() * 100
        dataframe['price_volatility'] = dataframe['price_change_rate'].abs().rolling(5).mean()

        # SMA
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['fast_ema'] = ta.EMA(dataframe, timeperiod=int(self.fast_ema.value))
        dataframe['slow_ema'] = ta.EMA(dataframe, timeperiod=int(self.slow_ema.value))

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['fastd'] = stoch_fast['fastd']

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dmi = fta.DMI(dataframe, period=14)
        dataframe['dmi_plus'] = dmi['DI+']
        dataframe['dmi_minus'] = dmi['DI-']

        # 计算short趋势强度评分
        dataframe['trend_short_strength'] = np.where(
            (dataframe['close'] < dataframe['fast_ema']) &
            (dataframe['rsi'] < self.rsi_trend_short_thd2.value), # 33
            2,  # 强势下跌趋势
            np.where(
                (dataframe['close'] < dataframe['fast_ema']) &
                (dataframe['rsi'] < self.rsi_trend_short_thd1.value), # 47
                1,  # 普通下跌趋势
                0  # 无趋势
            )
        )

        # 计算long趋势强度评分
        dataframe['trend_long_strength'] = np.where(
            (dataframe['close'] > dataframe['fast_ema']) &  # 收盘价在快速EMA上方
            (dataframe['rsi'] > self.rsi_trend_strong_thd2.value),  # 71 RSI超买区域
            2,  # 强势上涨趋势
            np.where(
                (dataframe['close'] > dataframe['fast_ema']) &  # 收盘价在快速EMA上方
                (dataframe['rsi'] > self.rsi_trend_strong_thd1.value),  # 65 RSI中性偏强区域
                1,  # 普通上涨趋势
                0  # 无趋势
            )
        )

        # 一阶动量：价格变化速度
        dataframe['mom'] = ta.MOM(dataframe, timeperiod = 14 )
        # 当前价 - 14周期前价 # 二阶动量：动量变化率（加速度）
        dataframe ['mom_accel'] = ta.MOM(dataframe['mom'], timeperiod=14) # mom(t) - mom(t-14)
        dataframe['mom_accel_ratio'] = dataframe['mom_accel'] / dataframe['atr']

        # 一阶动量：价格变化速度
        dataframe['mom_fast'] = ta.MOM(dataframe, timeperiod=3)
        # 当前价 - 3周期前价 # 二阶动量：动量变化率（加速度）
        dataframe ['mom_fast_accel'] = ta.MOM(dataframe['mom_fast'], timeperiod=3) # mom(t) - mom(t-3)
        dataframe['mom_fast_accel_ratio'] = dataframe['mom_fast_accel'] / dataframe['atr']
        dataframe ['mom_fast_accel_ratio_accel'] = ta.MOM(dataframe['mom_fast_accel_ratio'], timeperiod=3)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ## FIXME 6
        # 1. 去掉entry_long_1，entry_short_1，效果变差
        # 2. 增加价格和布林带的对比

        dataframe.loc[
            (
                    # 5.7 ~ 6.5
                    (dataframe['adx'] > self.adx_long_min_1.value) & # 4.7
                    (dataframe['adx'] < self.adx_long_max_1.value) & # 7.7
                    (dataframe['close'] > dataframe['middle_band_l']) &  # FIXME 6
                    (dataframe['close'] < dataframe['upper_band_l']) # FIXME 6
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long_1')

        dataframe.loc[
            (
                    (dataframe['adx'] > self.adx_long_min_2.value)  # 21
                    & (dataframe['adx'] < self.adx_long_max_2.value)  # 44

                    & (dataframe['close'] > dataframe['middle_band_l'])
                    #### & (dataframe['close'] < dataframe['upper_band_l']) # FIXME 5-41

                    & (dataframe['trend_long_strength'] > 0)

                    & (dataframe['rsi_fast'] < self.rsi_fast_el3_thd.value) # 65 FIXME 4-9

                    # # FIXME 5-40
                    & (
                            (
                                    (dataframe['rsi_fast'] > dataframe['rsi'])
                                    & (dataframe['rsi_fast'].shift(1) < dataframe['rsi'].shift(1))
                            )
                            # # # FIXME 5-42
                            # | (dataframe['fastk'] > dataframe['fastd'])
                    )
                    & (dataframe['price_volatility'] > self.price_volatility_el3_thd.value)

            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long_3')

        ################################################################

        # #9.9 ~ 21.4
        base_trend = (
                (dataframe['adx'] > self.adx_es1_min.value) &  # 16
                (dataframe['adx'] < self.adx_es1_max.value) &  # 51.5 FIXME 4-20

                (
                        (
                                (dataframe['close'] < dataframe['middle_band_s']) &
                                (dataframe['close'] > dataframe['lower_band_s'])
                        )
                        # FIXME 5-21 close在middle上方也考虑
                        | (
                                (dataframe['close'] < dataframe['upper_band_s']) &
                                (dataframe['mom_fast_accel_ratio'] < 0.0)
                        )

                )

                #& (dataframe['close'].shift(1) < dataframe['middle_band_s'].shift(1))  # FIXME 5-16
                #& (dataframe['close'].shift(1) > dataframe['lower_band_s'].shift(1)) # FIXME 5-16

                & (dataframe['close'] < dataframe['sma_15'])  # FIXME 5-30
                & (dataframe['trend_short_strength'] > 0)  #0 FIXME 5-17
                & (dataframe['rsi_fast'] < dataframe["rsi"])  # FIXME 4-21
                & (dataframe['bandwidth_s'] > dataframe['bandwidth_s_thd']) # FIXME 5-15
                #& (dataframe['price_volatility'] > 1) # FIXME 5-24
                & (dataframe['price_change_rate'] < 1) # FIXME 5-23
                & (dataframe['mom_fast_accel_ratio'] < 1)  # FIXME 5-24
                & (dataframe['fastk'] > self.fastk_es1_thd.value) # 7 FIXME 5-25 防止超卖，价格回弹
                & (dataframe['fastk'] < dataframe["fastd"] * self.fastk_fastd_es1_ratio.value) # 1.0 FIXME 5-28
        )
        avoid_reversal = (
                (dataframe['bandwidth_s'] > dataframe['bandwidth_s'].rolling(100).quantile(0.2)) & # 波动未收缩
                ~ (
                    # 排除底背离
                    (dataframe['price_low'] < dataframe['price_low'].shift(1)) &
                    (dataframe['rsi_low'] > dataframe['rsi_low'].shift(1))
                )
        )
        dataframe.loc[
            base_trend
            # & avoid_reversal # FIXME 5-13
            ,
            ['enter_short','enter_tag']] = (1, 'entry_short_1')

        # 30.3 ~ 50
        dataframe.loc[
            (
                    (dataframe['adx'] > self.adx_short_min_2.value) &  # 30.3 trend strength confirmation
                    (dataframe['adx'] < self.adx_short_max_2.value) &  # 50.8 trend strength confirmation
                    (dataframe['trend_s'] == -1) &
                    (dataframe['volume'] > dataframe['volume_mean_s']) # volume weighted indicator
                    & (dataframe['trend_short_strength'] > 0)  # FIXME 4-4
                    & (dataframe['trend_short_strength'].shift(1) > 0)  # FIXME 4-5
            ),
            ['enter_short', 'enter_tag']] = (1, 'entry_short_2')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 当持有多仓时，若收盘价低于长期 EMA 减去一定倍数的 ATR，则平多仓
        # 当持有空仓时，若收盘价高于长期 EMA 加上一定倍数的 ATR，则平空仓。

        # 动态止损逻辑（多头：EMA下方，空头：EMA上方）
        dataframe['trailing_stop_long'] = dataframe['fast_ema'] - self.atr_mult_long.value * dataframe['atr']
        dataframe['trailing_stop_short'] = dataframe['fast_ema'] + self.atr_mult_short.value * dataframe['atr']

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        exit_long = (
           (
                (
                        (dataframe['close'] < (dataframe['ema_l'] - (self.atr_long_mul.value * dataframe['atr']))) &
                        (dataframe['volume'] > dataframe['volume_mean_exit'])
                )
                | (
                        (dataframe['close'] < dataframe['lower_band_l']) &
                        (dataframe['volume'] > 0)
                )
                | (
                    (dataframe['mom_fast'] < 0) &
                    (dataframe['mom_fast_accel'] < 0)
                )
                # FIXME 5-43
                | (
                      (dataframe['mom_fast_accel_ratio_accel'] < 0) &
                      (dataframe['mom_fast_accel_ratio_accel'].shift(1) < 0)
                )

           )
           & (dataframe['volume'] > 0)
           # #  # FIXME 5-51
           # & (
           #     (dataframe['rsi_fast'] > dataframe['rsi_slow'])
           # )
           # & (
           #      (dataframe['price_change_rate'] < -0.1) &
           #      (dataframe['price_change_rate'].shift(1) > 0.1)
           # )

        )


        conditions_long.append(exit_long)
        dataframe.loc[exit_long, 'exit_tag'] += 'exit_long'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        exit_short_1 = (
                (dataframe['close'] > (dataframe['ema_s'] + (self.atr_short_mul.value * dataframe['atr']))) &
                (dataframe['volume'] > dataframe['volume_mean_exit_s'])
        )
        conditions_short.append(exit_short_1)
        dataframe.loc[exit_short_1, 'exit_tag'] += '/exit_short_1'

        exit_short_2 = (
                (
                        #(dataframe['close'] > dataframe['middle_band_s']) | # FIXME 5-34
                        #(dataframe['close'] < dataframe['upper_band_s']) | # FIXME 5-34
                        (
                                (dataframe["rsi_fast"] > dataframe["rsi"]) &
                                (dataframe["mom_fast_accel_ratio"] > self.mom_fast_accel_ratio_xs_thd.value)  # 3.0
                                #(dataframe["mom_fast_accel"] > 0.1)  # 1.6 # FIXME 5-0
                        )
                        # # # FIXME 5-3
                        | (
                                (dataframe["fastk"] > dataframe["fastd"]) &
                                (dataframe["mom_fast_accel"] > self.mom_fast_accel_xs_thd.value) # 0.2
                        )
                        # FIXME 5-26
                        #| (dataframe["price_change_rate"] > 1)
                        # FIXME 5-27
                        #| (dataframe['close'] > dataframe['sma_15'])
                ) & (dataframe['volume'] > 0)
                & (dataframe['fastk'] < 80) # FIXME 5-31 处于超买区域，价格会下降，short暂时不退出。
                # FIXME 5-32
                & (
                    (dataframe['price_change_rate'] > 0) &
                    (dataframe['price_change_rate'].shift(1) < 0)
                )
                # FIXME 5-33
                & (
                    (dataframe['rsi_fast'] < dataframe['rsi']) |
                    (dataframe['rsi_fast'].shift(1) < dataframe['rsi'].shift(1))
                )
        )

        conditions_short.append(exit_short_2)
        dataframe.loc[exit_short_2, 'exit_tag'] += '/exit_short_2'

        # # FIXME 5-35
        # exit_short_3= (
        #     (dataframe['price_change_rate'] > 1)
        #     & (dataframe['mom_fast_accel_ratio'] > 1)
        # )
        #
        # conditions_short.append(exit_short_3)
        # dataframe.loc[exit_short_3, 'exit_tag'] += '/exit_short_3'

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        # return self.leverage_num.value
        return self.leverage_num




