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


# custom indicators
# ##############################################################################################################################################################################################


# ##############################################################################################################################################################################################

class QuanStra_02(IStrategy):


    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy.
    # minimal_roi = {
    #     "0": 0.20,
    #     "416": 0.14,
    #     "933": 0.073,
    #     "1982": 0
    # }

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.1
    ## FIXME 2-1
    ##stoploss = -0.05

    # # 禁用固定止损，启用自定义止损
    # use_custom_stoploss = False

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
    #leverage_optimize = True
    #leverage_num = IntParameter(low=1, high=5, default=1, space='buy', optimize=leverage_optimize)
    leverage_num = 1

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

    stoch_oversold_thd = IntParameter(20, 30, default=20, space="buy")
    stoch_overbuy_thd = IntParameter(70, 80, default=80, space="buy")
    stoch_long_thd = IntParameter(40, 60, default=50, space="buy")
    stoch_short_thd = IntParameter(40, 60, default=50, space="buy")

    stoch_rolling_window = IntParameter(2, 6, default=4, space="buy")
    rsi_rolling_window = IntParameter(2, 6, default=4, space="buy")
    macd_rolling_window = IntParameter(2, 6, default=4, space="buy")


    rsi_long_thd = IntParameter(40, 60, default=50, space='buy')
    rsi_short_thd = IntParameter(40, 60, default=50, space='buy')

    volume_length = IntParameter(20, 30, default=20, space='buy')

    resistance_window = IntParameter(3, 6, default=5, space='buy') # * 20 = [60, 80, 100, 120]
    support_window = IntParameter(3, 6, default=5, space='buy') # * 20 = [60, 80, 100, 120]

    adx_thd = IntParameter(15, 30, default=10, space='buy') #[15, 20, 25]

    atr_mult = DecimalParameter(2.0, 3.5, default=2.0, decimals=1, space="sell")
    risk_ratio = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="sell")

    super_trend_period = IntParameter(7, 14, default=10, space="buy")
    super_trend_multiplier = IntParameter(2, 4, default=3, space="buy")
    ema_period = IntParameter(20, 100, default=50, space="buy")

    ema_fast = IntParameter(5, 12, default=8, space='buy')
    ema_slow = IntParameter(18, 30, default=21, space='buy')

    ## sell_long_close_thd = DecimalParameter(0.0, 0.4, default=0.2, decimals=1, space="sell")

    # max_allowed_drawdown = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space="protection",
    #                                         optimize=protect_optimize)
    plot_config = {
        'main_plot': {
            'support': {'color': '#75fc40'},
            'resistance': {'color': '#54e5f8'},
            'sar': {'color': '#FF9912'},
            'sma_15': {'color': '#A020F0'},
            'supert_s': {'color': '#A0522D'},
            'supert_l': {'color': '#A020F0'},
            'supert': {'color': '#6B8E23'},
            'ema_fast': {'color': '#03A89E'},
            'ema_slow': {'color': '#6B8E23'},
        },
        'subplots': {
            'rsi': {
                'rsi': {'color': '#FF9912'},
                'rsi_fast': {'color': '#03A89E'},
                'rsi_slow': {'color': '#6B8E23'},
            },
            'macd': {
                'macd': {'color': '#75fc40'},
                'macd_signal': {'color': '#54e5f8'}
            },
            'stoch': {
                'slowk': {'color': '#75fc40'},
                'slowd': {'color': '#54e5f8'}
            },
            'conds': {
                'stoch_trend': {'color': '#75fc40'},
                'rsi_trend': {'color': '#54e5f8'},
                'macd_trend': {'color': '#DA70D6'},
                'trend': {'color': '#7FFF00'},
                'adx': {'color': '#A0522D'},
                'mom': {'color': '#A020F0'},
                'supert_d': {'color': '#A020F0'},
                'ema_fast_slope': {'color': '#03A89E'},
                'ema_slow_slope': {'color': '#6B8E23'},
                # 'dmi_width_ratio': {'color': '#698B22'}
            },

            'dmi': {
                # 'dmi_trend': {'color': '#75fc40'},
                # 'dmi_cross': {'color': '#54e5f8'},
                # 'dmi_cross_sum5': {'color': '#DA70D6'},
                # 'dmi_cross_sum3': {'color': '#7FFF00'},
                'dmi_plus': {'color': '#A0522D'},
                'dmi_minus': {'color': '#A020F0'},
                # 'dmi_plus_minus_ratio': {'color': '#00EEEE'},
                # 'dmi_minus_plus_ratio': {'color': '#228B22'},
                # 'dmi_plus_slope': {'color': '#7D26CD'},
                # 'dmi_plus_minus_ratio_mean': {'color': '#6B8E23'},
                # 'dmi_minus_plus_ratio_mean': {'color': '#A020F0'},
                # 'dmi_width': {'color': '#03A89E'},
                # 'dmi_width_mean': {'color': '#6B8E23'},
                # 'dmi_width_ratio': {'color': '#698B22'}
            },
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

        ################### 定义基础指标 ##########################

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=int(self.ema_fast.value))
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=int(self.ema_slow.value))

        # 计算EMA斜率（最近3根K线的平均变化率）
        dataframe['ema_fast_slope'] = (
                                              dataframe['ema_fast'].diff(1) +
                                              dataframe['ema_fast'].diff(2) +
                                              dataframe['ema_fast'].diff(3)
                                      ) / 3
        dataframe['ema_slow_slope'] = (
                                              dataframe['ema_slow'].diff(1) +
                                              dataframe['ema_slow'].diff(2) +
                                              dataframe['ema_slow'].diff(3)
                                      ) / 3

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        stoch_slow = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch_slow['slowk']
        dataframe['slowd'] = stoch_slow['slowd']

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        ## DMI 相关指标
        dmi = fta.DMI(dataframe, period=14)
        dataframe['dmi_plus'] = dmi['DI+']
        dataframe['dmi_minus'] = dmi['DI-']

        # SAR
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)

        dataframe['volume_mean'] = dataframe['volume'].rolling(self.volume_length.value).mean().shift(1)

        # 计算过去 50 根 K 线的最高价作为阻力位
        dataframe['resistance'] = dataframe['high'].rolling(window=20*self.resistance_window.value).max().shift(1)

        # 计算过去 50 根 K 线的最低价作为支撑位
        dataframe['support'] = dataframe['low'].rolling(window=20*self.support_window.value).min().shift(1)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        # 计算超级趋势指标
        super_trend = pta.supertrend(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            length=self.super_trend_period.value,
            multiplier=self.super_trend_multiplier.value
        )

        dataframe['supert_d'] = super_trend[f'SUPERTd_{self.super_trend_period.value}_{self.super_trend_multiplier.value}.0']
        dataframe['supert_s'] = super_trend[f'SUPERTs_{self.super_trend_period.value}_{self.super_trend_multiplier.value}.0']
        dataframe['supert_l'] = super_trend[f'SUPERTl_{self.super_trend_period.value}_{self.super_trend_multiplier.value}.0']
        dataframe['supert'] = super_trend[f'SUPERT_{self.super_trend_period.value}_{self.super_trend_multiplier.value}.0']

        ################### 计算趋势指标 ##########################

        stoch_long = (
            (qtpylib.crossed_above(dataframe['slowk'], self.stoch_oversold_thd.value)) # 20-30
            & (dataframe['slowk'] < self.stoch_long_thd.value) # 40-50
        )
        rsi_long = (dataframe['rsi'] > self.rsi_long_thd.value)
        macd_long = (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macd_signal'])
        )

        stoch_short = (
                (qtpylib.crossed_below(dataframe['slowk'], self.stoch_overbuy_thd.value)) # 70-80
                & (dataframe['slowk'] > self.stoch_short_thd.value) # 40-60
        )
        rsi_short = (dataframe['rsi'] < self.rsi_short_thd.value)
        macd_short = (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macd_signal'])
        )

        dataframe['stoch_trend'] = np.where(stoch_long, 1, np.where(stoch_short, -1, 0))
        dataframe['rsi_trend'] = np.where(rsi_long, 1, np.where(rsi_short, -1, 0))
        dataframe['macd_trend'] = np.where(macd_long, 1, np.where(macd_short, -1, 0))

        long_trend = (
                (dataframe['stoch_trend'].rolling(window=self.stoch_rolling_window.value).max() == 1) &
                (dataframe['rsi_trend'].rolling(window=self.rsi_rolling_window.value).max() == 1) &
                (dataframe['macd_trend'].rolling(window=self.macd_rolling_window.value).max() == 1)
        )

        short_trend = (
                (dataframe['stoch_trend'].rolling(window=self.stoch_rolling_window.value).min() == -1) &
                (dataframe['rsi_trend'].rolling(window=self.rsi_rolling_window.value).min() == -1) &
                (dataframe['macd_trend'].rolling(window=self.macd_rolling_window.value).min() == -1)
        )

        dataframe["trend"] = np.where(long_trend, 1, np.where(short_trend, -1, 0))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ########################### long below #####################################

        dataframe.loc[
            (
                (dataframe["trend"] == 1)
                & (dataframe['volume'] > 0)
                ## & (dataframe['supert_d'] > 0) # FIXME 2-1 增加supertrend过滤
                ## & (dataframe['high'] > dataframe['supert']) # FIXME 2-2 增加supert过滤
                # & (dataframe['ema_fast_slope'] > 0) # FIXME 2-4
                # & (dataframe['ema_slow_slope'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long')

        ########################### short below #####################################

        dataframe.loc[
            (
                (dataframe["trend"] == -1)
                & (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'entry_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 此处主要用于兼容性设置，实际退出逻辑由自定义方法处理
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe

    def custom_stoploss(self, current_time: datetime, current_rate: float,
                         current_profit: float, trade: 'Trade', **kwargs) -> float:
        # 获取交易开仓时的K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        current_candle = dataframe.iloc[-1].squeeze()
        last_candle = dataframe.iloc[-2].squeeze()

        # 如果止损值已缓存，直接使用
        if hasattr(trade, 'cached_entry_atr'):
            atr_value = trade.cached_entry_atr
        else:
            # 确保时间列为 UTC 时区
            dataframe = dataframe.assign(
                date_utc=pd.to_datetime(dataframe["date"], utc=True)
            )

            # 找到入场时间对应的 K 线索引
            open_date = trade.open_date_utc


            entry_idx = dataframe["date_utc"].searchsorted(open_date, side="right") - 1
            entry_idx = max(0, entry_idx)  # 防止负索引
            entry_candle = dataframe.iloc[entry_idx]

            # 使用入场时的ATR值
            atr_value = entry_candle['atr']

            # 将止损价缓存到 trade 对象中
            trade.cached_entry_atr = atr_value

        # 计算动态止损
        if trade.is_short:
            # 空头止损 = 入场价 + ATR倍数
            stop_loss_price = trade.open_rate + (self.atr_mult.value * atr_value)
        else:
            # 多头止损 = 入场价 - ATR倍数
            stop_loss_price = trade.open_rate - (self.atr_mult.value * atr_value)

        return stop_loss_price

    def custom_exit(self, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):

        # 获取交易开仓时的K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        current_candle = dataframe.iloc[-1].squeeze()
        last_candle = dataframe.iloc[-2].squeeze()

        # 如果止损值已缓存，直接使用
        if hasattr(trade, 'cached_entry_atr'):
            atr_value = trade.cached_entry_atr
        else:
            # 确保时间列为 UTC 时区
            dataframe = dataframe.assign(
                date_utc=pd.to_datetime(dataframe["date"], utc=True)
            )

            # 找到入场时间对应的 K 线索引
            open_date = trade.open_date_utc

            entry_idx = dataframe["date_utc"].searchsorted(open_date, side="right") - 1
            entry_idx = max(0, entry_idx)  # 防止负索引
            entry_candle = dataframe.iloc[entry_idx]

            # 使用入场时的ATR值
            atr_value = entry_candle['atr']

            # 将止损价缓存到 trade 对象中
            trade.cached_entry_atr = atr_value

        # 计算止盈价格
        if trade.is_short:
            take_profit_price = trade.open_rate - (self.atr_mult.value * self.risk_ratio.value  * atr_value)
            if current_rate <= take_profit_price:
                return 'take_profit_short'
        else:
            # FIXME 2-3
            current_close = current_candle['close']
            current_sar = current_candle['sar']
            previous_close = last_candle['close']
            previous_sar = last_candle['sar']
            current_dmi_ratio = (current_candle['dmi_plus'] / current_candle['dmi_minus'] - 1)
            previous_dmi_ratio = (last_candle['dmi_plus'] / last_candle['dmi_minus'] - 1)

            # if (
            #         (current_close < current_sar)
            #         & (previous_close > previous_sar)
            #         & (current_candle['dmi_minus'] > current_candle['dmi_plus'])
            # ):
            #     return 'sell_long_sar_1'
            ###FIXME 2-5
            # if (
            #         (current_close < current_sar)
            #         & (previous_close > previous_sar)
            #         & (current_dmi_ratio < previous_dmi_ratio) # dmi+ 和dmi- 的差距缩小
            # ):
            #     return 'sell_long_sar_2'

            take_profit_price = trade.open_rate + (self.atr_mult.value * self.risk_ratio.value * atr_value)
            if current_rate >= take_profit_price:
                return 'take_profit_long'

        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        #return self.leverage_num.value
        return self.leverage_num




