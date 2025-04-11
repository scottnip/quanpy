"""
TODO：
1. swing_high/swing_low: 未来函数问题，需要修正。

SMC
* FVG策略
https://www.xiaohongshu.com/explore/67e4d159000000001d025b21?app_platform=android&ignoreEngage=true&app_version=8.76.0&share_from_user_hidden=true&xsec_source=app_share&type=video&xsec_token=CBkX_D247Q-ZmwMzr1uKw6LvsTegjz2lqxU1xAZVFVEYY=&author_share=1&xhsshare=CopyLink&shareRedId=OD04ODRHPUo2NzUyOTgwNjc7OThJPTs_&apptime=1743910740&share_id=6358b6ff912f411287d18bf8a003e44c&share_channel=copy_link

1. fvg是未经测试过的。
2. 检测fvg内蜡烛的反应，要么在区域内收盘，要么朝着区域方向收盘。above_l_fvg
3. fvg突破前方支撑。

"""

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
from scipy . signal import argrelextrema

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


# ##############################################################################################################################################################################################

class SMC_FVG_Stra_01(IStrategy):
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

    ## 局部极值：窗口
    swing_order = IntParameter(1, 5, default=3, space="buy", optimize=optimize_label)

    ## long fvg 参数
    above_l_fvg_window = IntParameter(5, 20, default=10, space="buy", optimize=optimize_label)
    valid_l_fvg_window = IntParameter(5, 20, default=10, space="buy", optimize=optimize_label)

    ## short fvg 参数
    below_s_fvg_window = IntParameter(5, 20, default=10, space="buy", optimize=optimize_label)
    valid_s_fvg_window = IntParameter(5, 20, default=10, space="buy", optimize=optimize_label)

    ## 止盈参数
    take_profit_long_factor = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="sell", optimize=optimize_label)
    take_profit_short_factor = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="sell", optimize=optimize_label)

    plot_config = {
        'main_plot': {
            'swing_high': {'color': '#75fc40'},
            'swing_low': {'color': '#54e5f8'},
            'l_fvg_k_low': {'color': '#FF9912'},
            's_fvg_k_high': {'color': '#A0522D'},
            'ema_20': {'color': '#03A89E'},
            # 'long_pullback_max': {'color': '#6B8E23'},
            # 'long_pullback_min': {'color': '#7FFF00'},
            # 'short_pullback_max': {'color': '#DA70D6'},
            # 'short_pullback_min': {'color': '#03A89E'},
            # 'higher_trend_stoploss_price': {'color': '#A020F0'},
            # 'lower_trend_stoploss_price': {'color': '#698B22'},
        },
        'subplots': {
            'conds': {
                'l_fvg_high': {'color': '#A0522D'},
                'l_fvg_low': {'color': '#DA70D6'},
                'l_fvg': {'color': '#6B8E23'},
                'above_l_fvg': {'color': '#75fc40'},
                'last_l_fvg_high': {'color': '#228B22'},
                'last_l_fvg_low': {'color': '#A020F0'},
                'l_fvg_label': {'color': '#698B22'},
                'valid_l_fvg': {'color': '#54e5f8'},

                's_fvg_high': {'color': '#A0522D'},
                's_fvg_low': {'color': '#DA70D6'},
                's_fvg': {'color': '#6B8E23'},
                'below_s_fvg': {'color': '#75fc40'},
                'last_s_fvg_high': {'color': '#228B22'},
                'last_s_fvg_low': {'color': '#A020F0'},
                's_fvg_label': {'color': '#698B22'},
                'valid_s_fvg': {'color': '#54e5f8'},

                # 'long_pullback': {'color': '#7FFF00'},
                # 'short_pullback': {'color': '#A020F0'},
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

        ## 多头fvg
        dataframe['high_2'] = dataframe['high'].shift(2)
        dataframe['l_fvg'] = np.where(dataframe['low'] > dataframe['high_2'], 1, 0)
        dataframe['l_fvg_high'] = np.where(dataframe['low'] > dataframe['high_2'], dataframe['low'], np.nan)
        dataframe['l_fvg_low'] = np.where(dataframe['low'] > dataframe['high_2'], dataframe['high_2'], np.nan)

        # 形成多头fvg蜡烛的低点
        dataframe["l_fvg_k_low"] = dataframe["low"].shift(1).where(dataframe["l_fvg"] == 1)
        dataframe['l_fvg_k_low'] = dataframe['l_fvg_k_low'].ffill()

        # 空头fvg
        dataframe['low_2'] = dataframe['low'].shift(2)
        dataframe['s_fvg'] = np.where(dataframe['high'] < dataframe['low_2'], 1, 0)
        dataframe['s_fvg_high'] = np.where(dataframe['high'] < dataframe['low_2'], dataframe['low_2'], np.nan)
        dataframe['s_fvg_low'] = np.where(dataframe['high'] < dataframe['low_2'], dataframe['high'], np.nan)

        # 形成空头fvg蜡烛的高点
        dataframe["s_fvg_k_high"] = dataframe["high"].shift(1).where(dataframe["s_fvg"] == 1)
        dataframe['s_fvg_k_high'] = dataframe['s_fvg_k_high'].ffill()

        # 局部极值点
        dataframe['prev_high'] = dataframe['high'].shift(self.swing_order.value)
        dataframe['swing_high'] = dataframe.iloc[argrelextrema(dataframe['prev_high'].values, np.greater, order=self.swing_order.value)[0]]['prev_high']
        dataframe['swing_high'] = dataframe['swing_high'].ffill()
        dataframe['prev_low'] = dataframe['low'].shift(self.swing_order.value)
        dataframe['swing_low'] = dataframe.iloc[argrelextrema(dataframe['prev_low'].values, np.less, order=self.swing_order.value)[0]]['prev_low']
        dataframe['swing_low'] = dataframe['swing_low'].ffill()
        ################### 定义高级指标 ##########################

        ###################  long fvg
        dataframe = self.l_fvg_feature(dataframe)

        # 1. 当前fvg未被测试过
        l_fvg_valid = (dataframe['valid_l_fvg'].shift(1).rolling(self.valid_l_fvg_window.value).max() == 0)

        # 2. 之前K线没有收盘于fvg下方
        l_fvg_reaction = (
            # 当前K线收于fvg上方
            (dataframe['above_l_fvg'] == 1)
            # 之前K线没有收于fvg下方
            & (dataframe['above_l_fvg'].rolling(self.valid_l_fvg_window.value).min() >= 0)
        )

        # 3. fvg突破前方阻力。
        l_fvg_break = (dataframe['last_l_fvg_high'] > dataframe['swing_high'])

        dataframe['l_fvg_label'] = np.where(
            # 阴线
            (dataframe['open'] > dataframe['close'])
            & l_fvg_valid
            & l_fvg_reaction
            & l_fvg_break
            , 1, 0
        )

        ################ short fvg
        dataframe = self.s_fvg_feature(dataframe)

        # 1. 当前fvg未被测试过
        s_fvg_valid = (dataframe['valid_s_fvg'].shift(1).rolling(self.valid_s_fvg_window.value).max() == 0)

        # 2. 之前K线没有收盘于fvg上方
        s_fvg_reaction = (
                # 当前K线收于fvg下方
                (dataframe['below_s_fvg'] == 1)
                # 之前K线没有收于fvg上方
                & (dataframe['below_s_fvg'].rolling(self.valid_s_fvg_window.value).min() >= 0)
        )

        # 3. fvg突破前方支撑。
        s_fvg_break = (dataframe['last_s_fvg_low'] < dataframe['swing_low'])

        dataframe['s_fvg_label'] = np.where(
            # 阳线
            (dataframe['open'] < dataframe['close'])
            & s_fvg_valid
            & s_fvg_reaction
            & s_fvg_break
            , 1, 0
        )

        ################### 定义 其他指标 ###################################

        return dataframe


    def l_fvg_feature(self, dataframe: DataFrame) -> DataFrame:
        """检查当前K线收在 l_fvg 的上方"""

        # 使用rolling窗口寻找最近的有效FVG值
        dataframe['last_l_fvg_high'] = (
            dataframe['l_fvg_high']
            .shift(1)  # 排除当前K线
            .rolling(window=self.above_l_fvg_window.value, min_periods=1)
            .apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan)
        )

        dataframe['last_l_fvg_low'] = (
            dataframe['l_fvg_low']
            .shift(1)
            .rolling(window=self.above_l_fvg_window.value, min_periods=1)
            .apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan)
        )

        # 1. 如果K线的low 落在fvg区间，则为被测试过。
        dataframe['valid_l_fvg'] = np.where(
            (dataframe['low'] < dataframe['last_l_fvg_high'])
            , 1, 0
        )

        dataframe['above_l_fvg'] = np.where(
            (dataframe ['close'] > dataframe['last_l_fvg_low'])
            & (dataframe['open'] > dataframe['last_l_fvg_low'])
            & (dataframe['last_l_fvg_low'].notna())
            , 1, np.where(
                (dataframe['close'] < dataframe['last_l_fvg_low'])
                & (dataframe['last_l_fvg_low'].notna()),
                -1, 0)
        )

        return dataframe

    def s_fvg_feature(self, dataframe: DataFrame) -> DataFrame:
        """检查当前K线收在 s_fvg 的下方"""

        # 使用rolling窗口寻找最近的有效FVG值
        dataframe['last_s_fvg_high'] = (
            dataframe['s_fvg_high']
            .shift(1)  # 排除当前K线
            .rolling(window=self.below_s_fvg_window.value, min_periods=1)
            .apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan)
        )

        dataframe['last_s_fvg_low'] = (
            dataframe['s_fvg_low']
            .shift(1)
            .rolling(window=self.below_s_fvg_window.value, min_periods=1)
            .apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan)
        )

        # 1. 如果K线的low 落在fvg区间，则为被测试过。
        dataframe['valid_s_fvg'] = np.where(
            (dataframe['high'] > dataframe['last_s_fvg_low'])
            , 1, 0
        )

        dataframe['below_s_fvg'] = np.where(
            (dataframe ['close'] < dataframe['last_s_fvg_high'])
            & (dataframe['open'] < dataframe['last_s_fvg_high'])
            & (dataframe['last_s_fvg_high'].notna())
            , 1, np.where(
                (dataframe['close'] > dataframe['last_s_fvg_high'])
                & (dataframe['last_s_fvg_high'].notna()),
                -1, 0)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ########################### long below #####################################

        dataframe.loc[
            (
                (dataframe['l_fvg_label'] == 1)
                & (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'entry_long_fvg')

        ########################### short below #####################################

        dataframe.loc[
            (
                (dataframe["s_fvg_label"] == 1)
                & (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'entry_short_fvg')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 此处主要用于兼容性设置，实际退出逻辑由自定义方法处理

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        ##############################  exit long below ############################
        # exit_long_1 = (
        #     (dataframe['close'] < dataframe["ema_20"])
        # )
        #
        # conditions_long.append(exit_long_1)
        # dataframe.loc[exit_long_1, 'exit_tag'] += 'exit_long_ema'
        #
        # if conditions_long:
        #     dataframe.loc[
        #         reduce(lambda x, y: x | y, conditions_long),
        #         'exit_long'] = 1

        ##############################  exit short below ############################
        # exit_short_1 = (
        #     (dataframe['cross_above_ema'] == 1)
        # )
        # conditions_short.append(exit_short_1)
        # dataframe.loc[exit_short_1, 'exit_tag'] += 'exit_short_ema'
        #
        # if conditions_short:
        #     dataframe.loc[
        #         reduce(lambda x, y: x | y, conditions_short),
        #         'exit_short'] = 1

        return dataframe

    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
    #                         current_time: datetime, entry_tag: Optional[str],
    #                         **kwargs) -> bool:
    #
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     if len(dataframe) < 1:
    #         return False
    #     entry_candle = dataframe.iloc[-1].squeeze()
    #     if 'long' in entry_tag:
    #         if rate < entry_candle['long_pullback_min']:
    #             return False
    #         if rate > entry_candle['higher_trend_max']:
    #             return False
    #
    #     if 'short' in entry_tag:
    #         if rate > entry_candle['short_pullback_max']:
    #             return False
    #         if rate < entry_candle['lower_trend_min']:
    #             return False
    #     return True

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

        long_stoploss_price = entry_candle['l_fvg_k_low']
        short_stoploss_price = entry_candle['s_fvg_k_high']

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
        long_stoploss_price = entry_candle['l_fvg_k_low']
        short_stoploss_price = entry_candle['s_fvg_k_high']

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
            pass

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
