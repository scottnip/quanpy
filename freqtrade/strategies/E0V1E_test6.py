from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)
# 用于临时存储交易 ID
TMP_HOLD = []
TMP_HOLD1 = []

# Buy hyperspace params:
buy_params = {
        "base_nb_candles_buy": 12,
        "ewo_high": 3.147,
        "ewo_low": -17.145,
        "low_offset": 0.987,
        "rsi_buy": 57,
    }

# Sell hyperspace params:
sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.008,
        "high_offset_2": 1.016,
    }


def EWO(dataframe, ema_length=5, ema2_length=30):
    df = dataframe.copy()
    # 短期EMA
    ema1 = ta.EMA(df, timeperiod=ema_length)
    # 长期EMA
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class E0V1E_test6(IStrategy):
    # 定义最小投资回报率（ROI）：从交易开始的第 0 个周期起，期望的回报率为 100%
    # 相当于不设置
    minimal_roi = {
        "0": 1
    }
    # 策略使用的时间框架为 5 分钟
    timeframe = '5m'
    inf_1h = '1h'
    # 只处理新生成的 K 线数据，提高策略处理效率
    process_only_new_candles = True
    # 策略启动时需要的初始 K 线数量
    startup_candle_count = 400

    order_types = {
        'entry': 'market', # 市价单
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False, # False 表示不在交易所设置止损单
        'stoploss_on_exchange_interval': 60, # 检查止损单的时间间隔
        'stoploss_on_exchange_market_ratio': 0.99 # 止损单的市价比例
    }
    # 防止交易过程中出现过度滑点的保护机制
    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }
    cc = {}

    # 设置固定止损为 -25 %
    stoploss = -0.25

    # 表示使用自定义的止损函数
    use_custom_stoploss = False

    ## 优化参数部分

    # 布尔变量，用于控制某些参数是否进行优化
    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)
    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    # 布尔变量，用于控制与 CCI 相关的参数是否进行优
    cci_opt = False
    sell_loss_cci = IntParameter(low=0, high=600, default=120, space='sell', optimize=cci_opt)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.05, decimals=2, space='sell', optimize=cci_opt)

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    # 启用追踪止损
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',  # gtc 表示直到取消
        'sell': 'gtc'
    }

    # my plot add ...
    plot_config = {
        'main_plot': {
            'ma120': {'color': '#c85656'},
            'ma240': {'color': '#c5950e'},
            'sma_15': {'color': '#4f91ac'}
        },
        'subplots': {
            'rsi': {
                'rsi': {'color': '#38e25a'},
                'rsi_fast': {'color': '#90eed2'},
                'rsi_slow': {'color': '#eb4760'}
            },
            'cti': {
                'cti': {'color': '#75fc40'}
            },
            'cci': {
                'cci': {'color': '#54e5f8'}
            },
            'conds': {
                'change': {'color': '#bcb622'},
                'fastk': {'color': '#e830dc'}
            },
        }
    }

    # 分别表示两种保护机制的设置：低收益对保护，冷却期保护。
    @property
    def protections(self):

        return [
            # 监控交易对的盈利能力，如果某个交易对在指定时间段内的盈利低于设定的阈值，则停止该交易对的交易。
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 60, # 表示检查过去 60 根 K 线的交易表现
                "trade_limit": 1, # 限制同一时间内该交易对的交易数量最多为 1 笔
                "stop_duration_candles": 60, # 如果某个交易对的盈利情况不满足要求，策略会停止该交易对的交易，停止的时长为 60 根 K 线所代表的时间。在这 60 根 K 线的时间段内，不会对该交易对进行新的开仓操作。
                "required_profit": -0.05 # 如果在回顾的 60 根 K 线内，该交易对的盈利低于 -5%，则会触发策略的停止交易机制。
            },
            # # 设置冷却期为 18 根 K 线，即在交易完成后的 18 根 K 线内不允许再次开仓
            {
                "method": "CooldownPeriod", # 在一次交易完成后，设置一个冷却期，在此期间不允许再次开仓，以避免过度交易和频繁操作
                "stop_duration_candles": 5 # 冷却期的时长为 5 根 K 线所代表的时
            }
        ]

    # 指标计算方法
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # buy_1 indicators
        # 15 天的简单移动平均线（SMA）
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        # 衡量资产价格的趋势强度和方向
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        # 通过上涨幅度和下跌幅度的对比，衡量资产价格的超买或超卖状态，判断趋势强度及潜在反转信号
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # profit sell indicators
        # 衡量当前价格在近期价格区间中的相对位置，帮助判断超买或超卖状态
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        # 衡量价格相对于统计平均值的偏离程度，判断资产的超买超卖状态及趋势强度
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)
        dataframe['ma240'] = ta.MA(dataframe, timeperiod=240)

        # my add, only for plot the % change of the candle ...
        dataframe['change'] = (100 / dataframe['open'] * dataframe['close'] - 100)

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # HMA，保持平滑度的同时显著降低传统移动平均线的滞后问题
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
                # 慢速RSI下降，表明中期动量正在减弱，避免在强势上涨趋势中追高，等待短期回调信号。
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                # 快速RSI（如7周期）低于预设值（如30），处于超卖区域，捕捉短期超卖后的反弹机会
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                # 标准RSI高于阈值（40），保持在中性区域上方。确保中期趋势未完全转弱，避免在长期下跌趋势中逆势操作。
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                # 收盘价低于均线折扣，价格低于15日均线的某个比例（如95%）。利用均值回归逻辑，寻找价格回调至均线支撑位的买入点。
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                # CTI低于阈值（如-0.5），表明趋势强度较弱或处于回调阶段。避免在强势趋势中逆势交易，侧重震荡市或趋势初期的反转机会。
                (dataframe['cti'] < self.buy_cti_32.value)
        )

        # buy_2 = (
        #     (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1)) &
        #     (dataframe["rsi_fast"] < 34) &
        #     (dataframe["rsi"] > 28) &
        #     (dataframe["close"] < dataframe["sma_15"] * 0.96) &
        #     (dataframe["cti"] < self.buy_cti_32.value)
        # )

        buy_3 = (
                # 市场处于超卖状态，意味着价格可能已经下跌过多，存在反弹的可能性
                (dataframe['rsi_fast'] < 35)&
                # 收盘价小于买入移动平均线的值乘以偏移值
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                # 短期EMA突破长期EMA，金叉
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        # conditions.append(buy_2)
        # dataframe.loc[buy_2, 'enter_tag'] += 'buy_2'

        conditions.append(buy_3)
        dataframe.loc[buy_3, 'enter_tag'] += 'buy_3'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                # 收盘价高于HMA
                    (dataframe['close'] > dataframe['hma_50']) &
                    # 收盘价高于MA
                    (dataframe['close'] > (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                    # 市场处于上升趋势
                    (dataframe['rsi'] > 50) &
                    # 成交量
                    (dataframe['volume'] > 0) &
                    # 当快速 RSI 超过慢速 RSI 时，说明短期市场的上涨动能较强，价格在短期内有加速上涨的趋势
                    (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
            |
            (
                # 收盘价低于HMA
                    (dataframe['close'] < dataframe['hma_50']) &
                    # 收盘价高于MA
                    (dataframe['close'] > (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['volume'] > 0) &
                    # 快速RSI突破慢速RSI
                    (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe
