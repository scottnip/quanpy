from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)
#用于临时存储交易 ID
TMP_HOLD = []
TMP_HOLD1 = []


class E0V1E_V2(IStrategy):
    # 定义最小投资回报率（ROI）：从交易开始的第 0 个周期起，期望的回报率为 100%
    # 相当于不设置
    minimal_roi = {
        "0": 1
    }
    # 策略使用的 K 线时间周期，这里设置为 5 分钟
    timeframe = "5m"
    # 只处理新生成的 K 线数据，提高策略处理效率
    process_only_new_candles = True
    # 策略启动时需要的初始 K 线数量
    startup_candle_count = 240
    order_types = {
        "entry": "market", # 市价单
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False, # False 表示不在交易所设置止损单
        "stoploss_on_exchange_interval": 60, # 检查止损单的时间间隔
        "stoploss_on_exchange_market_ratio": 0.99, # 止损单的市价比例
    }
    # 当亏损达到 25% 时触发止损
    stoploss = -0.25
    # False 表示不启用动态止损
    trailing_stop = False
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    # 使用自定义的止损逻辑
    use_custom_stoploss = True

    ## 参数优化

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space="buy", optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space="buy", optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space="buy", optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space="buy", optimize=is_optimize_32)
    sell_fastx = IntParameter(50, 100, default=84, space="sell", optimize=True)

    cci_opt = False
    sell_loss_cci = IntParameter(low=0, high=600, default=120, space="sell", optimize=cci_opt)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.05, decimals=2, space="sell", optimize=cci_opt)

    ## >> 比E0V1E多了几个参数
    buy_rsi_period = IntParameter(10, 190, default=20, space="buy")
    buy_rsi_fast_period = IntParameter(10, 190, default=10, space="buy")
    buy_rsi_slow_period = IntParameter(10, 190, default=40, space="buy")
    buy_sma_period = IntParameter(10, 190, default=15, space="buy")

    # 设置冷却期为 18 根 K 线，即在交易完成后的 18 根 K 线内不允许再次开仓
    @property
    def protections(self):

        return [
            {
                "method": "CooldownPeriod",
                 "stop_duration_candles": 18
            }
        ]

    # 自定义止损方法
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs,) -> float:
        # 如果当前利润大于等于 5%，则将止损比例设置为 -0.2%
        if current_profit >= 0.05:
            return -0.002
        # 如果入场标签为 buy_new 且当前利润大于等于 3%，则将止损比例设置为 -0.3%
        if str(trade.enter_tag) == "buy_new" and current_profit >= 0.03:
            return -0.003
        # 否则返回 None，表示使用默认止损逻辑
        return None

    # 指标计算方法
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # buy_1 indicators
        # N 周期的简单移动平均线（SMA）
        dataframe["sma_15"] = ta.SMA(dataframe, timeperiod=int(self.buy_sma_period.value))
        # 20 周期的 CTI 指标
        dataframe["cti"] = pta.cti(dataframe["close"], length=20)
        # N1 周期和 N2 周期、N3 周期的相对强弱指数（RSI）
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=int(self.buy_rsi_period.value))
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=int(self.buy_rsi_fast_period.value))
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=int(self.buy_rsi_slow_period.value))

        # profit sell indicators
        # 快速随机指标（STOCHF）的 fastk 值
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastk"] = stoch_fast["fastk"]

        # 20 周期的 CCI 指标
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)

        # 120 周期和 240 周期的移动平均线（MA）
        dataframe["ma120"] = ta.MA(dataframe, timeperiod=120)
        dataframe["ma240"] = ta.MA(dataframe, timeperiod=240)

        return dataframe

    # 入场信号生成方法
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""
        # buy_1是列向量，每一行是True or False
        # Example: buy_1 = [True, False, True, False,...]
        buy_1 = (
            (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] < self.buy_rsi_fast_32.value)
            & (dataframe["rsi"] > self.buy_rsi_32.value)
            & (dataframe["close"] < dataframe["sma_15"] * self.buy_sma15_32.value)
            & (dataframe["cti"] < self.buy_cti_32.value)
        )

        buy_new = (
            (dataframe["rsi_slow"] < dataframe["rsi_slow"].shift(1))
            & (dataframe["rsi_fast"] < 34)
            & (dataframe["rsi"] > 28)
            & (dataframe["close"] < dataframe["sma_15"] * 0.96)
            & (dataframe["cti"] < self.buy_cti_32.value)
        )

        conditions.append(buy_1)
        # buy_1为True的行，设置标签
        dataframe.loc[buy_1, "enter_tag"] += "buy_1"

        conditions.append(buy_new)
        dataframe.loc[buy_new, "enter_tag"] += "buy_new"

        # enter_long 通常是 Freqtrade 策略中用于标记买入信号的一个列名，在 pandas.DataFrame 数据结构里作为一个特定的列存在。
        # 当 enter_long 列中的某个值为 1（或 True ，具体取决于策略代码的实现）时，代表在对应的时间点和交易对有买入（做多）的信号；
        # 若值为 0（或 False），则表示没有买入信号。
        if conditions:
            # 如果每行，buy_1 或者 buy_new 有一个为True，设置enter_long=1
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1
        return dataframe

    # 自定义退出方法
    def custom_exit(self, pair: str, trade: "Trade", current_time: "datetime", current_rate: float,
                    current_profit: float, **kwargs,):

        # 获取当前交易对的分析数据和当前 K 线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        # 计算最小盈利比率 = (当前价格 - 买入价格) / 买入价格
        min_profit = trade.calc_profit_ratio(trade.min_rate)

        # 当前价格大于 120 周期和 240 周期均线时，将交易 ID 添加到 TMP_HOLD 列表
        if (
            current_candle["close"] > current_candle["ma120"]
            and current_candle["close"] > current_candle["ma240"]
        ):
            if trade.id not in TMP_HOLD:
                TMP_HOLD.append(trade.id)

        if (trade.open_rate - current_candle["ma120"]) / trade.open_rate >= 0.1:
            if trade.id not in TMP_HOLD1:
                TMP_HOLD1.append(trade.id)
        # 当前交易盈利，并且当前 K 线的 fastk 指标值大于阈值时，卖出，并返回卖出信号
        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"
        # 如果交易处于亏损状态且亏损幅度达到或超过 10%
        if min_profit <= -0.1:
            # 当前的盈利是否已经回升并超过了设定的这个阈值
            if current_profit > self.sell_loss_cci_profit.value:
                # 当前 K 线的 CCI 值是否大于设定的阈值。
                # 当 CCI 值超过该阈值时，可能暗示市场处于超买状态，价格可能即将下跌。
                # 若满足该条件，则触发最后的卖出逻辑
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell"

        if trade.id in TMP_HOLD1 and current_candle["close"] < current_candle["ma120"]:
            TMP_HOLD1.remove(trade.id)
            return "ma120_sell_fast"

        if (
            trade.id in TMP_HOLD
            and current_candle["close"] < current_candle["ma120"]
            and current_candle["close"] < current_candle["ma240"]
        ):
            if min_profit <= -0.1:
                TMP_HOLD.remove(trade.id)
                return "ma120_sell"

        return None

    # 出场信号生成方法
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 默认不触发平仓信号
        # exit_long 主要用于标记多头交易的退出信号
        dataframe.loc[:, ["exit_long", "exit_tag"]] = (0, "long_out")
        return dataframe