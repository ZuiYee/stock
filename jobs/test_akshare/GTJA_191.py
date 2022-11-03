#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.stats import rankdata
import logging
import multiprocessing
import pandas as pd
import libs.common as common


class GTJA_191:

    def __init__(self, df_data, symbol):
        import copy
        df = copy.deepcopy(df_data)
        # self.open = df['open'].to_frame(name=symbol)
        self.open = pd.to_numeric(df['open'], errors='coerce').to_frame(name=symbol)
        self.open_price = self.open
        # self.high = df["high"].to_frame(name=symbol)
        self.high = pd.to_numeric(df['high'], errors='coerce').to_frame(name=symbol)
        # self.low = df["low"].to_frame(name=symbol)
        self.low = pd.to_numeric(df['low'], errors='coerce').to_frame(name=symbol)
        # self.close = df["close"].to_frame(name=symbol)
        self.close = pd.to_numeric(df['close'], errors='coerce').to_frame(name=symbol)
        # self.volume = df["volume"].to_frame(name=symbol)
        self.volume = pd.to_numeric(df['volume'], errors='coerce').to_frame(name=symbol)
        # self.returns = df["turnover"].to_frame(name=symbol)  # 涨跌幅(%)
        self.returns = pd.to_numeric(df['turnover'], errors='coerce').to_frame(name=symbol)
        # self.avg = df["avg"].to_frame(name="000066.XSHE")  # 均价(VWAP)

    def func_rank(self, na):
        return rankdata(na)[-1] / rankdata(na).max()

    def func_decaylinear(self, na):
        n = len(na)
        decay_weights = np.arange(1, n + 1, 1)
        decay_weights = decay_weights / decay_weights.sum()

        return (na * decay_weights).sum()

    def func_highday(self, na):
        return len(na) - na.values.argmax()

    def func_lowday(self, na):
        return len(na) - na.values.argmin()

    #############################################################################

    def alpha_001(self):
        data1 = np.log(self.volume).diff(periods=1).rank(axis=0, pct=True)
        data2 = ((self.close - self.open) / self.open).rank(axis=0, pct=True)
        alpha = data1.iloc[-6:, :].corrwith(data2.iloc[-6:, :]).dropna()
        alpha = -alpha
        alpha = alpha.dropna()
        return alpha

    def alpha_002(self):
        ##### -1 * delta((((close-low)-(high-close))/((high-low)),1))####
        result = (((self.close - self.low) - (self.high - self.close)) / ((self.high - self.low))).diff()
        m = result.iloc[-1, :].dropna()
        alpha = m[(m < np.inf) & (m > -np.inf)]
        alpha = -alpha
        return alpha.dropna()

        ################################################################

    def alpha_003(self):
        delay1 = self.close.shift()
        condtion1 = (self.close == delay1)
        condition2 = (self.close > delay1)
        condition3 = (self.close < delay1)

        part2 = (self.close - np.minimum(delay1[condition2], self.low[condition2])).iloc[-6:, :]  # 取最近的6位数据
        part3 = (self.close - np.maximum(delay1[condition3], self.low[condition3])).iloc[-6:, :]

        result = part2.fillna(0) + part3.fillna(0)
        alpha = result.sum()
        return alpha.dropna()

    ########################################################################
    def alpha_004(self):
        # condition1 = (pd.rolling_std(self.close, 8) < pd.rolling_sum(self.close, 2) / 2)
        condition1 = (self.close.rolling(8).std() < self.close.rolling(2).sum() / 2)
        # condition2 = (pd.rolling_sum(self.close, 2) / 2 < (pd.rolling_sum(self.close, 8) / 8 - pd.rolling_std(self.close, 8)))
        condition2 = (self.close.rolling(2).sum() / 2 < (self.close.rolling(8).sum() / 8 - self.close.rolling(8).std()))
        # condition3 = (1 <= self.volume / pd.rolling_mean(self.volume, 20))
        condition3 = (1 <= self.volume / self.volume.rolling(20).mean())

        indicator1 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index,
                                  columns=self.close.columns)  # [condition2]
        indicator2 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index,
                                   columns=self.close.columns)  # [condition3]

        # part0 = pd.rolling_sum(self.close, 8) / 8
        part0 = self.close.rolling(8).sum() / 8
        part1 = indicator2[condition1].fillna(0)
        part2 = (indicator1[~condition1][condition2]).fillna(0)
        part3 = (indicator1[~condition1][~condition2][condition3]).fillna(0)
        part4 = (indicator2[~condition1][~condition2][~condition3]).fillna(0)

        result = part0 + part1 + part2 + part3 + part4
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ################################################################
    def alpha_005(self):
        ts_volume = (self.volume.iloc[-7:, :]).rank(axis=0, pct=True)
        ts_high = (self.high.iloc[-7:, :]).rank(axis=0, pct=True)
        # corr_ts = pd.rolling_corr(ts_high, ts_volume, 5)
        corr_ts = ts_high.rolling(5).corr(ts_volume)
        alpha = corr_ts.max().dropna()
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        return alpha

        ###############################################################

    def alpha_006(self):
        condition1 = ((self.open_price * 0.85 + self.high * 0.15).diff(4) > 1)
        condition2 = ((self.open_price * 0.85 + self.high * 0.15).diff(4) == 1)
        condition3 = ((self.open_price * 0.85 + self.high * 0.15).diff(4) < 1)
        indicator1 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)
        indicator2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        indicator3 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1 = indicator1[condition1].fillna(0)
        part2 = indicator2[condition2].fillna(0)
        part3 = indicator3[condition3].fillna(0)
        result = part1 + part2 + part3
        alpha = (result.rank(axis=1, pct=True)).iloc[-1, :]  # cross section rank
        return alpha.dropna()

    ##################################################################
    def alpha_007(self):
        part1 = (np.maximum(self.avg_price - self.close, 3)).rank(axis=1, pct=True)
        part2 = (np.minimum(self.avg_price - self.close, 3)).rank(axis=1, pct=True)
        part3 = (self.volume.diff(3)).rank(axis=1, pct=True)
        result = part1 + part2 * part3
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_008(self):
        temp = (self.high + self.low) * 0.2 / 2 + self.avg * 0.8
        result = -temp.diff(4)
        alpha = result.rank(axis=0, pct=True)
        alpha = alpha.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_009(self):
        temp = (self.high + self.low) * 0.5 - (self.high.shift() + self.low.shift()) * 0.5 * (
                self.high - self.low) / self.volume  # 计算close_{i-1}
        result = pd.DataFrame.ewm(temp, alpha=2 / 7).mean()
        # result = temp.ewma(alpha= 2/7).mean()
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_010(self):
        ret = self.close.pct_change()
        condtion = (ret < 0)
        part1 = (pd.rolling_std(ret, 20)[condtion]).fillna(0)
        part2 = (self.close[~condtion]).fillna(0)
        result = np.maximum((part1 + part2) ** 2, 5)
        alpha = result.rank(axis=1, pct=True)
        alpha = alpha.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_011(self):
        temp = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        result = temp * self.volume
        alpha = result.iloc[-6:, :].sum()
        return alpha.dropna()

    ##################################################################
    def alpha_012(self):
        vwap10 = pd.rolling_sum(self.avg_price, 10) / 10
        temp1 = self.open_price - vwap10
        part1 = temp1.rank(axis=1, pct=True)
        temp2 = (self.close - self.avg_price).abs()
        part2 = -temp2.rank(axis=1, pct=True)
        result = part1 * part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_013(self):
        result = ((self.high - self.low) ** 0.5) - self.avg_price
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_014(self):
        result = self.close - self.close.shift(5)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_015(self):
        result = self.open_price / self.close.shift() - 1
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_016(self):
        temp1 = self.volume.rank(axis=1, pct=True)
        temp2 = self.avg_price.rank(axis=1, pct=True)
        part = pd.rolling_corr(temp1, temp2, 5)  #
        part = part[(part < np.inf) & (part > -np.inf)]
        result = part.iloc[-5:, :]
        result = result.dropna(axis=1)
        alpha = -result.max()
        return alpha.dropna()

    ##################################################################
    def alpha_017(self):
        temp1 = pd.rolling_max(self.avg_price, 15)
        temp2 = (self.close - temp1).dropna()
        part1 = temp2.rank(axis=1, pct=True)
        part2 = self.close.diff(5)
        result = part1 ** part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_018(self):
        delay5 = self.close.shift(5)
        alpha = self.close / delay5
        alpha = alpha.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_019(self):
        delay5 = self.close.shift(5)
        condition1 = (self.close < delay5)
        condition3 = (self.close > delay5)
        part1 = (self.close[condition1] - delay5[condition1]) / delay5[condition1]
        part1 = part1.fillna(0)
        part2 = (self.close[condition3] - delay5[condition3]) / self.close[condition3]
        part2 = part2.fillna(0)
        result = part1 + part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_020(self):
        delay6 = self.close.shift(6)
        result = (self.close - delay6) * 100 / delay6
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_021(self):
        # A = pd.rolling_mean(self.close, 6).iloc[-6:, :]
        A = self.close.rolling(6).mean().iloc[-6:, :]
        B = np.arange(1, 7)
        temp = A.apply(lambda x: sp.stats.linregress(x, B), axis=0)
        drop_list = [i for i in range(len(temp)) if temp[i][3] > 0.05]
        temp.drop(temp.index[drop_list], inplace=True)
        beta_list = [temp[i].slope for i in range(len(temp))]
        alpha = pd.Series(beta_list, index=temp.index)
        return alpha.dropna()

    ##################################################################
    def alpha_022(self):
        # part1 = (self.close - pd.rolling_mean(self.close, 6)) / pd.rolling_mean(self.close, 6)
        part1 = (self.close - self.close.rolling(6).mean()) / self.close.rolling(6).mean()
        # temp = (self.close - pd.rolling_mean(self.close, 6)) / pd.rolling_mean(self.close, 6)
        temp = (self.close - self.close.rolling(6).mean()) / self.close.rolling(6).mean()
        part2 = temp.shift(3)
        result = part1 - part2
        # result = pd.ewma(result, alpha=1.0 / 12)
        result = pd.DataFrame.ewm(result, alpha=1.0 / 12).mean()
        alpha = result.iloc[-1, :]
        return alpha.dropna()

        ##################################################################

    def alpha_023(self):
        condition1 = (self.close > self.close.shift())
        # temp1 = pd.rolling_std(self.close, 20)[condition1]
        temp1 = self.close.rolling(20).std()[condition1]
        temp1 = temp1.fillna(0)
        # temp2 = pd.rolling_std(self.close, 20)[~condition1]
        temp2 = self.close.rolling(20).std()[~condition1]
        temp2 = temp2.fillna(0)
        part1 = pd.DataFrame.ewm(temp1, alpha=1.0 / 20).mean()
        part2 = pd.DataFrame.ewm(temp2, alpha=1.0 / 20).mean()
        result = part1 * 100 / (part1 + part2)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_024(self):
        delay5 = self.close.shift(5)
        result = self.close - delay5
        # result = pd.ewma(result, alpha=1.0 / 5)
        result = pd.DataFrame.ewm(result, alpha=1.0 / 5).mean()
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_025(self):
        n = 9
        part1 = (self.close.diff(7)).rank(axis=1, pct=True)
        part1 = part1.iloc[-1, :]
        temp = self.volume / pd.rolling_mean(self.volume, 20)
        temp1 = temp.iloc[-9:, :]
        seq = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        weight = np.array(seq)

        temp1 = temp1.apply(lambda x: x * weight)
        ret = self.close.pct_change()
        rank_sum_ret = (ret.sum()).rank(pct=True)
        part2 = 1 - temp1.sum()
        part3 = 1 + rank_sum_ret
        alpha = -part1 * part2 * part3
        return alpha.dropna()

    ##################################################################
    def alpha_026(self):
        # part1 = pd.rolling_sum(self.close, 7) / 7 - self.close
        part1 = self.close.rolling(7).sum() / 7 - self.close
        part1 = part1.iloc[-1, :]
        delay5 = self.close.shift(5)
        part2 = pd.rolling_corr(self.avg_price, delay5, 230)
        part2 = part2.iloc[-1, :]
        alpha = part1 + part2
        return alpha.dropna()

    ##################################################################
    def alpha_027(self):
        return 0

    ##################################################################
    def alpha_028(self):
        # temp1 = self.close - pd.rolling_min(self.low, 9)
        temp1 = self.close - self.low.rolling(9).min()
        # temp2 = pd.rolling_max(self.high, 9) - pd.rolling_min(self.low, 9)
        temp2 = self.high.rolling(9).max() - self.low.rolling(9).min()
        part1 = 3 * pd.DataFrame.ewm(temp1 * 100 / temp2, alpha=1.0 / 3).mean()
        temp3 = pd.DataFrame.ewm(temp1 * 100 / temp2, alpha=1.0 / 3).mean()
        part2 = 2 * pd.DataFrame.ewm(temp3, alpha=1.0 / 3).mean()
        result = part1 - part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_029(self):
        delay6 = self.close.shift(6)
        result = (self.close - delay6) * self.volume / delay6
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_030(self):
        return 0

    ##################################################################
    def alpha_031(self):
        # result = (self.close - pd.rolling_mean(self.close, 12)) * 100 / pd.rolling_mean(self.close, 12)
        result = (self.close - self.close.rolling(12).mean()) * 100 / self.close.rolling(12).mean()
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_032(self):
        temp1 = self.high.rank(axis=0, pct=True)
        temp2 = self.volume.rank(axis=0, pct=True)
        # temp3 = pd.rolling_corr(temp1, temp2, 3)
        temp3 = temp1.rolling(window=3).corr(temp2)
        temp3 = temp3[(temp3 < np.inf) & (temp3 > -np.inf)].fillna(0)
        result = (temp3.rank(axis=0, pct=True)).iloc[-3:, :]
        alpha = -result.sum()
        return alpha.dropna()

    ##################################################################
    def alpha_033(self):
        ret = self.close.pct_change()
        # temp1 = pd.rolling_min(self.low, 5)  # TS_MIN
        temp1 = self.low.rolling(5).min()
        part1 = temp1.shift(5) - temp1
        part1 = part1.iloc[-1, :]
        temp2 = (pd.rolling_sum(ret, 240) - pd.rolling_sum(ret, 20)) / 220
        temp2 = (ret.rolling(240).sum() - ret.rolling(20).sum()) / 220
        part2 = temp2.rank(axis=1, pct=True)
        part2 = part2.iloc[-1, :]
        temp3 = self.volume.iloc[-5:, :]
        part3 = temp3.rank(axis=0, pct=True)  # TS_RANK
        part3 = part3.iloc[-1, :]
        alpha = part1 + part2 + part3
        return alpha.dropna()

    ##################################################################
    def alpha_034(self):
        # result = pd.rolling_mean(self.close, 12) / self.close
        result = self.close.rolling(12).mean() / self.close
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_035(self):
        n = 15
        m = 7
        temp1 = self.open_price.diff()
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = 0.65 * self.open_price + 0.35 * self.open_price
        temp2 = pd.rolling_corr(temp2, self.volume, 17)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        alpha = np.minimum(part1.iloc[-1, :], -part2.iloc[-1, :])
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_036(self):
        temp1 = self.volume.rank(axis=1, pct=True)
        temp2 = self.avg_price.rank(axis=1, pct=True)
        part1 = pd.rolling_corr(temp1, temp2, 6)
        result = pd.rolling_sum(part1, 2)
        result = result.rank(axis=1, pct=True)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_037(self):
        ret = self.close.pct_change()
        # temp = pd.rolling_sum(self.open_price, 5) * pd.rolling_sum(ret, 5)
        temp = self.open_price.rolling(5).sum() * ret.rolling(5).sum()
        part1 = temp.rank(axis=1, pct=True)
        part2 = temp.shift(10)
        result = -part1 - part2
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_038(self):
        # sum_20 = pd.rolling_sum(self.high, 20) / 20
        sum_20 = self.high.rolling(20).sum() / 20
        delta2 = self.high.diff(2)
        condition = (sum_20 < self.high)
        result = -delta2[condition].fillna(0)
        alpha = result.iloc[-1, :]
        return alpha

    ##################################################################
    def alpha_039(self):
        n = 8
        m = 12
        temp1 = self.close.diff(2)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]

        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = 0.3 * self.avg_price + 0.7 * self.open_price
        volume_180 = pd.rolling_mean(self.volume, 180)
        sum_vol = pd.rolling_sum(volume_180, 37)
        temp3 = pd.rolling_corr(temp2, sum_vol, 14)
        temp3 = temp3.iloc[-m:, :]
        part2 = -temp3.apply(lambda x: x * weight2)
        part2.rank(axis=1, pct=True)
        result = part1.iloc[-1, :] - part2.iloc[-1, :]
        alpha = result
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_040(self):
        delay1 = self.close.shift()
        condition = (self.close > delay1)
        vol = self.volume[condition].fillna(0)
        # vol_sum = pd.rolling_sum(vol, 26)
        vol_sum = vol.rolling(26).sum()
        vol1 = self.volume[~condition].fillna(0)
        # vol1_sum = pd.rolling_sum(vol1, 26)
        vol1_sum = vol1.rolling(26).sum()
        result = 100 * vol_sum / vol1_sum
        result = result.iloc[-1, :]
        alpha = result
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_041(self):
        delta_avg = self.avg_price.diff(3)
        part = np.maximum(delta_avg, 5)
        result = -part.rank(axis=1, pct=True)
        alpha = result.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_042(self):
        # part1 = pd.rolling_corr(self.high, self.volume, 10)
        part1 = self.high.rolling(window=10).corr(self.volume)

        # part2 = pd.rolling_std(self.high, 10)
        part2 = self.high.rolling(window=10).mean()
        part2 = part2.rank(axis=0, pct=True)
        result = -part1 * part2
        alpha = result.iloc[-1, :]
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_043(self):
        delay1 = self.close.shift()
        condition1 = (self.close > delay1)
        condition2 = (self.close < delay1)
        temp1 = self.volume[condition1].fillna(0)
        temp2 = -self.volume[condition2].fillna(0)
        result = temp1 + temp2
        # result = pd.rolling_sum(result, 6)
        result = result.rolling(6).sum()
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_044(self):
        part1 = self.open_price * 0.4 + self.close * 0.6
        n = 6
        m = 10
        temp1 = pd.rolling_corr(self.low, pd.rolling_mean(self.volume, 10), 7)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.iloc[-4:, ].rank(axis=0, pct=True)

        temp2 = self.avg_price.diff(3)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        part2 = part1.iloc[-5:, ].rank(axis=0, pct=True)
        alpha = part1.iloc[-1, :] + part2.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_045(self):
        temp1 = self.close * 0.6 + self.open_price * 0.4
        part1 = temp1.diff()
        part1 = part1.rank(axis=1, pct=True)
        temp2 = pd.rolling_mean(self.volume, 150)
        part2 = pd.rolling_corr(self.avg_price, temp2, 15)
        part2 = part2.rank(axis=1, pct=True)
        result = part1 * part2
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_046(self):
        # part1 = pd.rolling_mean(self.close, 3)
        part1 = self.close.rolling(3).mean()
        # part2 = pd.rolling_mean(self.close, 6)
        part2 = self.close.rolling(6).mean()
        # part3 = pd.rolling_mean(self.close, 12)
        part3 = self.close.rolling(12).mean()
        # part4 = pd.rolling_mean(self.close, 24)
        part4 = self.close.rolling(24).mean()
        result = (part1 + part2 + part3 + part4) * 0.25 / self.close
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_047(self):
        # part1 = pd.rolling_max(self.high, 6) - self.close
        part1 = self.high.rolling(6).max() - self.close
        # part2 = pd.rolling_max(self.high, 6) - pd.rolling_min(self.low, 6)
        part2 = self.high.rolling(6).max() - self.low.rolling(6).min()
        result = pd.DataFrame.ewm(100 * part1 / part2, alpha=1.0 / 9).mean()
        alpha = result.iloc[-1, :].dropna()
        return alpha

        ##################################################################

    def alpha_048(self):
        condition1 = (self.close > self.close.shift())
        condition2 = (self.close.shift() > self.close.shift(2))
        condition3 = (self.close.shift(2) > self.close.shift(3))

        indicator1 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            condition1].fillna(0)
        indicator2 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            condition2].fillna(0)
        indicator3 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            condition3].fillna(0)

        indicator11 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            (~condition1) & (self.close != self.close.shift())].fillna(0)
        indicator22 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            (~condition2) & (self.close.shift() != self.close.shift(2))].fillna(0)
        indicator33 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            (~condition3) & (self.close.shift(2) != self.close.shift(3))].fillna(0)

        summ = indicator1 + indicator2 + indicator3 + indicator11 + indicator22 + indicator33
        result = -summ * pd.rolling_sum(self.volume, 5) / pd.rolling_sum(self.volume, 20)
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_049(self):
        delay_high = self.high.shift()
        delay_low = self.low.shift()
        condition1 = (self.high + self.low >= delay_high + delay_low)
        condition2 = (self.high + self.low <= delay_high + delay_low)
        part1 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part1 = part1[~condition1]
        part1 = part1.iloc[-12:, :].sum()

        part2 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part2 = part2[~condition2]
        part2 = part2.iloc[-12:, :].sum()
        result = part1 / (part1 + part2)
        alpha = result.dropna()
        return alpha

    ##################################################################
    def alpha_050(self):

        return 0

    ##################################################################
    def alpha_051(self):

        return 0

    ##################################################################
    def alpha_052(self):
        delay = ((self.high + self.low + self.close) / 3).shift()
        part1 = (np.maximum(self.high - delay, 0)).iloc[-26:, :]

        part2 = (np.maximum(delay - self.low, 0)).iloc[-26:, :]
        alpha = part1.sum() + part2.sum()
        return alpha

    ##################################################################
    def alpha_053(self):
        delay = self.close.shift()
        condition = self.close > delay
        result = self.close[condition].iloc[-12:, :]
        alpha = result.count() * 100 / 12
        return alpha.dropna()

    ##################################################################
    def alpha_054(self):
        part1 = (self.close - self.open_price).abs()
        part1 = part1.std()
        part2 = (self.close - self.open_price).iloc[-1, :]
        part3 = self.close.iloc[-10:, :].corrwith(self.open_price.iloc[-10:, :])
        result = (part1 + part2 + part3).dropna()
        alpha = result.rank(pct=True)
        return alpha.dropna()

    ##################################################################
    def alpha_055(self):

        return 0

    ##################################################################
    def alpha_056(self):
        part1 = self.open_price.iloc[-1, :] - self.open_price.iloc[-12:, :].min()
        part1 = part1.rank(pct=1)
        temp1 = (self.high + self.low) / 2
        temp1 = pd.rolling_sum(temp1, 19)
        temp2 = pd.rolling_sum(pd.rolling_mean(self.volume, 40), 19)
        part2 = temp1.iloc[-13:, :].corrwith(temp2.iloc[-13:, :])
        part2 = (part2.rank(pct=1)) ** 5
        part2 = part2.rank(pct=1)

        part1[part1 < part2] = 1
        part1 = part1.apply(lambda x: 0 if x < 1 else None)
        alpha = part1.fillna(1)
        return alpha.dropna()

    ##################################################################
    def alpha_057(self):
        # part1 = self.close - pd.rolling_min(self.low, 9)
        part1 = self.close - self.low.rolling(9).min()
        # part2 = pd.rolling_max(self.high, 9) - pd.rolling_min(self.low, 9)
        part2 = self.high.rolling(9).max() - self.low.rolling(9).min()
        result = pd.DataFrame.ewm(100 * part1 / part2, alpha=1.0 / 3).mean()
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_058(self):
        delay = self.close.shift()
        condition = self.close > delay
        result = self.close[condition].iloc[-20:, :]
        alpha = result.count() * 100 / 20
        return alpha.dropna()

        ##################################################################

    def alpha_059(self):
        delay = self.close.shift()
        condition1 = (self.close > delay)
        condition2 = (self.close < delay)
        part1 = np.minimum(self.low[condition1], delay[condition1]).fillna(0)
        part2 = np.maximum(self.high[condition2], delay[condition2]).fillna(0)
        part1 = part1.iloc[-20:, :]
        part2 = part2.iloc[-20:, :]
        result = self.close - part1 - part2
        alpha = result.sum()
        return alpha

    ##################################################################
    def alpha_060(self):
        part1 = (self.close.iloc[-20:, :] - self.low.iloc[-20:, :]) - (
                self.high.iloc[-20:, :] - self.close.iloc[-20:, :])
        part2 = self.high.iloc[-20:, :] - self.low.iloc[-20:, :]
        result = self.volume.iloc[-20:, :] * part1 / part2
        alpha = result.sum()
        return alpha

    ##################################################################
    def alpha_061(self):
        n = 12
        m = 17
        temp1 = self.avg_price.diff()
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]

        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = self.low
        temp2 = pd.rolling_corr(temp2, pd.rolling_mean(self.volume, 80), 8)
        temp2 = temp2.rank(axis=1, pct=1)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        part2 = -part2.rank(axis=1, pct=1)
        alpha = np.maximum(part1.iloc[-1, :], part2.iloc[-1, :])
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_062(self):
        volume_rank = self.volume.rank(axis=0, pct=1)
        result = self.high.iloc[-5:, :].corrwith(volume_rank.iloc[-5:, :])
        # result = self.high.iloc[-5:, :].rolling(5).corr(volume_rank)
        alpha = -result
        return alpha.dropna()

        ##################################################################

    def alpha_063(self):
        part1 = np.maximum(self.close - self.close.shift(), 0)
        part1 = pd.DataFrame.ewm(part1, alpha=1.0 / 6).mean()
        part2 = (self.close - self.close.shift()).abs()
        part2 = pd.DataFrame.ewm(part2, alpha=1.0 / 6).mean()
        result = part1 * 100 / part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_064(self):
        n = 4
        m = 14
        temp1 = pd.rolling_corr(self.avg_price.rank(axis=1, pct=1), self.volume.rank(axis=1, pct=1), 4)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = self.close.rank(axis=1, pct=1)
        temp2 = pd.rolling_corr(temp2, pd.rolling_mean(self.volume, 60), 4)
        temp2 = np.maximum(temp2, 13)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        part2 = -part2.rank(axis=1, pct=1)
        alpha = np.maximum(part1.iloc[-1, :], part2.iloc[-1, :])
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_065(self):
        part1 = self.close.iloc[-6:, :]
        alpha = part1.mean() / self.close.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_066(self):
        part1 = self.close.iloc[-6:, :]
        alpha = (self.close.iloc[-1, :] - part1.mean()) / part1.mean()
        return alpha

    ##################################################################
    def alpha_067(self):
        temp1 = self.close - self.close.shift()
        part1 = np.maximum(temp1, 0)
        part1 = pd.DataFrame.ewm(part1, alpha=1.0 / 24).mean()
        temp2 = temp1.abs()
        part2 = pd.DataFrame.ewm(temp2, alpha=1.0 / 24).mean()
        result = part1 * 100 / part2
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_068(self):
        part1 = (self.high + self.low) / 2 - self.high.shift()
        part2 = 0.5 * self.low.shift() * (self.high - self.low) / self.volume
        result = (part1 + part2) * 100
        result = pd.DataFrame.ewm(result, alpha=2.0 / 15).mean()
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_069(self):

        return 0

    ##################################################################
    def alpha_070(self):
        #### STD(AMOUNT, 6)
        ##
        alpha = self.amount.iloc[-6:, :].std().dropna()
        return alpha

    #############################################################################
    def alpha_071(self):
        # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        #
        # data = self.close - pd.rolling_mean(self.close, 24) / pd.rolling_mean(self.close, 24)
        data = self.close - self.close.rolling(24).mean() / self.close.rolling(24).mean()
        alpha = data.iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_072(self):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        #
        # data1 = pd.rolling_max(self.high, 6) - self.close
        data1 = self.high.rolling(6).max() - self.close
        # data2 = pd.rolling_max(self.high, 6) - pd.rolling_min(self.low, 6)
        data2 = self.high.rolling(6).max() - self.low.rolling(6).min()
        alpha = pd.DataFrame.ewm(data1 / data2 * 100, alpha=1 / 15).mean().iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_073(self):

        return 0

    #############################################################################
    def alpha_074(self):
        # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
        #
        data1 = pd.rolling_sum((self.low * 0.35 + self.avg_price * 0.65), window=20)
        data2 = pd.rolling_mean(self.volume, window=40)
        rank1 = pd.rolling_corr(data1, data2, window=7).rank(axis=1, pct=True)
        data3 = self.avg_price.rank(axis=1, pct=True)
        data4 = self.volume.rank(axis=1, pct=True)
        rank2 = pd.rolling_corr(data3, data4, window=6).rank(axis=1, pct=True)
        alpha = (rank1 + rank2).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_075(self):
        # COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
        #
        benchmark = get_price('000001.SH', None, end_date, '1d', ['open', 'close'], False, None, 50)
        condition = benchmark['close'] < benchmark['open']
        data1 = benchmark[condition]
        numbench = len(data1)
        timelist = data1.index.tolist()
        data2 = pd.merge(self.close, data1, left_index=True, right_index=True).drop(['close', 'open'], axis=1)
        data3 = pd.merge(self.open_price, data1, left_index=True, right_index=True).drop(['close', 'open'], axis=1)
        data4 = data2[data2 > data3]
        alpha = 1 - data4.isnull().sum(axis=0) / numbench
        return alpha

    #############################################################################
    def alpha_076(self):
        # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        #

        data1 = abs((self.close / ((self.open - 1) / self.volume).shift(20))).std()
        data2 = abs((self.close / ((self.open - 1) / self.volume).shift(20))).mean()
        alpha = (data1 / data2).dropna()
        return alpha

    #############################################################################
    def alpha_077(self):
        # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
        #

        data1 = ((self.high + self.low) / 2 + self.high - (self.avg_price + self.high)).iloc[-20:, :]
        decay_weights = np.arange(1, 20 + 1, 1)[::-1]
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.apply(lambda x: x * decay_weights).rank(axis=1, pct=True)
        data2 = pd.rolling_corr((self.high + self.low) / 2, pd.rolling_mean(self.volume, window=40), window=3).iloc[-6:,
                :]
        decay_weights2 = np.arange(1, 6 + 1, 1)[::-1]
        decay_weights2 = decay_weights2 / decay_weights2.sum()
        rank2 = data2.apply(lambda x: x * decay_weights2).rank(axis=1, pct=True)
        alpha = np.minimum(rank1.iloc[-1], rank2.iloc[-1])
        return alpha

    #############################################################################
    def alpha_078(self):
        # ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
        #
        # data1 = (self.high + self.low + self.close) / 3 - pd.rolling_mean((self.high + self.low + self.close) / 3,
        #                                                                   window=12)
        data1 = (self.high + self.low + self.close) / 3 - ((self.high + self.low + self.close) / 3).rolling(12).mean()
        # data2 = abs(self.close - pd.rolling_mean((self.high + self.low + self.close) / 3, window=12))
        data2 = abs(self.close - ((self.high + self.low + self.close) / 3).rolling(12).mean())
        # data3 = pd.rolling_mean(data2, window=12) * 0.015
        data3 = data2.rolling(12).mean() * 0.015
        alpha = (data1 / data3).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_079(self):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        #
        data1 = pd.DataFrame.ewm(np.maximum((self.close - self.open), 0), alpha=1 / 12).mean()
        data2 = pd.DataFrame.ewm(abs(self.close - self.open), alpha=1 / 12).mean()
        alpha = (data1 / data2 * 100).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_080(self):
        # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        #
        alpha = ((self.volume - self.volume.shift(5)) / self.volume.shift(5) * 100).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_081(self):
        result = pd.DataFrame.ewm(self.volume, alpha=2.0 / 21).mean()
        alpha = result.iloc[-1, :].dropna()
        return alpha

    #############################################################################
    def alpha_082(self):
        # part1 = pd.rolling_max(self.high, 6) - self.close
        part1 = self.high.rolling(6).max() - self.close
        # part2 = pd.rolling_max(self.high, 6) - pd.rolling_min(self.low, 6)
        part2 = self.high.rolling(6).max() - self.low.rolling(6).min()
        result = pd.DataFrame.ewm(100 * part1 / part2, alpha=1.0 / 20).mean()
        alpha = result.iloc[-1, :].dropna()
        return alpha

    #############################################################################
    def alpha_083(self):
        part1 = self.high.rank(axis=0, pct=True)
        part1 = part1.iloc[-5:, :]
        part2 = self.volume.rank(axis=0, pct=True)
        part2 = part2.iloc[-5:, :]
        result = part1.corrwith(part2)
        alpha = -result
        return alpha.dropna()

    #############################################################################
    def alpha_084(self):
        condition1 = (self.close > self.close.shift())
        condition2 = (self.close < self.close.shift())
        part1 = self.volume[condition1].fillna(0)
        part2 = -self.volume[condition2].fillna(0)
        result = part1.iloc[-20:, :] + part2.iloc[-20:, :]
        alpha = result.sum().dropna()
        return alpha

    #############################################################################
    def alpha_085(self):
        temp1 = self.volume.iloc[-20:, :] / self.volume.iloc[-20:, :].mean()
        temp1 = temp1
        part1 = temp1.rank(axis=0, pct=True)
        part1 = part1.iloc[-1, :]

        delta = self.close.diff(7)
        temp2 = -delta.iloc[-8:, :]
        part2 = temp2.rank(axis=0, pct=True).iloc[-1, :]
        part2 = part2
        alpha = part1 * part2
        return alpha.dropna()

    #############################################################################
    def alpha_086(self):

        delay10 = self.close.shift(10)
        delay20 = self.close.shift(20)
        indicator1 = pd.DataFrame(-np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)
        indicator2 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)

        temp = (delay20 - delay10) / 10 - (delay10 - self.close) / 10
        condition1 = (temp > 0.25)
        condition2 = (temp < 0)
        temp2 = (self.close - self.close.shift()) * indicator1

        part1 = indicator1[condition1].fillna(0)
        part2 = indicator2[~condition1][condition2].fillna(0)
        part3 = temp2[~condition1][~condition2].fillna(0)
        result = part1 + part2 + part3
        alpha = result.iloc[-1, :].dropna()

        return alpha

    #############################################################################
    def alpha_087(self):
        n = 7
        m = 11
        temp1 = self.avg_price.diff(4)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]

        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = self.low - self.avg_price
        temp3 = self.open_price - 0.5 * (self.high + self.low)
        temp2 = temp2 / temp3
        temp2 = temp2.iloc[-m:, :]
        part2 = -temp2.apply(lambda x: x * weight2)

        part2 = part2.rank(axis=0, pct=1)
        alpha = part1.iloc[-1, :] + part2.iloc[-1, :]
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    '''
    ########################################################################
    '''

    def alpha_088(self):
        # (close-delay(close,20))/delay(close,20)*100
        ####################
        data1 = self.close.iloc[-21, :]
        alpha = ((self.close.iloc[-1, :] - data1) / data1) * 100
        alpha = alpha.dropna()
        return alpha

    def alpha_089(self):
        # 2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))
        ######################
        data1 = pd.DataFrame.ewm(self.close, span=12, adjust=False).mean()
        data2 = pd.DataFrame.ewm(self.close, span=26, adjust=False).mean()
        data3 = pd.DataFrame.ewm(data1 - data2, span=9, adjust=False).mean()
        alpha = ((data1 - data2 - data3) * 2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_090(self):
        # (rank(corr(rank(vwap),rank(volume),5))*-1)
        #######################
        data1 = self.avg_price.rank(axis=1, pct=True)
        data2 = self.volume.rank(axis=1, pct=True)
        corr = data1.iloc[-5:, :].corrwith(data2.iloc[-5:, :])
        rank1 = corr.rank(pct=True)
        alpha = rank1 * -1
        alpha = alpha.dropna()
        return alpha

    def alpha_091(self):
        # ((rank((close-max(close,5)))*rank(corr((mean(volume,40)),low,5)))*-1)
        #################
        data1 = self.close
        cond = data1 > 5
        data1[~cond] = 5
        rank1 = ((self.close - data1).rank(axis=1, pct=True)).iloc[-1, :]
        mean = pd.rolling_mean(self.volume, window=40)
        corr = mean.iloc[-5:, :].corrwith(self.low.iloc[-5:, :])
        rank2 = corr.rank(pct=True)
        alpha = rank1 * rank2 * (-1)
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_092(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1) #
        delta = (self.close * 0.35 + self.avg_price * 0.65) - (self.close * 0.35 + self.avg_price * 0.65).shift(2)
        rank1 = (pd.rolling_apply(delta, 3, self.func_decaylinear)).rank(axis=1, pct=True)
        rank2 = pd.rolling_apply(pd.rolling_apply(self.volume.rolling(180).mean().rolling(13).corr(self.close).abs(), 5,
                                                  self.func_decaylinear), 15, self.func_rank)
        cond_max = rank1 > rank2
        rank2[cond_max] = rank1[cond_max]
        alpha = (-rank2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_093(self):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20) #
        cond = self.open_price >= self.open_price.shift()
        data1 = self.open_price - self.low
        data2 = self.open_price - self.open_price.shift()
        cond_max = data1 > data2
        data2[cond_max] = data1[cond_max]
        data2[cond] = 0
        alpha = data2.iloc[-20:, :].sum()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_094(self):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30) #
        cond1 = self.close > self.prev_close
        cond2 = self.close < self.prev_close
        value = -self.volume
        value[~cond2] = 0
        value[cond1] = self.volume[cond1]
        alpha = value.iloc[-30:, :].sum()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_095(self):
        # STD(AMOUNT,20) #
        alpha = self.amount.iloc[-20:, :].std()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_096(self):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) #
        sma1 = pd.DataFrame.ewm(
            100 * (self.close - self.low.rolling(9).min()) / (self.high.rolling(9).max() - self.low.rolling(9).min()),
            span=5, adjust=False).mean()
        alpha = pd.DataFrame.ewm(sma1, span=5, adjust=False).mean().iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_097(self):
        # STD(VOLUME,10) #
        alpha = self.volume.iloc[-10:, :].std()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_098(self):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        sum_close = self.close.rolling(100).sum()
        cond = (sum_close / 100 - (sum_close / 100).shift(100)) / self.close.shift(100) <= 0.05
        left_value = -(self.close - self.close.rolling(100).min())
        right_value = -(self.close - self.close.shift(3))
        right_value[cond] = left_value[cond]
        alpha = right_value.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_099(self):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))) #
        alpha = (-pd.rolling_cov(self.close.rank(axis=1, pct=True), self.volume.rank(axis=1, pct=True), window=5).rank(
            axis=1, pct=True)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_100(self):
        # STD(VOLUME,20) #
        alpha = self.volume.iloc[-20:, :].std()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_101(self):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1) #
        rank1 = (
            self.close.rolling(window=15).corr((self.volume.rolling(window=30).mean()).rolling(window=37).sum())).rank(
            axis=1, pct=True)
        rank2 = (self.high * 0.1 + self.avg_price * 0.9).rank(axis=1, pct=True)
        rank3 = self.volume.rank(axis=1, pct=True)
        rank4 = (rank2.rolling(window=11).corr(rank3)).rank(axis=1, pct=True)
        alpha = -(rank1 < rank4)
        alpha = alpha.iloc[-1, :].dropna()
        return alpha

    #
    def alpha_102(self):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100 #
        max_cond = (self.volume - self.volume.shift()) > 0
        max_data = self.volume - self.volume.shift()
        max_data[~max_cond] = 0
        sma1 = pd.DataFrame.ewm(max_data, span=11, adjust=False).mean()
        sma2 = pd.DataFrame.ewm((self.volume - self.volume.shift()).abs(), span=11, adjust=False).mean()
        alpha = (sma1 / sma2 * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_103(self):
        ##### ((20-LOWDAY(LOW,20))/20)*100
        ##
        alpha = (20 - self.low.iloc[-20:, :].apply(self.func_lowday)) / 20 * 100
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_104(self):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))) #
        corr = self.high.rolling(window=5).corr(self.volume)
        alpha = (-(corr - corr.shift(5)) * ((self.close.rolling(window=20).std()).rank(axis=1, pct=True))).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_105(self):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10)) #
        alpha = -((self.open_price.rank(axis=1, pct=True)).iloc[-10:, :]).corrwith(
            self.volume.iloc[-10:, :].rank(axis=1, pct=True))
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_106(self):
        # CLOSE-DELAY(CLOSE,20) #
        alpha = (self.close - self.close.shift(20)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_107(self):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1)))) #
        rank1 = -(self.open_price - self.high.shift()).rank(axis=1, pct=True)
        rank2 = (self.open_price - self.close.shift()).rank(axis=1, pct=True)
        rank3 = (self.open_price - self.low.shift()).rank(axis=1, pct=True)
        alpha = (rank1 * rank2 * rank3).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_108(self):
        # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1) #
        min_cond = self.high > 2
        data = self.high
        data[min_cond] = 2
        rank1 = (self.high - data).rank(axis=1, pct=True)
        rank2 = (self.avg_price.rolling(window=6).corr(self.volume.rolling(window=120).mean())).rank(axis=1, pct=True)
        alpha = (-rank1 ** rank2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_109(self):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        data = self.high - self.low
        sma1 = pd.DataFrame.ewm(data, span=9, adjust=False).mean()
        sma2 = pd.DataFrame.ewm(sma1, span=9, adjust=False).mean()
        alpha = (sma1 / sma2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_110(self):
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100 #
        data1 = self.high - self.prev_close
        data2 = self.prev_close - self.low
        max_cond1 = data1 < 0
        max_cond2 = data2 < 0
        data1[max_cond1] = 0
        data2[max_cond2] = 0
        sum1 = data1.rolling(window=20).sum()
        sum2 = data2.rolling(window=20).sum()
        alpha = sum1 / sum2 * 100
        alpha = alpha.dropna()
        return alpha.iloc[-1, :]

    def alpha_111(self):
        # sma(vol*((close-low)-(high-close))/(high-low),11,2)-sma(vol*((close-low)-(high-close))/(high-low),4,2)
        ######################
        data1 = self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        x = pd.DataFrame.ewm(data1, span=10).mean()
        y = pd.DataFrame.ewm(data1, span=3).mean()
        alpha = (x - y).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_112(self):
        # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100 #
        cond1 = self.close > self.prev_close
        cond2 = self.close < self.prev_close
        data1 = self.close - self.prev_close
        data2 = self.close - self.prev_close
        data1[~cond1] = 0
        data2[~cond2] = 0
        data2 = data2.abs()
        sum1 = data1.rolling(window=12).sum()
        sum2 = data2.rolling(window=12).sum()
        alpha = ((sum1 - sum2) / (sum1 + sum2) * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_113(self):
        # (-1*((rank((sum(delay(close,5),20)/20))*corr(close,volume,2))*rank(corr(sum(close,5),sum(close,20),2))))
        #####################
        data1 = self.close.iloc[:-5, :]
        rank1 = (pd.rolling_sum(data1, window=20) / 20).rank(axis=1, pct=True)
        corr1 = self.close.iloc[-2:, :].corrwith(self.volume.iloc[-2:, :])
        data2 = pd.rolling_sum(self.close, window=5)
        data3 = pd.rolling_sum(self.close, window=20)
        corr2 = data2.iloc[-2:, :].corrwith(data3.iloc[-2:, :])
        rank2 = corr2.rank(axis=0, pct=True)
        alpha = (-1 * rank1 * corr1 * rank2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_114(self):
        # ((rank(delay(((high-low)/(sum(close,5)/5)),2))*rank(rank(volume)))/(((high-low)/(sum(close,5)/5))/(vwap-close)))
        #####################
        data1 = (self.high - self.low) / (pd.rolling_sum(self.close, window=5) / 5)
        rank1 = (data1.iloc[-2, :]).rank(axis=0, pct=True)
        rank2 = ((self.volume.rank(axis=1, pct=True)).rank(axis=1, pct=True)).iloc[-1, :]
        data2 = (((self.high - self.low) / (pd.rolling_sum(self.close, window=5) / 5)) / (
                self.avg_price - self.close)).iloc[-1, :]
        alpha = (rank1 * rank2) / data2
        alpha = alpha.dropna()
        return alpha

        #

    def alpha_115(self):
        # RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)) #
        data1 = (self.high * 0.9 + self.close * 0.1)
        data2 = self.volume.rolling(window=30).mean()
        rank1 = (data1.iloc[-10:, :].corrwith(data2.iloc[-10:, :])).rank(pct=True)
        tsrank1 = pd.rolling_apply((self.high + self.low) / 2, 4, self.func_rank)
        tsrank2 = pd.rolling_apply(self.volume, 10, self.func_rank)
        rank2 = tsrank1.iloc[-7:, :].corrwith(tsrank2.iloc[-7:, :]).rank(pct=True)
        alpha = rank1 ** rank2
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_116(self):
        # REGBETA(CLOSE,SEQUENCE,20) #
        sequence = pd.Series(range(1, 21), index=self.close.iloc[-20:, ].index)  # 1~20
        corr = self.close.iloc[-20:, :].corrwith(sequence)
        alpha = corr
        alpha = alpha.dropna()
        return alpha

    def alpha_117(self):
        #######((tsrank(volume,32)*(1-tsrank(((close+high)-low),16)))*(1-tsrank(ret,32)))
        ####################
        data1 = (self.close + self.high - self.low).iloc[-16:, :]
        data2 = 1 - data1.rank(axis=0, pct=True)
        data3 = (self.volume.iloc[-32:, :]).rank(axis=0, pct=True)
        ret = (self.close / self.close.shift() - 1).iloc[-32:, :]
        data4 = 1 - ret.rank(axis=0, pct=True)
        alpha = (data2.iloc[-1, :]) * (data3.iloc[-1, :]) * (data4.iloc[-1, :])
        alpha = alpha.dropna()
        return alpha

    def alpha_118(self):
        ######sum(high-open,20)/sum((open-low),20)*100
        ###################
        data1 = self.high - self.open_price
        data2 = self.open_price - self.low
        # data3 = pd.rolling_sum(data1, window=20)
        data3 = data1.rolling(20).sum()
        # data4 = pd.rolling_sum(data2, window=20)
        data4 = data2.rolling(20).sum()
        alpha = ((data3 / data4) * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_119(self):
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        sum1 = (self.volume.rolling(window=5).mean()).rolling(window=26).sum()
        corr1 = self.avg_price.rolling(window=5).corr(sum1)
        rank1 = pd.rolling_apply(corr1, 7, self.func_decaylinear).rank(axis=1, pct=True)
        rank2 = self.open_price.rank(axis=1, pct=True)
        rank3 = (self.volume.rolling(window=15).mean()).rank(axis=1, pct=True)
        rank4 = pd.rolling_apply(rank2.rolling(window=21).corr(rank3).rolling(window=9).min(), 7, self.func_rank)
        rank5 = pd.rolling_apply(rank4, 8, self.func_decaylinear).rank(axis=1, pct=True)
        alpha = (rank1 - rank5).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_120(self):
        ###############(rank(vwap-close))/(rank(vwap+close))
        ###################
        data1 = (self.avg_price - self.close).rank(axis=1, pct=True)
        data2 = (self.avg_price + self.close).rank(axis=1, pct=True)
        alpha = (data1 / data2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_121(self):

        return 0

    def alpha_122(self):
        ##### (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))
        ##### / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        ##
        log_close = np.log(self.close)
        data = pd.DataFrame.ewm(
            pd.DataFrame.ewm(pd.DataFrame.ewm(log_close, span=12, adjust=False).mean(), span=12, adjust=False).mean(),
            span=12, adjust=False).mean()
        alpha = (data.iloc[-1, :] / data.iloc[-2, :]) - 1
        alpha = alpha.dropna()
        return alpha

    def alpha_123(self):
        #####((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        ##
        data1 = ((self.high + self.low) / 2).rolling(20).sum()
        data2 = self.volume.rolling(60).mean().rolling(20).sum()
        rank1 = data1.iloc[-9:, :].corrwith(data2.iloc[-9:, :]).dropna().rank(axis=0, pct=True)
        rank2 = self.low.iloc[-6:, :].corrwith(self.volume.iloc[-6:, :]).dropna().rank(axis=0, pct=True)
        rank1 = rank1[rank1.index.isin(rank2.index)]
        rank2 = rank2[rank2.index.isin(rank1.index)]
        alpha = (rank1 < rank2) * (-1)
        alpha = alpha.dropna()
        return alpha

    def alpha_124(self):
        ##### (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        ##
        data1 = self.close.rolling(30).max().rank(axis=1, pct=True)
        alpha = (self.close.iloc[-1, :] - self.avg_price.iloc[-1, :]) / (
                2. / 3 * data1.iloc[-2, :] + 1. / 3 * data1.iloc[-1, :])
        alpha = alpha.dropna()
        return alpha

    def alpha_125(self):
        ##### (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        ##
        data1 = pd.rolling_corr(self.avg_price, self.volume.rolling(80).mean(), window=17)
        decay_weights = np.arange(1, 21, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data2 = (self.close * 0.5 + self.avg_price * 0.5).diff(3)
        decay_weights = np.arange(1, 17, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data2.iloc[-16:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = rank1 / rank2
        alpha = alpha.dropna()
        return alpha

    def alpha_126(self):
        #### (CLOSE + HIGH + LOW) / 3
        ##
        alpha = (self.close.iloc[-1, :] + self.high.iloc[-1, :] + self.low.iloc[-1, :]) / 3
        alpha = alpha.dropna()
        return alpha

    def alpha_127(self):

        return

    def alpha_128(self):

        return

    def alpha_129(self):
        #### SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        ##
        data = self.close.diff(1)
        data[data >= 0] = 0
        data = abs(data)
        alpha = data.iloc[-12:, :].sum()
        alpha = alpha.dropna()
        return alpha

    def alpha_130(self):
        #### alpha_130
        #### (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
        ##
        data1 = (self.high + self.low) / 2
        data2 = self.volume.rolling(40).mean()
        data3 = pd.rolling_corr(data1, data2, window=9)
        decay_weights = np.arange(1, 11, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data3.iloc[-10:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.avg_price.rank(axis=1, pct=True)
        data2 = self.volume.rank(axis=1, pct=True)
        data3 = pd.rolling_corr(data1, data2, window=7)
        decay_weights = np.arange(1, 4, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data3.iloc[-3:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = (rank1 / rank2).dropna()
        return alpha

    def alpha_131(self):
        return 0

    def alpha_132(self):
        #### MEAN(AMOUNT, 20)
        ##
        alpha = self.amount.iloc[-20:, :].mean()
        alpha = alpha.dropna()
        return alpha

    def alpha_133(self):
        #### alpha_133
        #### ((20 - HIGHDAY(HIGH, 20)) / 20)*100 - ((20 - LOWDAY(LOW, 20)) / 20)*100
        ##

        alpha = (20 - self.high.iloc[-20:, :].apply(self.func_highday)) / 20 * 100 \
                - (20 - self.low.iloc[-20:, :].apply(self.func_lowday)) / 20 * 100
        alpha = alpha.dropna()
        return alpha

    def alpha_134(self):
        #### (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        ##
        alpha = ((self.close.iloc[-1, :] / self.close.iloc[-13, :] - 1) * self.volume.iloc[-1, :])
        alpha = alpha.dropna()
        return alpha

    def alpha_135(self):
        #### SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        ##
        def rolling_div(na):
            return na[-1] / na[-21]

        data1 = self.close.rolling(21).apply(rolling_div).shift(periods=1)
        alpha = pd.DataFrame.ewm(data1, com=19, adjust=False).mean().iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_136(self):
        #### ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        ##
        data1 = -(self.close / self.prev_close - 1).diff(3).rank(axis=1, pct=True)
        data2 = self.open_price.iloc[-10:, :].corrwith(self.volume.iloc[-10:, :])
        alpha = (data1.iloc[-1, :] * data2).dropna()

        return alpha

    def alpha_137(self):

        return

    def alpha_138(self):
        #### ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
        ##
        data1 = (self.low * 0.7 + self.avg_price * 0.3).diff(3)
        decay_weights = np.arange(1, 21, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.low.rolling(8).apply(self.func_rank)
        data2 = self.volume.rolling(60).mean().rolling(17).apply(self.func_rank)
        data3 = pd.rolling_corr(data1, data2, window=5).rolling(19).apply(self.func_rank)
        rank2 = data3.rolling(16).apply(self.func_decaylinear).iloc[-7:, :].rank(axis=0, pct=True).iloc[-1, :]

        alpha = (rank2 - rank1).dropna()
        return alpha

    def alpha_139(self):
        #### (-1 * CORR(OPEN, VOLUME, 10))
        ##
        alpha = - self.open_price.iloc[-10:, :].corrwith(self.volume.iloc[-10:, :]).dropna()
        return alpha

    def alpha_140(self):
        #### MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
        ##
        data1 = self.open_price.rank(axis=1, pct=True) + self.low.rank(axis=1, pct=True) \
                - self.high.rank(axis=1, pct=True) - self.close.rank(axis=1, pct=True)
        rank1 = data1.iloc[-8:, :].apply(self.func_decaylinear).rank(pct=True)

        data1 = self.close.rolling(8).apply(self.func_rank)
        data2 = self.volume.rolling(60).mean().rolling(20).apply(self.func_rank)
        data3 = pd.rolling_corr(data1, data2, window=8)
        data3 = data3.rolling(7).apply(self.func_decaylinear)
        rank2 = data3.iloc[-3:, :].rank(axis=0, pct=True).iloc[-1, :]

        '''
        alpha = min(rank1, rank2)   NaN如何比较？
        '''
        return alpha

    def alpha_141(self):
        #### (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        ##
        df1 = self.high.rank(axis=1, pct=True)
        df2 = self.volume.rolling(15).mean().rank(axis=1, pct=True)
        alpha = -df1.iloc[-9:, :].corrwith(df2.iloc[-9:, :]).rank(pct=True)
        alpha = alpha.dropna()
        return alpha

    def alpha_142(self):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        ##

        rank1 = self.close.iloc[-10:, :].rank(axis=0, pct=True).iloc[-1, :].rank(pct=True)
        rank2 = self.close.diff(1).diff(1).iloc[-1, :].rank(pct=True)
        rank3 = (self.volume / self.volume.rolling(20).mean()).iloc[-5:, :].rank(axis=0, pct=True).iloc[-1, :].rank(
            pct=True)

        alpha = -(rank1 * rank2 * rank3).dropna()
        alpha = alpha.dropna()
        return alpha

    def alpha_143(self):
        #### CLOSE > DELAY(CLOSE, 1)?(CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1) * SELF : SELF

        return 0

    def alpha_144(self):
        #### SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        ##
        df1 = self.close < self.prev_close
        sumif = ((abs(self.close / self.prev_close - 1) / self.amount) * df1).iloc[-20:, :].sum()
        count = df1.iloc[-20:, :].sum()

        alpha = (sumif / count).dropna()
        alpha = alpha.dropna()
        return alpha

    def alpha_145(self):
        #### (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        ##

        alpha = (self.volume.iloc[-9:, :].mean() - self.volume.iloc[-26:, :].mean()) / self.volume.iloc[-12:,
                                                                                       :].mean() * 100
        alpha = alpha.dropna()
        return alpha

    def alpha_146(self):

        return

    def alpha_147(self):

        return

    def alpha_148(self):
        #### ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        ##
        df1 = self.volume.rolling(60).mean().rolling(9).sum()
        rank1 = self.open_price.iloc[-6:, :].corrwith(df1.iloc[-6:, :]).rank(pct=True)
        rank2 = (self.open_price - self.open_price.rolling(14).min()).iloc[-1, :].rank(pct=True)

        alpha = -1 * (rank1 < rank2)
        alpha = alpha.dropna()
        return alpha

    def alpha_149(self):

        return

    def alpha_150(self):
        #### (CLOSE + HIGH + LOW)/3 * VOLUME
        ##

        alpha = ((self.close + self.high + self.low) / 3 * self.volume).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_151(self):

        return 0

    ######################## alpha_152 #######################
    #
    def alpha_152(self):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1) #
        #
        data1 = pd.rolling_mean((pd.ewma(((self.close / self.close.shift(9)).shift()), span=17, adjust=False)).shift(),
                                12)
        data2 = pd.rolling_mean((pd.ewma(((self.close / self.close.shift(9)).shift()), span=17, adjust=False)).shift(),
                                26)
        alpha = (pd.ewma(data1 - data2, span=17, adjust=False)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_153 #######################
    #
    def alpha_153(self):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 #
        alpha = ((pd.rolling_mean(self.close, 3) + pd.rolling_mean(self.close, 6) + pd.rolling_mean(self.close,
                                                                                                    12) + pd.rolling_mean(
            self.close, 24)) / 4).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_154 #######################
    #
    def alpha_154(self):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18))) #
        alpha = (self.avg_price - pd.rolling_min(self.avg_price, 16)).iloc[-1, :] < self.avg_price.iloc[-18:,
                                                                                    :].corrwith(
            (pd.rolling_mean(self.volume, 180)).iloc[-18:, :])
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_155 #######################
    #
    def alpha_155(self):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2) #
        sma1 = pd.ewma(self.volume, span=12, adjust=False)
        sma2 = pd.ewma(self.volume, span=26, adjust=False)
        sma = pd.ewma(sma1 - sma2, span=9, adjust=False)
        alpha = sma.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_156 #######################
    def alpha_156(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1 #
        rank1 = (pd.rolling_apply(self.avg_price - self.avg_price.shift(5), 3, self.func_decaylinear)).rank(axis=1,
                                                                                                            pct=True)
        rank2 = pd.rolling_apply(
            -((self.open_price * 0.15 + self.low * 0.85) - (self.open_price * 0.15 + self.low * 0.85).shift(2)) / (
                    self.open_price * 0.15 + self.low * 0.85), 3, self.func_decaylinear).rank(axis=1, pct=True)
        max_cond = rank1 > rank2
        result = rank2
        result[max_cond] = rank1[max_cond]
        alpha = (-result).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_157 #######################
    #
    def alpha_157(self):
        # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5)) #
        rank1 = (-((self.close - 1) - (self.close - 1).shift(5)).rank(axis=1, pct=True)).rank(axis=1, pct=True).rank(
            axis=1, pct=True)
        min1 = rank1.rolling(2).min()
        log1 = np.log(min1)
        rank2 = log1.rank(axis=1, pct=True).rank(axis=1, pct=True)
        cond_min = rank2 > 5
        rank2[cond_min] = 5
        tsrank1 = pd.rolling_apply((-((self.close / self.prev_close) - 1)).shift(6), 5, self.func_rank)
        alpha = (rank2 + tsrank1).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_158 #######################
    #
    def alpha_158(self):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        alpha = (((self.high - pd.ewma(self.close, span=14, adjust=False)) - (
                self.low - pd.ewma(self.close, span=14, adjust=False))) / self.close).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_159(self):
        #########((close-sum(min(low,delay(close,1)),6))/sum(max(high,delay(close,1))-min(low,delay(close,1)),6)*12*24+(close-sum(min(low,delay(close,1)),12))/sum(max(high,delay(close,1))-min(low,delay(close,1)),12)*6*24+(close-sum(min(low,delay(close,1)),24))/sum(max(high,delay(close,1))-min(low,delay(close,1)),24)*6*24)*100/(6*12+6*24+12*24)
        ###################
        data1 = self.low
        data2 = self.close.shift()
        cond = data1 > data2
        data1[cond] = data2
        data3 = self.high
        data4 = self.close.shift()
        cond = data3 > data4
        data3[~cond] = data4
        # 计算出公式核心部分x
        x = ((self.close - pd.rolling_sum(data1, 6)) / pd.rolling_sum((data2 - data1), 6)) * 12 * 24
        # 计算出公式核心部分y
        y = ((self.close - pd.rolling_sum(data1, 12)) / pd.rolling_sum((data2 - data1), 12)) * 6 * 24
        # 计算出公式核心部分z
        z = ((self.close - pd.rolling_sum(data1, 24)) / pd.rolling_sum((data2 - data1), 24)) * 6 * 24
        data5 = (x + y + z) * (100 / (6 * 12 + 12 * 24 + 6 * 24))
        alpha = data5.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_160(self):
        ################
        ############sma((close<=delay(close,1)?std(close,20):0),20,1)
        data1 = pd.rolling_std(self.close, 20)
        cond = self.close <= self.close.shift(0)
        data1[~cond] = 0
        data2 = pd.ewma(data1, span=39)
        alpha = data2.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_161(self):
        ###########mean((max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),12)
        ################
        data1 = (self.high - self.low)
        data2 = pd.Series.abs(self.close.shift() - self.high)
        cond = data1 > data2
        data1[~cond] = data2
        data3 = pd.Series.abs(self.close.shift() - self.low)
        cond = data1 > data3
        data1[~cond] = data3
        alpha = (pd.rolling_mean(data1, 12)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_162(self):
        ###############(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100,12))/(max(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12)-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12))
        #################
        # 算出公式核心部分X
        data1 = self.close - self.close.shift()
        cond = data1 > 0
        data1[~cond] = 0
        x = pd.ewma(data1, span=23)
        data2 = pd.Series.abs(self.close - self.close.shift())
        y = pd.ewma(data2, span=23)
        z = (x / y) * 100
        cond = z > 12
        z[cond] = 12
        c = (x / y) * 100
        cond = c > 12
        c[~cond] = 12
        data3 = (x / y) * 100 - (z / c) - c
        alpha = data3.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_163(self):
        ################
        #######rank(((((-1*ret)*,ean(volume,20))*vwap)*(high-close)))
        data1 = (-1) * (self.close / self.close.shift() - 1) * pd.rolling_mean(self.volume, 20) * self.avg_price * (
                self.high - self.close)
        data2 = (data1.rank(axis=1, pct=True)).iloc[-1, :]
        alpha = data2
        alpha = alpha.dropna()
        return alpha

    def alpha_164(self):
        ################
        ############sma((((close>delay(close,1))?1/(close-delay(close,1)):1)-min(((close>delay(close,1))?1/(close/delay(close,1)):1),12))/(high-low)*100,13,2)
        cond = self.close > self.close.shift()
        data1 = 1 / (self.close - self.close.shift())
        data1[~cond] = 1
        data2 = 1 / (self.close - self.close.shift())
        cond = data2 > 12
        data2[cond] = 12
        data3 = data1 - data2 / ((self.high - self.low) * 100)
        alpha = (pd.ewma(data3, span=12)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_165(self):

        return 0

    def alpha_166(self):

        return 0

    def alpha_167(self):
        ##
        ####sum(((close-delay(close,1)>0)?(close-delay(close,1)):0),12)####
        data1 = self.close - self.close.shift()
        cond = (data1 < 0)
        data1[cond] = 0
        data2 = (pd.rolling_sum(data1, 12)).iloc[-1, :]
        alpha = data2
        alpha = alpha.dropna()
        return alpha

    def alpha_168(self):
        ##
        #####-1*volume/mean(volume,20)####
        data1 = (-1 * self.volume) / pd.rolling_mean(self.volume, 20)
        alpha = data1.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_169(self):
        ##
        ###sma(mean(delay(sma(close-delay(close,1),9,1),1),12)-mean(delay(sma(close-delay(close,1),1,1),1),26),10,1)#####
        data1 = self.close - self.close.shift()
        data2 = (pd.ewma(data1, span=17)).shift()
        data3 = pd.rolling_mean(data2, 12) - pd.rolling_mean(data2, 26)
        alpha = (pd.ewma(data3, span=19)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_170(self):
        ##
        #####((((rank((1/close))*volume)/mean(volume,20))*((high*rank((high-close)))/(sum(high,5)/5)))-rank((vwap-delay(vwap,5))))####
        data1 = (1 / self.close).rank(axis=0, pct=True)
        data2 = pd.rolling_mean(self.volume, 20)
        x = (data1 * self.volume) / data2
        data3 = (self.high - self.close).rank(axis=0, pct=True)
        data4 = pd.rolling_mean(self.high, 5)
        y = (data3 * self.high) / data4
        z = (self.avg_price.iloc[-1, :] - self.avg_price.iloc[-5, :]).rank(axis=0, pct=True)
        alpha = (x * y - z).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_171(self):
        ##
        ####(((low-close)*open^5)*-1)/((close-high)*close^5)#####
        data1 = -1 * (self.low - self.close) * (self.open_price ** 5)
        data2 = (self.close - self.high) * (self.close ** 5)
        alpha = (data1 / data2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_172(self):
        # MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6) #
        hd = self.high - self.high.shift()
        ld = self.low.shift() - self.low
        temp1 = self.high - self.low
        temp2 = (self.high - self.close.shift()).abs()
        cond1 = temp1 > temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low - self.close.shift()).abs()
        cond2 = temp2 > temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3  # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = pd.rolling_sum(tr, 14)
        cond3 = ld > 0
        cond4 = ld > hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = pd.rolling_sum(data1, 14) * 100 / sum_tr14
        cond5 = hd > 0
        cond6 = hd > ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = pd.rolling_sum(data2, 14) * 100 / sum_tr14
        alpha = pd.rolling_mean((sum1 - sum2).abs() / (sum1 + sum2) * 100, 6).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_173(self):
        ##
        ####3*sma(close,13,2)-2*sma(sma(close,13,2),13,2)+sma(sma(sma(log(close),13,2),13,2),13,2)#####
        data1 = pd.ewma(self.close, span=12)
        data2 = pd.ewma(data1, span=12)
        close_log = np.log(self.close)
        data3 = pd.ewma(close_log, span=12)
        data4 = pd.ewma(data3, span=12)
        data5 = pd.ewma(data4, span=12)
        alpha = (3 * data1 - 2 * data2 + data5).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_174(self):
        ##
        ####sma((close>delay(close,1)?std(close,20):0),20,1)#####
        cond = self.close > self.prev_close
        data2 = pd.rolling_std(self.close, 20)
        data2[~cond] = 0
        alpha = (pd.ewma(data2, span=39, adjust=False)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_175(self):
        ##
        #####mean(max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),6)####
        data1 = self.high - self.low
        data2 = pd.Series.abs(self.close.shift() - self.high)
        cond = (data1 > data2)
        data2[cond] = data1[cond]
        data3 = pd.Series.abs(self.close.shift() - self.low)
        cond = (data2 > data3)
        data3[cond] = data2[cond]
        data4 = (pd.rolling_mean(data3, window=6)).iloc[-1, :]
        alpha = data4
        alpha = alpha.dropna()
        return alpha

    def alpha_176(self):
        ##
        ######### #########corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))),rank(volume),6)#############
        data1 = (self.close - pd.rolling_min(self.low, window=12)) / (
                pd.rolling_max(self.high, window=12) - pd.rolling_min(self.low, window=12))
        data2 = data1.rank(axis=0, pct=True)
        # 获取数据求出rank2
        data3 = self.volume.rank(axis=0, pct=True)
        corr = data2.iloc[-6:, :].corrwith(data3.iloc[-6:, :])
        alpha = corr
        alpha = alpha.dropna()
        return alpha

    ################## alpha_177 ####################
    #
    def alpha_177(self):
        ##### ((20-HIGHDAY(HIGH,20))/20)*100 #####
        alpha = (20 - self.high.iloc[-20:, :].apply(self.func_highday)) / 20 * 100
        alpha = alpha.dropna()
        return alpha

    def alpha_178(self):
        ##### (close-delay(close,1))/delay(close,1)*volume ####
        ##
        alpha = ((self.close - self.close.shift()) / self.close.shift() * self.volume).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_179(self):
        #####（rank(corr(vwap,volume,4))*rank(corr(rank(low),rank(mean(volume,50)),12))####
        ##
        rank1 = (self.avg_price.iloc[-4:, :].corrwith(self.volume.iloc[-4:, :])).rank(axis=0, pct=True)
        data2 = self.low.rank(axis=0, pct=True)
        data3 = (pd.rolling_mean(self.volume, window=50)).rank(axis=0, pct=True)
        rank2 = (data2.iloc[-12:, :].corrwith(data3.iloc[-12:, :])).rank(axis=0, pct=True)
        alpha = rank1 * rank2
        alpha = alpha.dropna()
        return alpha

        ##################### alpha_180 #######################

    #
    def alpha_180(self):
        ##### ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))) #####
        ma = pd.rolling_mean(self.volume, window=20)
        cond = (ma < self.volume).iloc[-20:, :]
        sign = delta_close_7 = self.close.diff(7)
        sign[sign.iloc[:, :] < 0] = -1
        sign[sign.iloc[:, :] > 0] = 1
        sign[sign.iloc[:, :] == 0] = 0
        left = (((self.close.diff(7).abs()).iloc[-60:, :].rank(axis=0, pct=True) * (-1)).iloc[-20:, :] * sign.iloc[-20:,
                                                                                                         :]).iloc[-20:,
               :]
        right = self.volume.iloc[-20:, :] * (-1)
        right[cond] = left[cond]
        alpha = right.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_181(self):

        return 0

    ######################## alpha_182 #######################
    #
    def count_cond_182(self, x):
        num = 0
        for i in x:
            if i == np.True_:
                num += 1
        return num

    def alpha_182(self):
        ##### COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20 #####
        cond1 = (self.close > self.open_price)
        cond2 = (self.benchmark_open_price > self.benchmark_close_price)
        cond3 = (self.close < self.open_price)
        cond4 = (self.benchmark_open_price < self.benchmark_close_price)
        func1 = lambda x: np.asarray(x) & np.asarray(cond2)
        func2 = lambda x: np.asarray(x) & np.asarray(cond4)
        cond = cond1.apply(func1) | cond3.apply(func2)
        count = pd.rolling_apply(cond, 20, self.count_cond_182)
        alpha = (count / 20).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_183(self):

        return 0

    def alpha_184(self):
        #####(rank(corr(delay((open-close),1),close,200))+rank((open-close))) ####
        ##
        data1 = self.open_price.shift() - self.close.shift()
        data2 = self.open_price.iloc[-1, :] - self.close.iloc[-1, :]
        corr = data1.iloc[-200:, :].corrwith(self.close.iloc[-200:, :])
        alpha = data2.rank(axis=0, pct=True) + corr.rank(axis=0, pct=True)
        alpha = alpha.dropna()
        return alpha

    def alpha_185(self):
        ##### RANK((-1 * ((1 - (OPEN / CLOSE))^2))) ####
        alpha = (-(1 - self.open_price / self.close) ** 2).rank(axis=1, pct=True).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_186(self):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #
        hd = self.high - self.high.shift()
        ld = self.low.shift() - self.low
        temp1 = self.high - self.low
        temp2 = (self.high - self.close.shift()).abs()
        cond1 = temp1 > temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low - self.close.shift()).abs()
        cond2 = temp2 > temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3  # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = pd.rolling_sum(tr, 14)
        cond3 = ld > 0
        cond4 = ld > hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = pd.rolling_sum(data1, 14) * 100 / sum_tr14
        cond5 = hd > 0
        cond6 = hd > ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = pd.rolling_sum(data2, 14) * 100 / sum_tr14
        mean1 = pd.rolling_mean((sum1 - sum2).abs() / (sum1 + sum2) * 100, 6)
        alpha = ((mean1 + mean1.shift(6)) / 2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_187(self):
        ##### SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20) ####

        cond = (self.open_price <= self.open_price.shift())
        data1 = self.high - self.low  # HIGH-LOW
        data2 = self.open_price - self.open_price.shift()  # OPEN-DELAY(OPEN,1)
        cond_max = data2 > data1
        data1[cond_max] = data2[cond_max]
        data1[cond] = 0
        alpha = data1.iloc[-20:, :].sum()
        alpha = alpha.dropna()
        return alpha

    def alpha_188(self):
        ##### ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100 #####

        sma = pd.ewma(self.high - self.low, span=10, adjust=False)
        alpha = ((self.high - self.low - sma) / sma * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_189(self):
        ##### mean(abs(close-mean(close,6),6)) ####
        ma6 = pd.rolling_mean(self.close, window=6)
        alpha = pd.rolling_mean((self.close - ma6).abs(), window=6).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_190(self):
        ##### LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)
        ##### -1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((
        ##### COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE
        ##### /DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))) ####

        return 0

    def alpha_191(self):
        ##### (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####

        # volume_avg = pd.rolling_mean(self.volume, window=20)
        volume_avg = self.volume.rolling(window=20).mean()
        corr = volume_avg.iloc[-5:, :].corrwith(self.low.iloc[-5:, :])
        alpha = corr + (self.high.iloc[-1, :] + self.low.iloc[-1, :]) / 2 - self.close.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha


def get_func_map(a):
    return {
        "alpha_002": a.alpha_002,
        "alpha_003": a.alpha_003,
        "alpha_004": a.alpha_004,
        "alpha_005": a.alpha_005,
        "alpha_009": a.alpha_009,
        "alpha_011": a.alpha_011,
        "alpha_014": a.alpha_014,
        "alpha_018": a.alpha_018,
        "alpha_019": a.alpha_019,
        "alpha_020": a.alpha_020,
        "alpha_022": a.alpha_022,
        "alpha_023": a.alpha_023,
        "alpha_024": a.alpha_024,
        "alpha_028": a.alpha_028,
        "alpha_029": a.alpha_029,
        "alpha_031": a.alpha_031,
        "alpha_034": a.alpha_034,
        "alpha_038": a.alpha_038,
        "alpha_040": a.alpha_040,
        "alpha_042": a.alpha_042,
        "alpha_043": a.alpha_043,
        "alpha_046": a.alpha_046,
        "alpha_047": a.alpha_047,
        "alpha_049": a.alpha_049,
        "alpha_052": a.alpha_052,
        "alpha_053": a.alpha_053,
        "alpha_054": a.alpha_054,
        "alpha_057": a.alpha_057,
        "alpha_058": a.alpha_058,
        "alpha_059": a.alpha_059,
        "alpha_060": a.alpha_060,
        "alpha_063": a.alpha_063,
        "alpha_065": a.alpha_065,
        "alpha_066": a.alpha_066,
        "alpha_067": a.alpha_067,
        "alpha_068": a.alpha_068,
        "alpha_071": a.alpha_071,
        "alpha_072": a.alpha_072,
        "alpha_076": a.alpha_076,
        "alpha_078": a.alpha_078,
        "alpha_079": a.alpha_079,
        "alpha_080": a.alpha_080,
        "alpha_081": a.alpha_081,
        "alpha_082": a.alpha_082,
        "alpha_084": a.alpha_084,
        "alpha_086": a.alpha_086,
        "alpha_088": a.alpha_088,
        "alpha_089": a.alpha_089,
        "alpha_093": a.alpha_093,
        "alpha_096": a.alpha_096,
        "alpha_097": a.alpha_097,
        "alpha_098": a.alpha_098,
        "alpha_100": a.alpha_100,
        "alpha_102": a.alpha_102,
        "alpha_103": a.alpha_103,
        "alpha_106": a.alpha_106,
        "alpha_109": a.alpha_109,
        "alpha_111": a.alpha_111,
        "alpha_116": a.alpha_116,
        "alpha_118": a.alpha_118,
        "alpha_126": a.alpha_126,
        "alpha_129": a.alpha_129,
        "alpha_133": a.alpha_133,
        "alpha_134": a.alpha_134,
        "alpha_139": a.alpha_139,
        "alpha_145": a.alpha_145,
        "alpha_150": a.alpha_150,
    }


def _run_symbol(symbol_list, process_id):
    for symbol in symbol_list:
        print(symbol)
        print("process_id", process_id)
        try:
            columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'outstanding_share', 'turnover']
            # symbol = "sz000001"
            table_name = "stock_zh_a_daily"
            search_sql = "where `symbol` = '{}'".format(symbol)
            sql = " SELECT %s FROM `%s` %s" % (','.join(columns),
                                               table_name, search_sql)
            result = common.select(sql)
            df = pd.DataFrame(list(result), columns=columns)

            rslt = pd.DataFrame()
            for i, rows in df.iterrows():
                date = rows.date
                cal_df = df[:int(i) + 1]
                print(i)
                a = GTJA_191(cal_df, symbol)
                func_map = get_func_map(a)
                row_rslt = pd.DataFrame(data=[date], columns=['date'], index=[symbol])
                for name, func in func_map.items():
                    try:
                        alpha_rslt = func()
                    except Exception as e:
                        logging.exception(e)
                        continue
                    try:
                        if rslt is None:
                            alpha_rslt.name = name
                            row_rslt = alpha_rslt
                        if isinstance(alpha_rslt, int) and alpha_rslt == 0:
                            row_rslt[name] = 0
                        else:
                            alpha_rslt.name = name
                            row_rslt = pd.concat([row_rslt, alpha_rslt], axis=1)
                    except Exception as e:
                        logging.exception(e)
                rslt = pd.concat([rslt, row_rslt], axis=0)
            rslt["symbol"] = symbol

            common.insert_db(rslt, "stock_alpha_191", True, "`id`")
        except Exception as e:
            logging.exception(e)
            continue


if __name__ == "__main__":

    # df = pd.DataFrame({"open":{"1662076800000":9.17,"1662336000000":9.21,"1662422400000":9.17,"1662508800000":9.2,"1662595200000":9.18,"1662681600000":9.08,"1663027200000":9.18},"close":{"1662076800000":9.26,"1662336000000":9.16,"1662422400000":9.18,"1662508800000":9.19,"1662595200000":9.04,"1662681600000":9.18,"1663027200000":9.19},"low":{"1662076800000":9.12,"1662336000000":9.08,"1662422400000":9.12,"1662508800000":9.15,"1662595200000":9.02,"1662681600000":9.0,"1663027200000":9.16},"high":{"1662076800000":9.33,"1662336000000":9.23,"1662422400000":9.23,"1662508800000":9.28,"1662595200000":9.18,"1662681600000":9.22,"1663027200000":9.27},"avg":{"1662076800000":9.217,"1662336000000":9.122,"1662422400000":9.163,"1662508800000":9.212,"1662595200000":9.06,"1662681600000":9.117,"1663027200000":9.194},"volume":{"1662076800000":16662399.0,"1662336000000":16401963.0,"1662422400000":14872417.0,"1662508800000":17890802.0,"1662595200000":20588100.0,"1662681600000":17528332.0,"1663027200000":12207627.0}})
    # df["pct_chg"] = (df["close"] - df["open"]) / df["open"]

    # df.index = pd.to_datetime(df.index, unit='ms')
    # df = df.transpose()

    # auth("18983287546", "Wang921207")
    # # df = get_price('601021.XSHG', None, "2022-09-13", '1d',
    # #                                ['open', 'close', 'low', 'high', 'avg', 'volume'], count=333)
    # from service.data import df_data
    #
    # symbol = '601021.XSHG'
    # df = df_data
    #
    # df.index = pd.to_datetime(df.index, unit='ms')
    # a = GTJA_191(df, symbol)
    # x1 = a.alpha_001()
    # x5 = a.alpha_032()
    # x6 = alpha191.alpha_032(symbol, "2022-09-13")
    #
    symbol_list = ['sz002463', 'sh600498', 'sh605566', 'sz000552', 'sz000878', 'sz300342', 'sh605020', 'sh600533',
                   'sz300786', 'sh603318', 'sz000025', 'sz000565', 'sz300746', 'sz002081', 'sh600538', 'sh600336',
                   'sz300181', 'sz300370', 'sh601689', 'sz300236', 'sz002730', 'sh603345', 'sz301285', 'sz300175',
                   'sh603208', 'sh603215', 'sz002588', 'sh603037', 'sh605151', 'sz300797', 'sz300079', 'sh603089',
                   'sz300622', 'sh603380', 'sz300112', 'sz300702', 'sh600139', 'sh605507', 'sz301136', 'sz300110',
                   'sz002977', 'sh600262', 'sz002800', 'sh603883', 'sh600886', 'sz300347', 'sh600706', 'sz000863',
                   'sh600755', 'sh600851', 'sh600823', 'sz000758', 'sz301226', 'sz000510', 'sh603737', 'sh600479',
                   'sz002165', 'sz002333', 'sz003022', 'sz003033', 'sz300707', 'sh600400', 'sz300446', 'sz000595',
                   'sh603929', 'sh600703', 'sz300903', 'sz002750', 'sh600143', 'sh603001', 'sz300760', 'sz000652',
                   'sz300679', 'sz300827', 'sz301049', 'sz002089', 'sz002984', 'sz300092', 'sz002793', 'sz002662',
                   'sh601169', 'sz001896', 'sz002199', 'sz000096', 'sz000068', 'sh600557', 'sh600610', 'sz000028',
                   'sz002501', 'sz000639', 'sz301032', 'sz301221', 'sh603680', 'sh601958', 'sz300879', 'sh600872',
                   'sz002488', 'sh603039', 'sh603015', 'sz002362', 'sh601218', 'sz301122', 'sz300154', 'sh600125',
                   'sz300650', 'sh600372', 'sz300692', 'sz300230', 'sh603977', 'sz300509', 'sz301018', 'sh600780',
                   'sz000892', 'sz000528', 'sz300368', 'sz002677', 'sh600710', 'sh603187', 'sz002785', 'sz301156',
                   'sz002338', 'sh601088', 'sh600893', 'sz000582', 'sh600909', 'sz300869', 'sh600795', 'sz300297',
                   'sz002258', 'sh600754', 'sh603939', 'sz300296', 'sz002589', 'sh601988', 'sh600563', 'sz000887',
                   'sh600370', 'sz002534', 'sh601963', 'sz002160', 'sz000638', 'sz300796', 'sz002572', 'sz002357',
                   'sz300856', 'sz002820', 'sh600836', 'sz300416', 'sz002806', 'sh605055', 'sz301270', 'sz002687',
                   'sh600361', 'sz002133', 'sz000650', 'sh601965', 'sz000995', 'sh600251', 'sh605196', 'sh603535',
                   'sh603530', 'sz301199', 'sh600419', 'sh600968', 'sz000970', 'sz002111', 'sz300031', 'sz000860',
                   'sh600378', 'sz300652', 'sz300694', 'sh600609', 'sh605016', 'sh603529', 'sz300247', 'sz002340',
                   'sz002943', 'sh601606', 'sh600486', 'sz002374', 'sz002605', 'sz003020', 'sh603150', 'sz000548',
                   'sz301152', 'sz001222', 'sh600773', 'sh603088', 'sz300975', 'sz301050', 'sz300660', 'sz300195',
                   'sz000738', 'sz002805', 'sz300203', 'sh603886', 'sh603016', 'sh600684', 'sz002248', 'sz301009',
                   'sz301237', 'sz000812', 'sz300256', 'sz000400', 'sh600718', 'sz300440', 'sh603998', 'sz002792',
                   'sz000023', 'sh601696', 'sz300476', 'sh600774', 'sz002439', 'sz000839', 'sh605060', 'sz002120',
                   'sz002688', 'sh600184', 'sz300107', 'sz301137', 'sz000012', 'sz002560', 'sz002274', 'sh603685',
                   'sh603657', 'sz300651', 'sz300699', 'sz002941', 'sz301090', 'sz300959', 'sh601015', 'sz000859',
                   'sz300678', 'sh603960', 'sz002360', 'sz300415', 'sh603232', 'sz001309', 'sz002706', 'sz002819',
                   'sz002640', 'sz300423', 'sz300307', 'sz000157', 'sz002527', 'sz002112', 'sh603565', 'sh600481',
                   'sz002351', 'sz300752', 'sz000666', 'sh603683', 'sh600769', 'sh600302', 'sz300929', 'sh600689',
                   'sz002208', 'sz300287', 'sz300715', 'sz300805', 'sh600109', 'sz300262', 'sz300619', 'sz300808',
                   'sh600415', 'sh603025', 'sz301048', 'sz000868', 'sh600186', 'sh603603', 'sz002136', 'sz300151',
                   'sz000062', 'sz002531', 'sh603668', 'sz300029', 'sz301215', 'sh603511', 'sz300189', 'sz002326',
                   'sz002841', 'sh600526', 'sh600603', 'sh603779', 'sh603131', 'sz002637', 'sh601933', 'sz000567',
                   'sz000665', 'sh603787', 'sz000883', 'sz002069', 'sz002606', 'sh603369', 'sz002240', 'sh600698',
                   'sz002008', 'sz300680', 'sz002497', 'sh600416', 'sh603922', 'sz002547', 'sz002336', 'sz002569',
                   'sz002765', 'sz300374', 'sz300978', 'sh600421', 'sz002026', 'sh603226', 'sz300010', 'sz300473',
                   'sz300531', 'sh603906', 'sh600691', 'sz002789', 'sh603809', 'sh603999', 'sh600326', 'sh601600',
                   'sz000008', 'sh600339', 'sz002181', 'sh605111', 'sz300338', 'sz002923', 'sz000881', 'sz300133',
                   'sz300441', 'sz300273', 'sh601888', 'sh600793', 'sz000988', 'sz002337', 'sh600371', 'sh601225',
                   'sh605058', 'sz300401', 'sh603661', 'sh603687', 'sz002725', 'sh600576', 'sz300910', 'sz002787',
                   'sz002440', 'sz000703', 'sh603009', 'sz300987', 'sh603679', 'sz300665', 'sz000786', 'sz300668',
                   'sz002150', 'sz000573', 'sz300916', 'sh603017', 'sz300083', 'sz300925', 'sz300723', 'sz300961',
                   'sz000902', 'sz300188', 'sh600182', 'sh603619', 'sz000631', 'sz300911', 'sh601137', 'sz002404',
                   'sz000723', 'sz002229', 'sz002708', 'sz000905', 'sz300132', 'sh605286', 'sz003026', 'sz000571',
                   'sz300179', 'sz300138', 'sz002347', 'sz002549', 'sh600389', 'sz300126', 'sz300456', 'sh603985',
                   'sh600097', 'sh603223', 'sz002773', 'sz000900', 'sh603912', 'sh605577', 'sz002713', 'sh603958',
                   'sz300039', 'sz002432', 'sh605066', 'sz300882', 'sz300007', 'sh601949', 'sh600746', 'sz300999',
                   'sz300053', 'sz000509', 'sh600221', 'sh600299', 'sh603871', 'sz000823', 'sz003004', 'sh601702',
                   'sz300855', 'sz002451', 'sh603233', 'sz300426', 'sz002949', 'sz300785', 'sh601228', 'sh601016',
                   'sz000755', 'sz300733', 'sz300687', 'sz300872', 'sh600848', 'sh603267', 'sh605488', 'sh603755',
                   'sh600156', 'sh603040', 'sz300298', 'sh603289', 'sz300252', 'sh600857', 'sz301041', 'sz301286',
                   'sz300602', 'sz300436', 'sz000705', 'sh600309', 'sh603898', 'sz002553', 'sz300221', 'sz300418',
                   'sz000672', 'sz301133', 'sz300085', 'sz300086', 'sh603022', 'sz300753', 'sz002654', 'sz300004',
                   'sz002135', 'sz002722', 'sh603290', 'sz000928', 'sz002516', 'sz300581', 'sz000423', 'sz300424',
                   'sz000876', 'sz300087', 'sz000655', 'sh605303', 'sh601618', 'sz300011', 'sh600561', 'sz000040',
                   'sz301268', 'sz301213', 'sz301036', 'sz002190', 'sz002170', 'sz300062', 'sh600436', 'sz301126',
                   'sh600645', 'sz002063', 'sh603110', 'sh600255', 'sz300815', 'sz300042', 'sz300631', 'sz002227',
                   'sh605369', 'sh601588', 'sh600861', 'sh603882', 'sz000797', 'sz002503', 'sh601528', 'sz002956',
                   'sz300536', 'sh600287', 'sh600311', 'sz003003', 'sz000727', 'sz300228', 'sh603869', 'sz002078',
                   'sz000416', 'sz300294', 'sh603393', 'sz002917', 'sh600300', 'sz300472', 'sz300765', 'sz002659',
                   'sh605122', 'sh600798', 'sz300246', 'sz000967', 'sz002430', 'sh603567', 'sz002004', 'sz300174',
                   'sz300328', 'sh600699', 'sz300207', 'sh603386', 'sz000779', 'sh600158', 'sz000717', 'sz002426',
                   'sz002496', 'sh603169', 'sh600343', 'sz300618', 'sz301113', 'sz300939', 'sh601677', 'sh600241',
                   'sh601882', 'sz002368', 'sz002423', 'sz002345', 'sz002616', 'sz002973', 'sz301005', 'sz300956',
                   'sz300250', 'sz000593', 'sh600284', 'sh603168', 'sz002815', 'sh600868', 'sz300033', 'sz002782',
                   'sh600408', 'sz002364', 'sz000099', 'sh603515', 'sz301056', 'sh603988', 'sh600708', 'sz000803',
                   'sz002866', 'sz300555', 'sz300977', 'sz002651', 'sz002881', 'sh603189', 'sz002279', 'sz300487',
                   'sz300223', 'sz301200', 'sz002926', 'sh600810', 'sh600104', 'sh600999', 'sz300771', 'sz002506',
                   'sh600170', 'sh600163', 'sz000977', 'sz301002', 'sz002551', 'sh603339', 'sz002465', 'sz002693',
                   'sh600624', 'sz002625', 'sh600237', 'sz300137', 'sz002634', 'sz300828', 'sz002598', 'sz000880',
                   'sh601611', 'sz300537', 'sz300793', 'sz301195', 'sh600512', 'sh601872', 'sz000677', 'sz300375',
                   'sz300417', 'sz300261', 'sz300148', 'sz300286', 'sz301302', 'sz002776', 'sz002045', 'sz300452',
                   'sz002523', 'sz300955', 'sz002235', 'sz000918', 'sz300013', 'sz000428', 'sh600393', 'sz002552',
                   'sz002639', 'sh600727', 'sz003039', 'sh603323', 'sz300255', 'sz000973', 'sh605133', 'sh600814',
                   'sh601956', 'sh605228', 'sz300834', 'sz001319', 'sz003006', 'sh600712', 'sz002249', 'sz002896',
                   'sh600776', 'sz300208', 'sh603948', 'sz300610', 'sh600573', 'sh603021', 'sz300483', 'sh600338',
                   'sz002818', 'sh603893', 'sh600101', 'sz300809', 'sz000609', 'sz002051', 'sz301092', 'sh601996',
                   'sh603686', 'sh603616', 'sz002630', 'sz301061', 'sz300667', 'sz300965', 'sz000830', 'sh603728',
                   'sh600734', 'sh603138', 'sh603122', 'sz002781', 'sh600151', 'sz300358', 'sz300814', 'sh603768',
                   'sz300968', 'sz300997', 'sz000021', 'sz300931', 'sz000623', 'sz002358', 'sh600570', 'sz301121',
                   'sh600360', 'sz002643', 'sh603156', 'sz000032', 'sz300895', 'sz300341', 'sh600692', 'sh600882',
                   'sz002237', 'sz300015', 'sh603036', 'sh600232', 'sh601199', 'sh603067', 'sz000063', 'sh603332',
                   'sz002892', 'sz300185', 'sh601012', 'sz301256', 'sz002965', 'sz002743', 'sh600981', 'sh603178',
                   'sh600966', 'sh603368', 'sh600584', 'sz000590', 'sz300527', 'sz300690', 'sz301011', 'sz000417',
                   'sz300782', 'sh605056', 'sz001210', 'sz300865', 'sh600715', 'sz301175', 'sh600853', 'sz300234',
                   'sh600501', 'sz002369', 'sh600520', 'sz300040', 'sh600600', 'sh600177', 'sh603926', 'sh603637',
                   'sz300288', 'sz300167', 'sz301196', 'sh603739', 'sh600510', 'sz002117', 'sz002737', 'sz300356',
                   'sz001234', 'sz002755', 'sz000737', 'sz300395', 'sz300632', 'sz002214', 'sh603866', 'sz002059',
                   'sz002953', 'sh600668', 'sz000950', 'sh600783', 'sz300512', 'sh600210', 'sz300582', 'sh603139',
                   'sz000407', 'sh603421', 'sh603505', 'sz301236', 'sz002472', 'sz002872', 'sz301110', 'sz000564',
                   'sz300280', 'sh600733', 'sh600982', 'sh605368', 'sz002124', 'sz300310', 'sz300669', 'sz002455',
                   'sh600903', 'sz300408', 'sz002073', 'sh600869', 'sz001914', 'sh600818', 'sz002656', 'sz000895',
                   'sh600623', 'sh600250', 'sh603800', 'sz300973', 'sh600604', 'sz002067', 'sz002851', 'sh603100',
                   'sh603828', 'sz300430', 'sz002483', 'sz300283', 'sz300035', 'sz002007', 'sz301321', 'sz002683',
                   'sz000514', 'sz002655', 'sh603518', 'sz300094', 'sz002192', 'sz301027', 'sz300422', 'sz300570',
                   'sh603706', 'sz002011', 'sh600173', 'sz000520', 'sh603019', 'sz300654', 'sh600196', 'sz002522',
                   'sz300116', 'sh601106', 'sz003018', 'sz002460', 'sh600628', 'sz002664', 'sz002211', 'sz300763',
                   'sz300065', 'sz002131', 'sz002294', 'sz300278', 'sz300854', 'sz002788', 'sh600449', 'sz300857',
                   'sz002978', 'sh601886', 'sh603121', 'sz000498', 'sz000798', 'sz000980', 'sz300237', 'sz301228',
                   'sz002239', 'sz002332', 'sz000408', 'sz300171', 'sz002897', 'sz002280', 'sz300894', 'sz000759',
                   'sh603087', 'sz001216', 'sz301100', 'sh600960', 'sh603222', 'sh603507', 'sz002209', 'sh600580',
                   'sz002595', 'sz300710', 'sz000669', 'sh601555', 'sz002036', 'sz300998', 'sz300775', 'sz000048',
                   'sz300227', 'sz001323', 'sh601966', 'sh603261', 'sz300226', 'sz301123', 'sh600527', 'sz000828',
                   'sz301349', 'sz300322', 'sh601878', 'sh600439', 'sh600513', 'sz002118', 'sz001331', 'sz000651',
                   'sz300672', 'sh603819', 'sz002372', 'sh600855', 'sz300396', 'sz300598', 'sz300850', 'sz300994',
                   'sz300466', 'sh600367', 'sh600572', 'sz000683', 'sz002278', 'sz300344', 'sz300675', 'sz300081',
                   'sz002050', 'sh605168', 'sz002016', 'sh600376', 'sz000301', 'sh601002', 'sh603716', 'sz300820',
                   'sh603288', 'sh603066', 'sh600515', 'sz002017', 'sz300628', 'sz301183', 'sz000411', 'sz002871',
                   'sh603268', 'sz002821', 'sz002600', 'sz002114', 'sh600506', 'sh603132', 'sh603033', 'sz300681',
                   'sh605266', 'sz002079', 'sh603005', 'sz000753', 'sh600310', 'sz002384', 'sh605178', 'sz000789',
                   'sz002316', 'sz002824', 'sz300067', 'sz000017', 'sz300114', 'sz300514', 'sh601512', 'sz300196',
                   'sz300496', 'sz301190', 'sz000917', 'sz002846', 'sh605259', 'sz000923', 'sh601155', 'sh600335',
                   'sz300113', 'sh601880', 'sz002076', 'sh603900', 'sh601236', 'sz300429', 'sz002058', 'sz002057',
                   'sh601126', 'sz002975', 'sz002816', 'sh601330', 'sh603055', 'sz300887', 'sh600433', 'sh603986',
                   'sz000155', 'sh605589', 'sh600996', 'sz002096', 'sh601116', 'sz002457', 'sz002183', 'sz002459',
                   'sh605169', 'sh600742', 'sz301187', 'sz300972', 'sz002882', 'sz300868', 'sh603817', 'sz002376',
                   'sz300822', 'sz000718', 'sz003015', 'sh600817', 'sh603338', 'sz002703', 'sh603726', 'sz002697',
                   'sh603011', 'sz300442', 'sz300543', 'sh603966', 'sz002198', 'sh600187', 'sz002951', 'sz300387',
                   'sz301339', 'sh600819', 'sh605089', 'sh601800', 'sz002583', 'sh600548', 'sz002311', 'sh600619',
                   'sh603709', 'sz300640', 'sz300966', 'sh603010', 'sz002446', 'sz002386', 'sz300369', 'sh601156',
                   'sh600556', 'sz301102', 'sh603043', 'sz002410', 'sh603711', 'sh600648', 'sz300920', 'sz301112',
                   'sh600905', 'sh601969', 'sh603035', 'sh600665', 'sz300333', 'sh600511', 'sh603556', 'sh600983',
                   'sz000826', 'sz002467', 'sz300433', 'sz300686', 'sh600462', 'sz300890', 'sz300923', 'sz002685',
                   'sh600552', 'sz300499', 'sz002113', 'sz002268', 'sh601098', 'sz300190', 'sz300103', 'sz002797',
                   'sz300302', 'sz002626', 'sz002342', 'sz002959', 'sz002466', 'sz000619', 'sz000536', 'sz301289',
                   'sz301043', 'sz000715', 'sz002908', 'sz002023', 'sz000151', 'sh600252', 'sh603335', 'sz003042',
                   'sz301296', 'sz000657', 'sh600199', 'sh600875', 'sh600490', 'sh600480', 'sh600337', 'sh601718',
                   'sz301046', 'sz300421', 'sz003025', 'sz300443', 'sz002366', 'sz002486', 'sz300503', 'sz301025',
                   'sz301071', 'sh600172', 'sh600293', 'sz300088', 'sh600256', 'sh600751', 'sz001696', 'sz000955',
                   'sz300663', 'sz002105', 'sz000538', 'sz300807', 'sh603712', 'sh605068', 'sz002119', 'sz002152',
                   'sh603826', 'sz300437', 'sh600388', 'sh603888', 'sh603118', 'sh600359', 'sz000966', 'sz000050',
                   'sz300705', 'sz300627', 'sh600505', 'sh603278', 'sz000029', 'sz300205', 'sh603392', 'sh600229',
                   'sz300813', 'sz300383', 'sz000935', 'sz300996', 'sz002875', 'sz300488', 'sz300510', 'sh603517',
                   'sh600581', 'sz002585', 'sh600425', 'sh600897', 'sh601916', 'sz300847', 'sz000587', 'sz000993',
                   'sh603083', 'sh600246', 'sh603918', 'sz002363', 'sh603767', 'sz002724', 'sh603950', 'sz301066',
                   'sz000932', 'sh605500', 'sh603070', 'sh603086', 'sz300761', 'sz002495', 'sz300769', 'sh603696',
                   'sz002666', 'sh601899', 'sh600898', 'sh600758', 'sh600854', 'sz300726', 'sz001206', 'sh600387',
                   'sh600587', 'sh603729', 'sz000503', 'sz301000', 'sh603041', 'sh600655', 'sz300001', 'sh600569',
                   'sz000983', 'sz002689', 'sz300901', 'sz300311', 'sz002575', 'sh601038', 'sh603324', 'sh603528',
                   'sh600667', 'sh603963', 'sh600321', 'sh603220', 'sz300641', 'sz300467', 'sz301179', 'sh600529',
                   'sz002905', 'sz000856', 'sh600616', 'sz300019', 'sz002233', 'sh601003', 'sh603101', 'sz300144',
                   'sh603520', 'sz000785', 'sh600963', 'sz301082', 'sh603387', 'sz000889', 'sz002100', 'sz300528',
                   'sz002682', 'sz300661', 'sz300545', 'sz300530', 'sz000026', 'sh600307', 'sh601319', 'sh600688',
                   'sz301003', 'sh600764', 'sh603090', 'sh600516', 'sz000606', 'sz002299', 'sz301326', 'sz002269',
                   'sh601377', 'sh601668', 'sh603519', 'sz000656', 'sz002633', 'sh600166', 'sh603006', 'sz300292',
                   'sz002286', 'sh600211', 'sz300984', 'sh600757', 'sz000777', 'sz300915', 'sh605186', 'sh603071',
                   'sz002244', 'sh600761', 'sh605100', 'sz300829', 'sz002999', 'sz000959', 'sh603815', 'sz002701',
                   'sh601918', 'sz002207', 'sz000821', 'sz002491', 'sz300917', 'sz002733', 'sz002510', 'sz000709',
                   'sz300318', 'sz000721', 'sh601000', 'sz001332', 'sh603605', 'sz000011', 'sz000066', 'sh600760',
                   'sz002169', 'sz002422', 'sz300508', 'sz002289', 'sz300722', 'sz000848', 'sz301062', 'sh603050',
                   'sz002828', 'sh600225', 'sz300657', 'sz002283', 'sz002298', 'sz300122', 'sz001227', 'sz002193',
                   'sz002873', 'sh605268', 'sz000791', 'sz002712', 'sh603896', 'sz002031', 'sz002887', 'sh600363',
                   'sz001236', 'sh605099', 'sz000544', 'sz002417', 'sz002661', 'sz300757', 'sh605358', 'sz002596',
                   'sh600399', 'sh603078', 'sh601456', 'sh600495', 'sz002906', 'sz300478', 'sz001270', 'sh603590',
                   'sz000545', 'sz300323', 'sz002027', 'sz002707', 'sh600271', 'sh603185', 'sh600980', 'sz301075',
                   'sz000035', 'sh600279', 'sh600383', 'sh600488', 'sz002554', 'sz300339', 'sz300365', 'sz300724',
                   'sz300900', 'sh603008', 'sz300513', 'sh601717', 'sz301263', 'sh603456', 'sz300163', 'sz301369',
                   'sh603353', 'sh603638', 'sh600352', 'sz000951', 'sz300161', 'sz000708', 'sz300388', 'sz002997',
                   'sh603031', 'sz002837', 'sz300644', 'sh600119', 'sz300245', 'sz002379', 'sh603555', 'sz300285',
                   'sh600362', 'sz300837', 'sh600530', 'sz002499', 'sz300637', 'sz300839', 'sh600997', 'sz300946',
                   'sh600468', 'sz300420', 'sh600543', 'sz002285', 'sh600280', 'sz002998', 'sz300591', 'sz301058',
                   'sh603880', 'sz301234', 'sh600958', 'sz301258', 'sh603969', 'sz300577', 'sz002910', 'sh600227',
                   'sh603198', 'sz300682', 'sz300293', 'sz300130', 'sz002835', 'sz002521', 'sh603618', 'sz300647',
                   'sz300142', 'sz002401', 'sz300558', 'sh601818', 'sz000612', 'sz000605', 'sz002044', 'sz300800',
                   'sz002083', 'sz002039', 'sz300623', 'sh600268', 'sz002353', 'sh603766', 'sh600123', 'sz301151',
                   'sh600157', 'sz002275', 'sz002612', 'sz002695', 'sz000622', 'sh603730', 'sz301040', 'sh600215',
                   'sz000586', 'sz300861', 'sz300735', 'sz002592', 'sz000922', 'sh603379', 'sz300450', 'sz002802',
                   'sz300881', 'sh600767', 'sz001308', 'sh600108', 'sz002568', 'sh600192', 'sz301218', 'sz300957',
                   'sz002251', 'sz002670', 'sz001336', 'sz002895', 'sz300241', 'sz002657', 'sz300985', 'sz000426',
                   'sh600804', 'sz002222', 'sz300389', 'sh600120', 'sz301037', 'sz301108', 'sz300653', 'sz301038',
                   'sz000421', 'sz301030', 'sh603970', 'sh601311', 'sh600658', 'sz002099', 'sh600620', 'sh603161',
                   'sh600348', 'sz300992', 'sz000757', 'sz002768', 'sz300049', 'sz002420', 'sz002563', 'sz301139',
                   'sh600249', 'sz300030', 'sz301115', 'sh600138', 'sz300357', 'sz301080', 'sh603383', 'sh603658',
                   'sz300471', 'sh603398', 'sh603690', 'sz300863', 'sz001313', 'sz000006', 'sh600545', 'sz002775',
                   'sz002893', 'sh601678', 'sz300319', 'sz003816', 'sz002129', 'sz000518', 'sz301180', 'sh600812',
                   'sh601995', 'sz002341', 'sz301022', 'sz300096', 'sz300673', 'sh600458', 'sh603236', 'sz301248',
                   'sz000985', 'sh603466', 'sh603538', 'sh603978', 'sh601238', 'sz002373', 'sh600846', 'sh603697',
                   'sz002903', 'sz300355', 'sh603329', 'sz301168', 'sz300089', 'sz000016', 'sz002767', 'sz002957',
                   'sz301045', 'sz301308', 'sz000070', 'sh600319', 'sz002145', 'sh601698', 'sz002266', 'sh603992',
                   'sh600420', 'sh603127', 'sz002929', 'sh603269', 'sz300377', 'sz000596', 'sz002741', 'sh600452',
                   'sz000010', 'sz002162', 'sh600737', 'sz000897', 'sz002013', 'sz300817', 'sh600306', 'sz300155',
                   'sh601006', 'sz300291', 'sz300698', 'sz300899', 'sh603227', 'sz300873', 'sz002845', 'sh600839',
                   'sh600397', 'sh605180', 'sz000691', 'sh600717', 'sz000901', 'sz000938', 'sz300833', 'sz300330',
                   'sz300935', 'sh605567', 'sz300321', 'sh600308', 'sz002256', 'sz002206', 'sz000539', 'sh605222',
                   'sz002586', 'sz002716', 'sz300648', 'sz000150', 'sz300165', 'sz002052', 'sz000020', 'sh600781',
                   'sz300791', 'sz300363', 'sh601798', 'sz002617', 'sz300204', 'sz002318', 'sh603093', 'sh600820',
                   'sh605123', 'sz300479', 'sz300556', 'sz300439', 'sz002485', 'sh600664', 'sh600390', 'sh600637',
                   'sh600650', 'sh603183', 'sz000404', 'sz300819', 'sh603703', 'sz002001', 'sz300893', 'sh600640',
                   'sh600653', 'sz002106', 'sz300177', 'sz300579', 'sz301130', 'sh603655', 'sz002414', 'sh603056',
                   'sh600285', 'sh601609', 'sh603803', 'sz003031', 'sh600409', 'sh605158', 'sz300634', 'sz300002',
                   'sh600636', 'sz002187', 'sh603650', 'sh600686', 'sh600824', 'sz301189', 'sz300563', 'sz002287',
                   'sh603559', 'sz002564', 'sz002602', 'sz002091', 'sz002217', 'sz002296', 'sz000989', 'sz000752',
                   'sz300238', 'sz300693', 'sh600126', 'sz301331', 'sz000532', 'sz002186', 'sz300394', 'sz300742',
                   'sh603283', 'sh601828', 'sz301055', 'sz000680', 'sz002566', 'sh601515', 'sh600815', 'sz300625',
                   'sh601825', 'sh600630', 'sz300242', 'sz301073', 'sz002322', 'sh600929', 'sh603026', 'sz301119',
                   'sh601789', 'sz301051', 'sh601858', 'sh603855', 'sz002890', 'sz002671', 'sz300106', 'sz300748',
                   'sz002084', 'sz002356', 'sz002545', 'sh600694', 'sz002681', 'sz003008', 'sz000926', 'sz300490',
                   'sh600782', 'sh600766', 'sh605033', 'sh600392', 'sh600395', 'sh600642', 'sh605136', 'sh600212',
                   'sh600644', 'sh603316', 'sz002731', 'sh600825', 'sz001330', 'sh600784', 'sz000413', 'sz301079',
                   'sz000861', 'sz003007', 'sz301181', 'sz000997', 'sz000636', 'sz001872', 'sh603639', 'sz000756',
                   'sh600178', 'sz002579', 'sz002937', 'sz300803', 'sh605011', 'sz300864', 'sz300840', 'sz300400',
                   'sh603813', 'sz300108', 'sz000799', 'sz301138', 'sh601799', 'sz002833', 'sz300214', 'sh600379',
                   'sh600537', 'sz301186', 'sz002786', 'sz300908', 'sh601198', 'sz002413', 'sz000517', 'sz002200',
                   'sz300546', 'sz002960', 'sz300102', 'sh601117', 'sz002325', 'sh600998', 'sz300720', 'sz300248',
                   'sz300604', 'sz300559', 'sz003005', 'sz002006', 'sz301162', 'sz002003', 'sz002962', 'sz300853',
                   'sh601997', 'sz000811', 'sz300309', 'sh603699', 'sz300461', 'sz000597', 'sh600919', 'sh600971',
                   'sh600550', 'sz002938', 'sh603155', 'sz301067', 'sz002453', 'sh603112', 'sh600423', 'sh600651',
                   'sh601339', 'sz002847', 'sh601208', 'sz000415', 'sz002752', 'sz300506', 'sz000820', 'sh603818',
                   'sz000663', 'sz300044', 'sz300024', 'sz300848', 'sh603578', 'sz000501', 'sz300314', 'sz000931',
                   'sh603899', 'sz300149', 'sz002201', 'sh600830', 'sz300193', 'sh600583', 'sz002032', 'sz300232',
                   'sh603176', 'sz300852', 'sh601279', 'sz002317', 'sz300147', 'sz300381', 'sh603358', 'sz002064',
                   'sz301132', 'sh603020', 'sh603758', 'sz000999', 'sz002047', 'sh600566', 'sz002127', 'sz000005',
                   'sz300741', 'sz002153', 'sz300709', 'sz002398', 'sz002319', 'sz002858', 'sh601727', 'sh603577',
                   'sz300340', 'sh600629', 'sz300772', 'sz000555', 'sz000409', 'sz300656', 'sz002365', 'sz300624',
                   'sz002407', 'sz300858', 'sh603390', 'sz002735', 'sz300070', 'sz000541', 'sh603759', 'sz002610',
                   'sz000815', 'sz002646', 'sh600110', 'sh603096', 'sz301330', 'sz000519', 'sh603876', 'sh603885',
                   'sz000002', 'sz300269', 'sz002983', 'sz002542', 'sz002128', 'sh603598', 'sz002742', 'sz000875',
                   'sz300115', 'sh600113', 'sh603330', 'sz000690', 'sz300026', 'sz300501', 'sz002213', 'sz000034',
                   'sh600099', 'sz002344', 'sz300532', 'sz000858', 'sz301209', 'sh603311', 'sz300335', 'sz002862',
                   'sh600197', 'sz300736', 'sh603887', 'sz300348', 'sh603665', 'sz300194', 'sh600153', 'sh600298',
                   'sh603833', 'sz002224', 'sz300399', 'sh603077', 'sh600638', 'sz300145', 'sh603501', 'sh603725',
                   'sz300889', 'sz300770', 'sh603698', 'sz002753', 'sz300534', 'sz301017', 'sz301116', 'sh600354',
                   'sz300253', 'sz000529', 'sz300263', 'sh601601', 'sz002715', 'sh603297', 'sh603217', 'sz000559',
                   'sh603196', 'sz002850', 'sh603551', 'sz001258', 'sz002441', 'sz300662', 'sh600353', 'sh601007',
                   'sz002324', 'sz300447', 'sh600428', 'sh600265', 'sh600845', 'sz000912', 'sz300586', 'sh600380',
                   'sz300621', 'sh600171', 'sh600346', 'sh600654', 'sz301118', 'sz002613', 'sh603901', 'sz002823',
                   'sz000159', 'sz300731', 'sh600200', 'sz002698', 'sh603181', 'sh603630', 'sh603719', 'sz300454',
                   'sh600567', 'sh603029', 'sz000603', 'sz000543', 'sz002252', 'sh600895', 'sz002293', 'sz002889',
                   'sh600405', 'sz002273', 'sh600802', 'sh603348', 'sz300160', 'sh601985', 'sz300811', 'sz300201',
                   'sz301106', 'sz300867', 'sh603718', 'sz300210', 'sh600633', 'sz301227', 'sh603103', 'sz002290',
                   'sz000547', 'sh603688', 'sz301068', 'sh603038', 'sz002925', 'sh600995', 'sz002780', 'sz301211',
                   'sz003012', 'sz000009', 'sz002507', 'sh605377', 'sh600729', 'sh600663', 'sz003023', 'sz300231',
                   'sz000732', 'sz300554', 'sh600312', 'sh600647', 'sz300121', 'sh603589', 'sz000576', 'sh605388',
                   'sz300948', 'sz001283', 'sz000524', 'sz002002', 'sh601816', 'sz000507', 'sz300995', 'sz002412',
                   'sh600528', 'sh603980', 'sh601216', 'sz002203', 'sz002901', 'sz002385', 'sh603313', 'sz002647',
                   'sz300485', 'sh600313', 'sh601616', 'sz002444', 'sz000158', 'sz301220', 'sh601599', 'sh600803',
                   'sh603915', 'sz002461', 'sz002475', 'sz002756', 'sz002146', 'sz002292', 'sz300391', 'sh600778',
                   'sz300213', 'sz301125', 'sh603959', 'sz300697', 'sz301093', 'sh603979', 'sz301015', 'sh601009',
                   'sz002500', 'sz301188', 'sz300721', 'sz000890', 'sh600809', 'sz000615', 'sz300962', 'sh600908',
                   'sz300150', 'sh603303', 'sz002247', 'sz003037', 'sh601162', 'sh600114', 'sh603045', 'sz002141',
                   'sz002608', 'sz002448', 'sh605050', 'sz300336', 'sz300409', 'sh601901', 'sz300455', 'sh600107',
                   'sz000338', 'sh603111', 'sh601619', 'sh600122', 'sz002346', 'sz000877', 'sz300615', 'sz000600',
                   'sh603171', 'sh601827', 'sh603319', 'sz300560', 'sh600865', 'sh601699', 'sz002355', 'sh600289',
                   'sh603363', 'sh605199', 'sz002727', 'sz002955', 'sz300561', 'sz301306', 'sh603115', 'sz000957',
                   'sz002879', 'sz002009', 'sz300199', 'sz300685', 'sh603219', 'sz000420', 'sz300482', 'sh600880',
                   'sh603609', 'sh603106', 'sz002832', 'sz002627', 'sz301128', 'sz002963', 'sz002212', 'sz301020',
                   'sh605198', 'sz300767', 'sh600805', 'sz002469', 'sz300180', 'sz300324', 'sz300118', 'sh600713',
                   'sz300495', 'sz002826', 'sh603721', 'sz300943', 'sz002395', 'sz002922', 'sh600562', 'sz000551',
                   'sh600523', 'sz300074', 'sz300885', 'sz300970', 'sh601390', 'sz000525', 'sz301336', 'sz300801',
                   'sh605069', 'sh601008', 'sh603615', 'sz300620', 'sh603558', 'sh600822', 'sz300688', 'sz300738',
                   'sh603486', 'sz002597', 'sz002842', 'sh600482', 'sz002668', 'sh600626', 'sz300017', 'sh603936',
                   'sh600796', 'sh601921', 'sh600266', 'sh603048', 'sz301269', 'sz002933', 'sh601021', 'sh601991',
                   'sz300713', 'sz002276', 'sh600617', 'sh601788', 'sh601369', 'sh600238', 'sz000906', 'sh600098',
                   'sz000937', 'sz002747', 'sz300192', 'sz000851', 'sh600213', 'sz000554', 'sz301328', 'sz002010',
                   'sz300129', 'sz000790', 'sz002082', 'sz301083', 'sz301163', 'sh603648', 'sh600257', 'sz300630',
                   'sz001979', 'sz300489', 'sh601226', 'sz300432', 'sz301169', 'sz002590', 'sh605300', 'sh600565',
                   'sz002518', 'sz300706', 'sz002125', 'sz001203', 'sh600297', 'sz000572', 'sz300523', 'sz002427',
                   'sz000710', 'sh600724', 'sz300376', 'sz300303', 'sz002090', 'sz300573', 'sz003000', 'sz002144',
                   'sh600800', 'sz300521', 'sh603305', 'sh601577', 'sz300009', 'sh603626', 'sh603629', 'sh600525',
                   'sh600749', 'sz300575', 'sh600959', 'sh600269', 'sz300676', 'sz300317', 'sz002870', 'sz002591',
                   'sh600208', 'sh603778', 'sz300538', 'sh600598', 'sz300613', 'sz300497', 'sh603757', 'sz300906',
                   'sz301276', 'sz002218', 'sh603881', 'sh601099', 'sh600858', 'sz300343', 'sz002040', 'sh600790',
                   'sz300398', 'sz000802', 'sz000534', 'sz002028', 'sh600489', 'sz002361', 'sz002533', 'sz300518',
                   'sz300392', 'sh603069', 'sz000739', 'sz000670', 'sz002166', 'sz300345', 'sz002132', 'sz002049',
                   'sz002676', 'sz000788', 'sz002931', 'sz002930', 'sh603256', 'sh601229', 'sh603890', 'sh603102',
                   'sh601869', 'sz001339', 'sz000628', 'sz002760', 'sz002863', 'sh600834', 'sz002869', 'sz002086',
                   'sz301096', 'sz002307', 'sz002264', 'sz300384', 'sz000403', 'sh600131', 'sz300162', 'sh600593',
                   'sz300716', 'sh600785', 'sh600739', 'sh600106', 'sz000833', 'sz000719', 'sz300580', 'sz300691',
                   'sz300701', 'sh600578', 'sz300061', 'sh600866', 'sz002748', 'sz002163', 'sz002686', 'sz300670',
                   'sh600148', 'sz300912', 'sz002329', 'sh603355', 'sh603159', 'sh600831', 'sz003035', 'sz300057',
                   'sz002809', 'sh601077', 'sz300427', 'sz300517', 'sz002921', 'sz002868', 'sh600827', 'sz002898',
                   'sz000546', 'sh600391', 'sh605298', 'sh603003', 'sz000762', 'sz300419', 'sz301161', 'sh600329',
                   'sz002942', 'sz300614', 'sh600202', 'sz301026', 'sz001316', 'sz300568', 'sz301266', 'sz000333',
                   'sh605116', 'sz002705', 'sz300989', 'sz002445', 'sh601128', 'sh600111', 'sz300821', 'sz301208',
                   'sz001213', 'sz300428', 'sz003032', 'sh603209', 'sz000960', 'sh605003', 'sz300136', 'sh603811',
                   'sh603408', 'sz002014', 'sz002174', 'sz002253', 'sz002354', 'sh600218', 'sz300812', 'sz300438',
                   'sh603836', 'sz002679', 'sh603309', 'sh603669', 'sz000685', 'sz301101', 'sh603889', 'sh601108',
                   'sh603308', 'sh600965', 'sz301160', 'sh605188', 'sz300860', 'sz002262', 'sh603377', 'sh603838',
                   'sh601375', 'sz002690', 'sz002880', 'sh601068', 'sz300846', 'sh603681', 'sz002967', 'sh603689',
                   'sh605287', 'sz002902', 'sh600273', 'sh600150', 'sz002122', 'sz002038', 'sz300960', 'sz000058',
                   'sz002899', 'sh601010', 'sz301063', 'sh600735', 'sh601336', 'sz300825', 'sz000796', 'sz002022',
                   'sz002481', 'sh600586', 'sz002732', 'sz301103', 'sz002396', 'sz301059', 'sh600716', 'sz002571',
                   'sh600518', 'sz300504', 'sh600639', 'sh600970', 'sz300535', 'sh600993', 'sz002403', 'sh603030',
                   'sh603829', 'sz000911', 'sh603797', 'sz002043', 'sh600133', 'sz301120', 'sz002303', 'sz300073',
                   'sz002699', 'sh603822', 'sh603158', 'sz002416', 'sh601231', 'sh600476', 'sh603320', 'sh601566',
                   'sh605277', 'sh600867', 'sh601139', 'sz000731', 'sh603225', 'sh603588', 'sz300958', 'sz300332',
                   'sz002593', 'sh601919', 'sz000910', 'sh601939', 'sz002857', 'sh600136', 'sh601860', 'sz003043',
                   'sz300290', 'sh600877', 'sh603239', 'sh603098', 'sz000059', 'sz000617', 'sz002859', 'sz300838',
                   'sz300502', 'sz000933', 'sz000768', 'sh605001', 'sh605128', 'sz300689', 'sz003001', 'sz003036',
                   'sz300592', 'sh601658', 'sh603385', 'sz000700', 'sz301338', 'sz300824', 'sz300913', 'sh601326',
                   'sz002658', 'sz002778', 'sz002535', 'sh600807', 'sz301193', 'sz300595', 'sz000616', 'sz300950',
                   'sh600979', 'sh600396', 'sz002993', 'sh600592', 'sz002958', 'sh600828', 'sz301069', 'sz300617',
                   'sz300095', 'sh601168', 'sz002087', 'sz301087', 'sz300674', 'sz300866', 'sz002813', 'sz301019',
                   'sz002380', 'sz002305', 'sh600847', 'sz002019', 'sh601186', 'sh601222', 'sz002375', 'sz300642',
                   'sz300469', 'sz300025', 'sz002758', 'sz000027', 'sh601388', 'sz300616', 'sz000838', 'sh600195',
                   'sh601877', 'sz002312', 'sz300884', 'sh603212', 'sh600956', 'sz002449', 'sz002629', 'sz002900',
                   'sh600467', 'sz300603', 'sz300878', 'sh601018', 'sz000659', 'sz002406', 'sz002969', 'sh600444',
                   'sh600989', 'sz300308', 'sz002641', 'sz300320', 'sh600612', 'sh603337', 'sz002180', 'sz002030',
                   'sz300153', 'sz002390', 'sh600165', 'sz300254', 'sz002115', 'sz000045', 'sz000807', 'sz002489',
                   'sz003011', 'sz001212', 'sz300719', 'sz000982', 'sh603863', 'sh600775', 'sz300414', 'sz300725',
                   'sz002228', 'sz000553', 'sz300403', 'sz002762', 'sz000488', 'sz002065', 'sz002093', 'sh601778',
                   'sz000850', 'sz002123', 'sz002281', 'sz002267', 'sz000630', 'sz002599', 'sz002991', 'sh601866',
                   'sz000014', 'sh603607', 'sz000060', 'sz301089', 'sh600316', 'sh600369', 'sh601890', 'sz002215',
                   'sz000888', 'sh600837', 'sz301313', 'sh600605', 'sz301278', 'sz301279', 'sz300257', 'sz002988',
                   'sz002391', 'sh601665', 'sz300295', 'sz000570', 'sh600787', 'sz301312', 'sz301023', 'sz002424',
                   'sh600236', 'sz002066', 'sh603613', 'sz300072', 'sz000697', 'sz000592', 'sz002241', 'sh603879',
                   'sz000831', 'sh600500', 'sh601233', 'sz002864', 'sz300411', 'sh600887', 'sh600927', 'sz300758',
                   'sz002121', 'sz300063', 'sz002438', 'sz300406', 'sh605398', 'sh603968', 'sz300936', 'sz001231',
                   'sz002757', 'sh603708', 'sh603596', 'sh600532', 'sh600722', 'sz300465', 'sh601011', 'sh601158',
                   'sz300717', 'sh605336', 'sh601163', 'sz300402', 'sz002130', 'sz300397', 'sh603396', 'sh603612',
                   'sz002470', 'sz002751', 'sz300659', 'sh603987', 'sz000913', 'sh600697', 'sz002339', 'sz300359',
                   'sz300941', 'sh603160', 'sz301052', 'sh600345', 'sh600588', 'sz002304', 'sz002644', 'sh603322',
                   'sz002558', 'sh603080', 'sh600283', 'sz300587', 'sz000898', 'sz300050', 'sz002620', 'sh603656',
                   'sh600976', 'sz000661', 'sh600267', 'sh600973', 'sh601811', 'sh600547', 'sz300783', 'sz002033',
                   'sh600406', 'sz300938', 'sz300282', 'sz002631', 'sz002774', 'sh603500', 'sh603585', 'sz000682',
                   'sz002801', 'sz300183', 'sh603499', 'sz002140', 'sz301021', 'sh600577', 'sz300963', 'sz000809',
                   'sz301127', 'sz301217', 'sz300157', 'sz301091', 'sz300562', 'sh603908', 'sz300609', 'sz002480',
                   'sz300141', 'sh600160', 'sh605183', 'sz000776', 'sz002544', 'sz002261', 'sz301039', 'sh603713',
                   'sz002015', 'sz300300', 'sz300684', 'sh600611', 'sz000810', 'sz002675', 'sz300084', 'sh600843',
                   'sh603327', 'sh605005', 'sz003017', 'sz002328', 'sz002853', 'sz002108', 'sh600939', 'sz000088',
                   'sz300942', 'sz301107', 'sz002419', 'sz002736', 'sh600159', 'sh603995', 'sz002005', 'sh601113',
                   'sh600935', 'sh603717', 'sz300768', 'sh601258', 'sh601288', 'sz300551', 'sz300971', 'sz301257',
                   'sh600358', 'sz300100', 'sz300097', 'sz000620', 'sz003009', 'sz300572', 'sz002216', 'sz300225',
                   'sz002415', 'sz000783', 'sz002538', 'sz300445', 'sz300607', 'sz300732', 'sz000007', 'sz301191',
                   'sh600448', 'sh600288', 'sh603228', 'sh603351', 'sz000419', 'sz300658', 'sz000795', 'sh601519',
                   'sh603126', 'sz003038', 'sh603722', 'sz300462', 'sz300778', 'sh603859', 'sh601808', 'sh603955',
                   'sz300449', 'sz002992', 'sh600707', 'sh600116', 'sz301008', 'sz000692', 'sh600519', 'sz002663',
                   'sz000632', 'sh600243', 'sz002772', 'sh600201', 'sz002528', 'sz301192', 'sz000676', 'sz300170',
                   'sh603188', 'sz300629', 'sz300481', 'sh603200', 'sh605077', 'sz300491', 'sh600281', 'sh601398',
                   'sh603777', 'sz000633', 'sz300599', 'sz002314', 'sz002154', 'sh600278', 'sz301085', 'sz001215',
                   'sz300498', 'sh601567', 'sz300843', 'sh600900', 'sz002817', 'sz002409', 'sz002504', 'sz002020',
                   'sh600679', 'sz000702', 'sz002271', 'sh605007', 'sh600961', 'sh605588', 'sh600539', 'sh603129',
                   'sz002623', 'sz002164', 'sz300818', 'sz300125', 'sz001218', 'sz000750', 'sz002883', 'sz002915',
                   'sz002615', 'sh603997', 'sz300926', 'sh603569', 'sz301029', 'sz300366', 'sh600676', 'sz000090',
                   'sz002272', 'sz002745', 'sz300891', 'sh600426', 'sh603666', 'sh603967', 'sz000042', 'sz300260',
                   'sh603095', 'sz300059', 'sz300730', 'sz301016', 'sh601608', 'sh600660', 'sz002282', 'sh600497',
                   'sh600220', 'sz300745', 'sz002550', 'sz002947', 'sz002638', 'sz300353', 'sz002197', 'sh601801',
                   'sz300982', 'sz300636', 'sh601686', 'sh600328', 'sh600933', 'sz002116', 'sz300284', 'sz002387',
                   'sh600602', 'sz000030', 'sz000589', 'sh601688', 'sh600683', 'sz300484', 'sh600789', 'sh603299',
                   'sz002948', 'sh601333', 'sz301205', 'sz000908', 'sz002961', 'sz002574', 'sh603051', 'sh603611',
                   'sz002576', 'sh600096', 'sh600130', 'sh603868', 'sz300708', 'sz002548', 'sh601177', 'sz300964',
                   'sh600459', 'sh605318', 'sz002188', 'sz300425', 'sz000968', 'sz300589', 'sz300197', 'sz002210',
                   'sz301135', 'sh603388', 'sz002148', 'sz000607', 'sz002519', 'sz000505', 'sz002691', 'sh601598',
                   'sh600188', 'sz301098', 'sz002024', 'sz002940', 'sh601005', 'sh603259', 'sh603018', 'sh601368',
                   'sz002771', 'sz000166', 'sz002796', 'sz300119', 'sz000729', 'sz300700', 'sz301004', 'sz300468',
                   'sz002476', 'sz002494', 'sh603938', 'sz301259', 'sh600901', 'sz300036', 'sz301095', 'sh603583',
                   'sh600750', 'sh600301', 'sh600248', 'sh603738', 'sz301057', 'sz300486', 'sz002348', 'sz300951',
                   'sz002856', 'sz300571', 'sh603557', 'sh600608', 'sh600728', 'sz300799', 'sh600507', 'sz002109',
                   'sz002245', 'sz002911', 'sz002103', 'sz301327', 'sz002302', 'sz000599', 'sh600894', 'sz002916',
                   'sz300539', 'sz002728', 'sz002739', 'sh600649', 'sh603733', 'sh601111', 'sz002098', 'sz300124',
                   'sz300583', 'sh603536', 'sz300533', 'sz300056', 'sz300666', 'sz000767', 'sz002431', 'sh603192',
                   'sh603128', 'sz300645', 'sz002159', 'sz002085', 'sh601669', 'sh600483', 'sh603693', 'sh600794',
                   'sz300135', 'sh603798', 'sz300557', 'sz300306', 'sh603858', 'sz300727', 'sz001267', 'sz000591',
                   'sz300585', 'sh603895', 'sh601857', 'sz002989', 'sh603663', 'sz002421', 'sz002761', 'sz000800',
                   'sz002101', 'sz002582', 'sh603933', 'sz002822', 'sh600398', 'sh603012', 'sh603458', 'sz000822',
                   'sz002231', 'sz300671', 'sh603076', 'sz002168', 'sz000711', 'sz000100', 'sh600674', 'sz002719',
                   'sz300068', 'sz300316', 'sz300444', 'sz000526', 'sz002624', 'sz002702', 'sh600962', 'sz300749',
                   'sz001207', 'sz000919', 'sz002177', 'sh600499', 'sh603931', 'sz002609', 'sz300006', 'sz300206',
                   'sz002012', 'sh600756', 'sz000019', 'sz002402', 'sh603186', 'sh601100', 'sz002810', 'sh603378',
                   'sz300219', 'sz000036', 'sh605155', 'sh603028', 'sz300762', 'sh600884', 'sz300790', 'sz300932',
                   'sz301206', 'sh603920', 'sz000626', 'sz300789', 'sz300596', 'sh601138', 'sh601500', 'sh605376',
                   'sh603496', 'sh601187', 'sh603976', 'sh600745', 'sh603331', 'sh600340', 'sz002297', 'sz002990',
                   'sh600808', 'sz000566', 'sz002790', 'sz300334', 'sh601607', 'sz300051', 'sz000701', 'sz300268',
                   'sz301239', 'sz002556', 'sz300928', 'sh603105', 'sz002383', 'sz000707', 'sz300492', 'sz300695',
                   'sh600711', 'sz002763', 'sh603701', 'sz300967', 'sz301065', 'sz000688', 'sz002156', 'sz002952',
                   'sh600517', 'sh600987', 'sz003016', 'sh600615', 'sh605378', 'sz001208', 'sz002970', 'sz300826',
                   'sz002041', 'sz002867', 'sz301298', 'sz300990', 'sz301216', 'sz002221', 'sh603359', 'sh600535',
                   'sh600463', 'sh605208', 'sz002543', 'sz001226', 'sh605166', 'sh603867', 'sh603167', 'sh600100',
                   'sh603587', 'sh600770', 'sz300117', 'sz002932', 'sz002456', 'sz002137', 'sz300354', 'sz300224',
                   'sz002388', 'sz000825', 'sz301088', 'sz000031', 'sh600575', 'sh603633', 'sz300862', 'sz300759',
                   'sh600403', 'sz002382', 'sz300918', 'sz002471', 'sh601166', 'sz300841', 'sz002046', 'sz300875',
                   'sz300239', 'sz300541', 'sz002151', 'sz002234', 'sz300608', 'sh603928', 'sz300405', 'sh600303',
                   'sh600816', 'sz301099', 'sz000698', 'sh601998', 'sh600801', 'sz002313', 'sh600768', 'sz300091',
                   'sz000837', 'sz300123', 'sz300270', 'sh603298', 'sz300832', 'sh603659', 'sz301333', 'sz000735',
                   'sh605118', 'sz000893', 'sz300078', 'sh600731', 'sz300244', 'sz002021', 'sz000818', 'sz000695',
                   'sh603707', 'sh603628', 'sh600167', 'sz300883', 'sz002696', 'sh600382', 'sh601633', 'sz300755',
                   'sz000761', 'sz000430', 'sh600176', 'sh600373', 'sh603808', 'sz300055', 'sh600132', 'sz002454',
                   'sz000969', 'sz300229', 'sz000055', 'sz301167', 'sh605580', 'sz000540', 'sz300164', 'sz300279',
                   'sh603013', 'sz002749', 'sz002088', 'sz000948', 'sz301233', 'sz000751', 'sz002766', 'sh603439',
                   'sz300927', 'sz002865', 'sz301078', 'sz300940', 'sh600748', 'sz300077', 'sz002524', 'sz301176',
                   'sh600149', 'sz000584', 'sz002848', 'sz300412', 'sz300041', 'sz002546', 'sz300677', 'sh600595',
                   'sh603903', 'sh600193', 'sh603927', 'sz000829', 'sh600992', 'sh600777', 'sz002996', 'sh603586',
                   'sz002603', 'sz002236', 'sz000801', 'sh605365', 'sz000996', 'sz300897', 'sz301182', 'sh600916',
                   'sh603527', 'sz300892', 'sh600508', 'sz002075', 'sz300404', 'sh601211', 'sz002825', 'sh605167',
                   'sz002452', 'sz002779', 'sh600259', 'sh600491', 'sz002678', 'sh603790', 'sh601881', 'sz002514',
                   'sh600657', 'sz002277', 'sz002246', 'sz002614', 'sz000998', 'sz300069', 'sz002539', 'sz000410',
                   'sz002532', 'sz002584', 'sh603823', 'sz002769', 'sh600260', 'sz002436', 'sz300453', 'sh605366',
                   'sh600117', 'sz002478', 'sz000716', 'sh603116', 'sz002223', 'sz300750', 'sz000736', 'sh605138',
                   'sz002694', 'sz300718', 'sz002909', 'sz002029', 'sz002530', 'sz002886', 'sh603989', 'sz301366',
                   'sh600112', 'sz300364', 'sz300831', 'sz301031', 'sh601058', 'sz002487', 'sz002935', 'sz002493',
                   'sz002370', 'sz002791', 'sz300519', 'sh603328', 'sz002068', 'sz002561', 'sh600410', 'sz300054',
                   'sz300349', 'sh600678', 'sh603602', 'sz000793', 'sh600272', 'sh605319', 'sh600986', 'sz300380',
                   'sh603027', 'sh600977', 'sz000899', 'sh600753', 'sz300037', 'sz002803', 'sz300983', 'sh600422',
                   'sh605162', 'sz001201', 'sz000733', 'sz000401', 'sz300120', 'sh603000', 'sz300522', 'sz002541',
                   'sh600985', 'sh603727', 'sz301053', 'sz300909', 'sz300877', 'sh603990', 'sz300371', 'sh603810',
                   'sz002581', 'sz001268', 'sz001296', 'sz300792', 'sz003028', 'sz300542', 'sz002709', 'sz300600',
                   'sh600599', 'sh605081', 'sh605008', 'sh603326', 'sz002167', 'sz300134', 'sz300235', 'sh600876',
                   'sh605177', 'sz002660', 'sz300525', 'sh603315', 'sh601728', 'sh603601', 'sh605028', 'sz000039',
                   'sz000712', 'sh600726', 'sh603301', 'sz300169', 'sh603429', 'sz300751', 'sz301235', 'sh600207',
                   'sh600635', 'sh601101', 'sz300550', 'sz000953', 'sz000668', 'sz000949', 'sh600258', 'sz300233',
                   'sh605179', 'sz300930', 'sz301024', 'sz301171', 'sz000713', 'sz301035', 'sh600277', 'sz000679',
                   'sz301081', 'sh603117', 'sz300976', 'sz002913', 'sz002667', 'sz300143', 'sh601975', 'sz000581',
                   'sz300905', 'sz002458', 'sh605088', 'sz002652', 'sz002946', 'sh600681', 'sz300222', 'sh600779',
                   'sz000925', 'sz002570', 'sz002673', 'sz300779', 'sh600917', 'sz300451', 'sz300953', 'sz002095',
                   'sh603180', 'sh605006', 'sh603023', 'sh601069', 'sh605098', 'sh600219', 'sh600496', 'sh600559',
                   'sh600883', 'sh603700', 'sh600879', 'sh603059', 'sz001238', 'sh600928', 'sz000001', 'sh603260',
                   'sz002034', 'sz300182', 'sz002037', 'sz001266', 'sz002936', 'sh600216', 'sz002097', 'sh600487',
                   'sz300493', 'sz300747', 'sh600276', 'sz300393', 'sh600522', 'sz002399', 'sz000930', 'sz002182',
                   'sz000513', 'sh601001', 'sh601518', 'sz000425', 'sh603667', 'sz300494', 'sz300198', 'sh600673',
                   'sz002377', 'sz002700', 'sz300327', 'sh603203', 'sz002557', 'sz300871', 'sz300240', 'sz002042',
                   'sh603789', 'sz000981', 'sz000037', 'sz300264', 'sh600873', 'sz000516', 'sz002255', 'sz300638',
                   'sh605338', 'sh600531', 'sh603109', 'sz300379', 'sh603300', 'sh600759', 'sz300823', 'sh600984',
                   'sz000726', 'sh603079', 'sh600859', 'sz300664', 'sz300787', 'sh600189', 'sz002134', 'sz002721',
                   'sz300842', 'sz002320', 'sh600844', 'sh603489', 'sh603566', 'sz300553', 'sh600191', 'sz002195',
                   'sz300021', 'sh601968', 'sz000671', 'sh601999', 'sz300274', 'sz000506', 'sz001202', 'sz300601',
                   'sz300830', 'sh603606', 'sh603166', 'sz002726', 'sz002950', 'sz300567', 'sz000598', 'sh600821',
                   'sh603060', 'sh603861', 'sh603919', 'sz001318', 'sz002484', 'sz301010', 'sz002189', 'sz300845',
                   'sh600478', 'sh600738', 'sz000792', 'sh601127', 'sz301086', 'sz300477', 'sz000987', 'sz002980',
                   'sz002529', 'sz300003', 'sh601992', 'sz300896', 'sh603081', 'sz002919', 'sz002433', 'sz301076',
                   'sz002242', 'sz002665', 'sz300480', 'sz301318', 'sh600791', 'sz000961', 'sh603195', 'sz002979',
                   'sh600351', 'sz000550', 'sz002852', 'sh600135', 'sz300159', 'sh601868', 'sz000627', 'sh605289',
                   'sh600206', 'sh603982', 'sz000885', 'sz002343', 'sz002717', 'sz300649', 'sh601019', 'sz002178',
                   'sz300902', 'sh600169', 'sh603279', 'sh600885', 'sh600377', 'sz300516', 'sz000929', 'sz300043',
                   'sz300876', 'sh603662', 'sz300712', 'sh601028', 'sz300780', 'sz000725', 'sz002405', 'sh600103',
                   'sz002330', 'sh603007', 'sz002434', 'sz002811', 'sz001288', 'sh600621', 'sz002158', 'sz002259',
                   'sz000920', 'sz002587', 'sh600941', 'sh600418', 'sz002920', 'sz002226', 'sh600318', 'sz301109',
                   'sh601952', 'sz300777', 'sh603444', 'sh600356', 'sz002884', 'sh605090', 'sh603579', 'sz300475',
                   'sz300584', 'sz300886', 'sh601118', 'sz000667', 'sh603839', 'sz300111', 'sz300520', 'sh600582',
                   'sz300305', 'sh600551', 'sh600456', 'sh601615', 'sh603786', 'sh600234', 'sz002238', 'sz002555',
                   'sh603676', 'sh600685', 'sh601086', 'sz300346', 'sz000533', 'sz300945', 'sz002490', 'sz002692',
                   'sz002323', 'sz301178', 'sh605255', 'sz301117', 'sz301229', 'sh603063', 'sz002928', 'sz002381',
                   'sz300993', 'sz000078', 'sz300565', 'sz002149', 'sz000963', 'sz300743', 'sh601318', 'sh600183',
                   'sh600560', 'sz301219', 'sz000965', 'sz000004', 'sz000156', 'sh600333', 'sz002176', 'sz300711',
                   'sz002511', 'sh601107', 'sh600730', 'sh600242', 'sz300034', 'sz003040', 'sh600460', 'sz000504',
                   'sh603949', 'sh600137', 'sz002860', 'sh605296', 'sz300470', 'sz300166', 'sz301033', 'sh603993',
                   'sz301309', 'sz301212', 'sh603848', 'sz002367', 'sh603956', 'sz300548', 'sz301153', 'sz002987',
                   'sh600469', 'sz300101', 'sz000678', 'sh603165', 'sz000975', 'sz002161', 'sh600860', 'sz300315',
                   'sz003010', 'sz002035', 'sz002738', 'sz300027', 'sz002594', 'sz002179', 'sh601328', 'sh600826',
                   'sh600549', 'sh600235', 'sz002635', 'sz300851', 'sz300921', 'sz301283', 'sz002173', 'sz300005',
                   'sh603678', 'sz002310', 'sz001230', 'sz300795', 'sz300267', 'sh600721', 'sh601990', 'sz000608',
                   'sz002517', 'sz002408', 'sz300187', 'sz300606', 'sz300991', 'sz300352', 'sh603306', 'sz001209',
                   'sh601360', 'sh600121', 'sh603416', 'sz002254', 'sz300276', 'sz300018', 'sh600585', 'sz002194',
                   'sh600503', 'sh600095', 'sz002062', 'sh600622', 'sh600682', 'sz002055', 'sz000429', 'sh600127',
                   'sz000816', 'sz002184', 'sz002309', 'sz300798', 'sh601928', 'sz002429', 'sh601865', 'sh600690',
                   'sz002839', 'sz300410', 'sz301060', 'sz002350', 'sh603516', 'sz301013', 'sz301166', 'sh601636',
                   'sh600461', 'sz000972', 'sz002092', 'sz300098', 'sz002443', 'sh600470', 'sh601595', 'sh603917',
                   'sh600918', 'sh601898', 'sh600368', 'sh603053', 'sz300952', 'sh600261', 'sz300802', 'sz002411',
                   'sz300350', 'sz300209', 'sh601212', 'sz300979', 'sz300109', 'sz002315', 'sh603608', 'sh603389',
                   'sz002335', 'sz000958', 'sh600579', 'sz002966', 'sz300258', 'sz002288', 'sz002746', 'sz000720',
                   'sz000530', 'sz300152', 'sz300605', 'sz000629', 'sz002327', 'sz000686', 'sh603806', 'sz300880',
                   'sh600366', 'sz002520', 'sh600355', 'sz002191', 'sz301072', 'sz301148', 'sh605117', 'sh600967',
                   'sh600661', 'sh600386', 'sz002807', 'sh603133', 'sz000558', 'sz002056', 'sh600797', 'sz000561',
                   'sz001219', 'sh600230', 'sz000886', 'sz002110', 'sz300919', 'sz301231', 'sz300788', 'sz002968',
                   'sh600643', 'sz301111', 'sh600282', 'sz002621', 'sh603983', 'sh600239', 'sz300844', 'sh603860',
                   'sz000625', 'sz002306', 'sz001289', 'sz002479', 'sz301155', 'sh603108', 'sz002537', 'sz300168',
                   'sz002540', 'sz002205', 'sz000766', 'sh603068', 'sz301007', 'sh600589', 'sz002502', 'sz002559',
                   'sz300266', 'sh600863', 'sz300870', 'sz002060', 'sh603113', 'sz002648', 'sz301158', 'sz300597',
                   'sz003030', 'sh605599', 'sz301319', 'sh600162', 'sh601179', 'sz001228', 'sh600509', 'sz002645',
                   'sz002139', 'sz002172', 'sh603177', 'sz301149', 'sz000915', 'sz000046', 'sh603179', 'sh600223',
                   'sz000049', 'sz300281', 'sh603221', 'sh603568', 'sz000962', 'sh603099', 'sz002393', 'sh600521',
                   'sz300611', 'sh603488', 'sz000782', 'sz300249', 'sz301282', 'sz000882', 'sz300593', 'sh605305',
                   'sz300313', 'sz300922', 'sz002729', 'sz001965', 'sz002878', 'sz002718', 'sz300329', 'sz002843',
                   'sz300937', 'sh601298', 'sh600811', 'sz002723', 'sh600874', 'sh600217', 'sh600662', 'sz300635',
                   'sz001317', 'sz003002', 'sz000061', 'sh603600', 'sz300128', 'sz002562', 'sz002171', 'sh603230',
                   'sh603258', 'sh600365', 'sz002508', 'sz300898', 'sh603937', 'sz002492', 'sz301129', 'sh605299',
                   'sz002971', 'sz002138', 'sz300211', 'sz300980', 'sz002888', 'sh600502', 'sh600290', 'sh603508',
                   'sz000402', 'sz300564', 'sh603277', 'sh600841', 'sh600141', 'sz000065', 'sz300271', 'sz300907',
                   'sz000563', 'sz002094', 'sz002482', 'sz001259', 'sz002836', 'sz002263', 'sh601777', 'sh605009',
                   'sz300448', 'sz002912', 'sz000601', 'sh600693', 'sz000971', 'sh600590', 'sz300131', 'sz300146',
                   'sh600203', 'sz002437', 'sz002053', 'sh603909', 'sz000903', 'sz000813', 'sz002185', 'sh600128',
                   'sh601020', 'sz002927', 'sh603286', 'sh603506', 'sh600179', 'sz002468', 'sz000531', 'sz002462',
                   'sz300046', 'sh600732', 'sh603825', 'sh600228', 'sh600161', 'sh601066', 'sz002577', 'sh600180',
                   'sz000990', 'sz002918', 'sh603085', 'sz002072', 'sz300949', 'sz002622', 'sh600105', 'sh600771',
                   'sz301197', 'sz300080', 'sz300331', 'sz002907', 'sz003021', 'sz002795', 'sz301042', 'sz300071',
                   'sh603991', 'sz300836', 'sz300578', 'sh600155', 'sz002308', 'sz300052', 'sz002649', 'sh600571',
                   'sh603214', 'sz002389', 'sh603367', 'sh600477', 'sz002291', 'sh603213', 'sh601766', 'sh603533',
                   'sh600675', 'sh603032', 'sz301201', 'sz300386', 'sh605339', 'sz300058', 'sz301047', 'sh603229',
                   'sz300756', 'sz301159', 'sz301077', 'sh600829', 'sz300184', 'sh600375', 'sh603773', 'sh600696',
                   'sz301198', 'sz002777', 'sz002877', 'sz002400', 'sz002985', 'sh600558', 'sh600295', 'sz002202',
                   'sh603366', 'sh603916', 'sz300626', 'sz002418', 'sz002840', 'sh600568', 'sz300474', 'sz002939',
                   'sh600671', 'sh600744', 'sz300382', 'sz300511', 'sh600702', 'sz002891', 'sz002580', 'sh600322',
                   'sz300774', 'sz300272', 'sh603123', 'sz301012', 'sh601900', 'sh600892', 'sz000056', 'sz002080',
                   'sz300500', 'sz300643', 'sz300373', 'sz300047', 'sz300806', 'sz300220', 'sz002107', 'sh600466',
                   'sh605108', 'sz300633', 'sz002827', 'sh605399', 'sz300008', 'sz300140', 'sh601200', 'sh603856',
                   'sh601568', 'sz000862', 'sh603336', 'sz300947', 'sz001205', 'sz002945', 'sz000610', 'sz000038',
                   'sz002515', 'sz002829', 'sh605598', 'sz000637', 'sh603595', 'sh600332', 'sz002798', 'sh600094',
                   'sh605189', 'sz002295', 'sz000921', 'sh601188', 'sz300259', 'sz002428', 'sh600955', 'sz300540',
                   'sz000852', 'sz300740', 'sh600889', 'sh605258', 'sz000521', 'sz001211', 'sz000422', 'sz002392',
                   'sz002607', 'sz002653', 'sz301097', 'sz002232', 'sz002567', 'sz300986', 'sz300459', 'sz300849',
                   'sz001229', 'sz300766', 'sz002270', 'sz002611', 'sz300988', 'sz300464', 'sz002759', 'sh600792',
                   'sh603199', 'sh600719', 'sh603317', 'sz300275', 'sz002972', 'sh605389', 'sz300835', 'sz300463',
                   'sz002565', 'sh600429', 'sz300655', 'sz002885', 'sz300590', 'sh600168', 'sh600325', 'sh600327',
                   'sh603682', 'sz002061', 'sh600641', 'sh603776', 'sz300594', 'sh600198', 'sz301207', 'sh600763',
                   'sz300981', 'sz300020', 'sz003029', 'sh605086', 'sz001217', 'sh600152', 'sh600381', 'sh603360',
                   'sz300093', 'sh601366', 'sz300569', 'sz300360', 'sz300299', 'sh600888', 'sh600743', 'sh600323',
                   'sz002830', 'sh600475', 'sz002601', 'sz002861', 'sz300127', 'sz300576', 'sh603058', 'sz002435',
                   'sh600435', 'sh600292', 'sh603610', 'sh601399', 'sz300277', 'sh603878', 'sz001255', 'sz000635',
                   'sz002334', 'sz002321', 'sz301177', 'sh600597', 'sz002077', 'sz002054', 'sz002225', 'sh603799',
                   'sz301006', 'sh600741', 'sz000560', 'sz300215', 'sz001269', 'sh600926', 'sh600231', 'sz002204',
                   'sz002352', 'sh605333', 'sz002799', 'sz300176', 'sz003013', 'sz002505', 'sz002074', 'sz300016',
                   'sz002175', 'sz002672', 'sz000089', 'sh600618', 'sz300458', 'sz002126', 'sz002849', 'sh605555',
                   'sh603321', 'sh603136', 'sz300859', 'sh600936', 'sh603238', 'sh603636', 'sz300683', 'sz002104',
                   'sz002142', 'sh600594', 'sh600438', 'sz300969', 'sh600990', 'sh600455', 'sh603677', 'sh600835',
                   'sh600736', 'sh605018', 'sh603197', 'sz300696', 'sh603365', 'sz300435', 'sz002048', 'sz300526',
                   'sz300351', 'sz000069', 'sh600330', 'sz002331', 'sz002995', 'sz300378', 'sz300385', 'sh601908',
                   'sz300781', 'sz300326', 'sh600666', 'sh603843', 'sz002674', 'sz000681', 'sz301131', 'sz300816',
                   'sz002219', 'sz000728', 'sz002157', 'sz300066', 'sz002397', 'sz300457', 'sz300515', 'sh600765',
                   'sz300507', 'sz300547', 'sh600115', 'sh600704', 'sh603801', 'sh601929', 'sz002155', 'sz000978',
                   'sz300012', 'sz300014', 'sz300265', 'sz300729', 'sz300076', 'sh600536', 'sz300413', 'sh603660',
                   'sz002474', 'sz000927', 'sh600606', 'sz000936', 'sz002982', 'sh600969', 'sz300434', 'sz300217',
                   'sz300218', 'sz300105', 'sz000952', 'sh603897', 'sz300172', 'sh600118', 'sz000819', 'sh600222',
                   'sz002284', 'sz002986', 'sz002855', 'sz002642', 'sz002578', 'sz002371', 'sh600226', 'sz002976',
                   'sh603266', 'sz002536', 'sz002669', 'sh605080', 'sz002349', 'sz300810', 'sz002636', 'sh600881',
                   'sz301238', 'sh601628', 'sz301185', 'sh600601', 'sz002442', 'sh600988', 'sh600185', 'sh600975',
                   'sh603788', 'sh603617', 'sz300549', 'sh600315', 'sz300304', 'sz003041', 'sz002102', 'sh600613',
                   'sz002250', 'sz300390', 'sh605499', 'sz002394', 'sh600596', 'sz301150', 'sh605288', 'sh603042',
                   'sz300301', 'sh600129', 'sz301300', 'sh600305', 'sz300612', 'sz300407', 'sz300099', 'sz002230',
                   'sz300200', 'sz300158', 'sz000537', 'sh600233', 'sh603356', 'sz002498', 'sz002838', 'sz300639',
                   'sh600705', 'sh601579', 'sh600838', 'sz300212', 'sz002512', 'sz002831', 'sh603218', 'sz002632',
                   'sz002740', 'sz300337', 'sz300737', 'sz000836', 'sz000568', 'sz002425', 'sz002243', 'sh600833',
                   'sz002812', 'sh601700', 'sh600540', 'sh601838', 'sz300022', 'sz002196', 'sz003027', 'sz002808',
                   'sz300933', 'sh605337', 'sz300191', 'sh600720', 'sz002734', 'sz300048', 'sz002876', 'sz300045',
                   'sz300566', 'sh600446', 'sh600864', 'sz300243', 'sh603357', 'sh600714', 'sz300529', 'sz300739',
                   'sz002513', 'sh603216', 'sz002265', 'sz300505', 'sz003019', 'sz300251', 'sz000778', 'sz000722',
                   'sz000557', 'sz300588', 'sh600546', 'sz002526', 'sh600493', 'sz002650', 'sh603399', 'sz002573',
                   'sz300460', 'sh600320', 'sh600850', 'sh603580', 'sz000153', 'sz000869', 'sz000976', 'sz300773',
                   'sz300032', 'sz002783', 'sz002378', 'sz301288', 'sz002300', 'sh601989', 'sz002628', 'sz300703',
                   'sh600190', 'sh600906', 'sz002714', 'sh600862', 'sh600740', 'sh600350', 'sh603333', 'sz000909',
                   'sz300552', 'sz300776', 'sz301070', 'sh603816', 'sz300888', 'sz002301', 'sz002981', 'sh600331',
                   'sz301222', 'sh600871', 'sz000806', 'sh600725', 'sz000523', 'sz300173', 'sz002025', 'sh603477',
                   'sz300139', 'sz300075', 'sz300082', 'sh601666', 'sh603002', 'sh605218', 'sh603599', 'sh603877',
                   'sz300289', 'sz301028']

    process_symbol_map = {}
    process_count = 10
    with multiprocessing.Manager() as mngr:
        pool = multiprocessing.Pool(process_count)

        for process_id in range(process_count):
            process_symbol_map[process_id] = []

        for i, symbol in enumerate(symbol_list):
            process_symbol_map[i % process_count].append(symbol)
        for process_id, symbol in process_symbol_map.items():
            pool.apply_async(_run_symbol,
                             args=(process_symbol_map[process_id], process_id,))
        pool.close()
        pool.join()
    print("finish!!")
