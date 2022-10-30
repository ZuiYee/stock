#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import akshare as ak
import libs.common as common
import pandas as pd

print(ak.__version__)

# 历史行情数据
# 日频率
# 接口: stock_zh_a_daily
# 目标地址: https://finance.sina.com.cn/realstock/company/sh600006/nc.shtml(示例)
# 描述: A 股数据是从新浪财经获取的数据, 历史数据按日频率更新; 注意其中的 sh689009 为 CDR, 请 通过 stock_zh_a_cdr_daily 接口获取
# 限量: 单次返回指定 A 股上市公司指定日期间的历史行情日频率数据
# adjust=""; 默认为空: 返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据;

# stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol="sz000066", adjust="")
# print(stock_zh_a_daily_qfq_df)
# sz_symbol_list = ak.stock_info_sz_name_code()
sz_symbol_list = ak.stock_info_sz_name_code()
sh_symbol_list = ak.stock_info_sh_name_code()
import time

all_list = list(sh_symbol_list['COMPANY_CODE'].values) + list(sz_symbol_list["A股代码"].values)
print(all_list)
# s_list = []
# f_list = []
# for v in list(sh_symbol_list['COMPANY_CODE'].values):
#     try:
#         symbol = 'sh' + v
#         stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol, start_date="20170930", end_date="20220930", adjust="")
#         print(stock_zh_a_daily_qfq_df)
#         stock_zh_a_daily_qfq_df['date'] = pd.to_datetime(stock_zh_a_daily_qfq_df['date']).apply(lambda x: x.date().strftime("%Y%m%d"))
#         stock_zh_a_daily_qfq_df["symbol"] = symbol
#
#         # 插入到 MySQL 数据库中
#         common.insert_db(stock_zh_a_daily_qfq_df, "stock_zh_a_daily", True, "`symbol`")
#         s_list.append(symbol)
#         time.sleep(60)
#     except Exception as e:
#         f_list.append(symbol)
#
#
# for v in list(sz_symbol_list["A股代码"].values):
#     # bj_symbol_list = ak.stock_info_bj_name_code()
#     # symbol="sz000001"
#     try:
#         symbol = 'sz' + v
#         stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol, start_date="20170930", end_date="20220930", adjust="")
#         print(stock_zh_a_daily_qfq_df)
#         stock_zh_a_daily_qfq_df['date'] = pd.to_datetime(stock_zh_a_daily_qfq_df['date']).apply(lambda x: x.date().strftime("%Y%m%d"))
#         stock_zh_a_daily_qfq_df["symbol"] = symbol
#
#         # 插入到 MySQL 数据库中
#         common.insert_db(stock_zh_a_daily_qfq_df, "stock_zh_a_daily", True, "`symbol`")
#         s_list.append(symbol)
#         time.sleep(60)
#     except Exception as e:
#         f_list.append(symbol)
# print(s_list)
# print(f_list)
