# 获取股票数据 并处理数据代码
import pandas as pd
import tushare as ts

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import warnings

import datetime
import numpy as np
from pandas import Series,DataFrame
def getEveryDay(begin_date,end_date):
    # 前闭后闭
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list

def deal(deal_data,i):
    if np.isnan(deal_data['close1'][i])==True:
        if np.isnan(deal_data['close1'][i+1])==True:
            deal(deal_data,i+1)
        if np.isnan(deal_data['close1'][i+1])==False:
            deal_data['close1'][i] = deal_data['close1'][i+1]
    return deal_data

def dealdata(deal_data):
    for i in range(len(deal_data)):
        if np.isnan(deal_data['close'][i])==False:
            deal_data['close1'][i] = deal_data['close'][i]
    for i in range(len(deal_data)):
        deal_data = deal(deal_data,i)
    return deal_data

# data_list = DataFrame(index = getEveryDay('2021-01-01','2021-05-31'),columns = {'close1':''})
tonghang1 = ['000799','600702','600519','000858','000568','000002','600048','000817','000069','603948']
tonghang2 = ['601988','601939','601398','600036','000001','002230','000100','000333','002035','000651']
tonghang3 = ['000426','000655','000688','000762','001203','000150','000503','000813','002004','002044']
# print(tonghang)
df=ts.get_hist_data('600690',start='2021-01-01',end='2021-06-30') # —次性获取全部日k线数据
# df=ts.get_hist_data('000799',start='2021-01-01',end='2021-06-30') # —次性获取全部日k线数据

data = df[::-1]#颠倒顺序
data = data[['open','high','close','low','volume','price_change']]
data.to_csv('600690data.csv')#导出数据
# data.to_csv('000799data.csv')#导出数据

# # # data['close']#查看数据
# # # data['close']
# # # datas = {'data':data.index,'close':data['close']}
datas = DataFrame(data['close'])
# datas.to_csv('000799.csv')
datas.to_csv('600690.csv')
# # deal_data = pd.concat([data_list,datas],axis=1)
# # deal_datas = dealdata(deal_data)
# # deal_datas = deal_datas['close1']
# # deal_datas.index.name = 'date'
# # deal_datas
# # deal_datas.to_csv('600690_deal_data.csv',index=True,header=True)
