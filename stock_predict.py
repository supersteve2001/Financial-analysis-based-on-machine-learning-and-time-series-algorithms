# 股票预测增降的八个权威参数的计算
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime

data = pd.read_csv('600690data.csv', encoding='utf-8',index_col='date')
# train = data['2021-01-04':'2021-05-31']
# test = data['2021-05-31':'2021-06-30']

# 简单移动平均线(SMA) (Simple Moving Average (SMA))
data['10_SMA'] = data['close'].rolling(window = 10, min_periods = 1).mean()
# data['20_SMA'] = data['close'].rolling(window = 20, min_periods = 1).mean()
# data['50_SMA'] = data['close'].rolling(window = 50, min_periods = 1).mean()

# 移动平均交叉策略(WMA) (Moving Average Crossover Strategy)这里采用平方系数加权移动平均线
def Moving_average(data):
    for i in range(len(data)):
        if i == 0:
            data['WMA'][i] = data['close'][i]
        elif i == 1:
            data['WMA'][i] = (data['close'][0]+data['close'][1]*4)/(1+4)
        elif i == 2:
            data['WMA'][i] = (data['close'][0]+data['close'][1]*4+data['close'][2]*9)/(1+4+9)
        elif i == 3:
            data['WMA'][i] = (data['close'][0]+data['close'][1]*4+data['close'][2]*9+data['close'][3]*16)/(1+4+9+16)
        elif i >= 4:
            data['WMA'][i] = (data['close'][i-4]+data['close'][i-3]*4+data['close'][i-2]*9+data['close'][i-1]*16+data['close'][i]*25)/(1+4+9+16+25)
    return data
data['WMA'] = 0.1
data = Moving_average(data)      

# 移动平均线收敛发散指标（MACD）    通过12期EMA中减去26期EMA来计算MACD。
# 获取EMA数据 , cps：close_prices 收盘价集合 days:日期 days=5 5日线
def get_EMA(cps, days):
    emas = cps.copy()  # 创造一个和cps一样大小的集合
    for i in range(len(cps)):
        if i == 0:
            emas[i] = cps[i]
        if i > 0:
            emas[i] = ((days - 1) * emas[i - 1] + 2 * cps[i]) / (days + 1)
    return emas
EMA12 = get_EMA(data['close'],12)
EMA26 = get_EMA(data['close'],24)
data['MACD'] = EMA12 - EMA26

# W_R 威廉指标，识别指数是超买还是超卖，主要是用来分析市场的短期趋势
def WR(data):
    for i in range(len(data)):
        if i < 5 :
            data['W_R'][i] = 100 * (max(data['high'][:i+1])-data['close'][i])/(max(data['high'][:i+1])-min(data['low'][:i+1]))
        elif i >= 5:
            data['W_R'][i] = 100 * (max(data['high'][i-5:i+1])-data['close'][i])/(max(data['high'][i-5:i+1])-min(data['low'][i-5:i+1]))
    return data
data['W_R'] = 0.1
data = WR(data)

# RSI 相对强弱指标，属于最基本但是实用性强的震荡类指标，用于显示选定时间段内多空双方的实力比较，从而分析出趋势，取周期参数为N=14
# 强弱指标理论认为，任何市价的大涨或大跌，均在0-100之间变动，根据常态分配，认为RSI值多在30-70之间变动，
# 通常80甚至90时被认为市场已到达超买状态，至此市场价格自然会回落调整。当价格低跌至30以下即被认为是超卖状态，市价将出现反弹回升。
def get_RSI(data):
    for i in range(len(data)):
        if i < 13:
            A_t = [i for i in data['price_change'][:i+1] if i > 0]
            B_t = [i for i in data['price_change'][:i+1] if i < 0]
            if len(A_t) == 0:
                A_t.append(0)
            if len(B_t) == 0:
                B_t.append(0)
            data['RSI'][i] = 100 * np.mean(A_t)/(np.mean(A_t)+abs(np.mean(B_t)))
        if i >= 13:
            A_t = [i for i in data['price_change'][i-13:i+1] if i > 0]
            B_t = [i for i in data['price_change'][i-13:i+1] if i < 0]
            if len(A_t) == 0:
                A_t.append(0)
            if len(B_t) == 0:
                B_t.append(0)
            data['RSI'][i] = 100 * np.mean(A_t)/(np.mean(A_t)+abs(np.mean(B_t)))
    return data
data['RSI'] = 0.1
data = get_RSI(data)

# CCI 顺势指标，是衡量价格变化与其平均价格的偏离度的分析工具，能够识别股票中短线投资的超买超卖现象。这里选取周期参数为n=14
def CCI(data):
    TP = (data['high']+data['low']+data['close'])/3
    for i in range(len(data)):
        D_t = 0
        data['CCI'][0] = 0
        if i < 13 and i != 0:
            for j in range(0,i+1):
                D_t = D_t + abs(TP[i-j]-TP.rolling(window = 14, min_periods = 1).mean()[i])
            D_t = D_t/(i+1)
            data['CCI'][i] = (TP[i]-TP.rolling(window = 14, min_periods = 1).mean()[i])/D_t
        elif i >= 13:
            for j in range(i-13,i+1):
                D_t = D_t + abs(TP[i-j]-TP.rolling(window = 14, min_periods = 1).mean()[i])
            D_t = D_t/14
            data['CCI'][i] = (TP[i]-TP.rolling(window = 14, min_periods = 1).mean()[i])/D_t
    return data
data['CCI'] = 0.1
data = CCI(data)

# A/D 集散指标，由价格和成交量的变化而决定的，是常用的量价指标的一种
def AD(data):
    for i in range(len(data)):
        data['AD'][i] = ((data['close'][i]-data['low'][i])-(data['high'][i]-data['close'][i]))*data['volume'][i]/(data['high'][i]-data['low'][i])
    return data
data['AD'] = 0.1
data = AD(data)

# MTM 动量指标，是以分析价格波动的速度、衡量价格波动的动能为目标，研究股票价格在波动过程中所出现的运动现象的指标。
# MTM 指标表示的是当前的价格相对于n日前的价格变化幅度大小。这里选取周期参数n=10
def get_MTM(data):
    for i in range(len(data)):
        if i < 10:
            data['MTM'][i] = data['close'][i] - data['close'][0]
        elif i >=10:
            data['MTM'][i] = data['close'][i] - data['close'][i-10]
    return data
data['MTM'] = 0.1
data = get_MTM(data)
# data = data[['10_SMA','WMA','MACD','W_R','RSI','CCI','AD','MTM']]
# data = data[1:]
# data.to_csv('1.csv')#导出数据
# 将技术指标离散化
def disperse(data):
    MACD_copy = data['MACD'].copy()
    MACD_copy[0] = -1
    W_R_copy = data['W_R'].copy()
    W_R_copy[0] = -1
    RSI_copy = data['RSI'].copy()
    RSI_copy[0] = -1
    CCI_copy = data['CCI'].copy()
    CCI_copy[0] = -1
    AD_copy = data['AD'].copy()
    AD_copy[0] = -1
    data['label'] = 0

    for i in range(len(data)):
        if(data['close'][i] > data['10_SMA'][i]):
            data['10_SMA'][i] = 1
        elif(data['close'][i] <= data['10_SMA'][i]):
            data['10_SMA'][i] = -1
        
        if(data['close'][i] > data['WMA'][i]):
            data['WMA'][i] = 1
        elif(data['close'][i] <= data['WMA'][i]):
            data['WMA'][i] = -1
        
        if i != 0:
            if(data['MACD'][i] > data['MACD'][i-1]):
                MACD_copy[i] = 1
            elif(data['MACD'][i] <= data['MACD'][i-1]):
                MACD_copy[i] = -1

        if i != 0:
            if data['W_R'][i] > 85:
                W_R_copy[i] = 1
            elif data['W_R'][i] < 15:
                W_R_copy[i] = -1
            elif data['W_R'][i] >= 15 and data['W_R'][i] <= 85:
                if(data['W_R'][i] > data['W_R'][i-1]):
                    W_R_copy[i] = 1
                elif(data['W_R'][i] <= data['W_R'][i-1]):
                    W_R_copy[i] = -1
        
        if i != 0:
            if data['RSI'][i] > 70:
                RSI_copy[i] = -1
            elif data['RSI'][i] < 30:
                RSI_copy[i] = 1
            elif data['RSI'][i] >= 30 and data['RSI'][i] <= 70:
                if(data['RSI'][i] > data['RSI'][i-1]):
                    RSI_copy[i] = 1
                elif(data['RSI'][i] <= data['RSI'][i-1]):
                    RSI_copy[i] = -1

        if i != 0:
            if data['CCI'][i] > 200:
                CCI_copy[i] = -1
            elif data['CCI'][i] < -200:
                CCI_copy[i] = 1
            elif data['CCI'][i] >= -200 and data['CCI'][i] <= 200:
                if(data['CCI'][i] > data['CCI'][i-1]):
                    CCI_copy[i] = 1
                elif(data['CCI'][i] <= data['CCI'][i-1]):
                    CCI_copy[i] = -1

        if i != 0:
            if(data['AD'][i] > data['AD'][i-1]):
                AD_copy[i] = 1
            elif(data['AD'][i] <= data['AD'][i-1]):
                AD_copy[i] = -1

        if(data['MTM'][i] > 0):
            data['MTM'][i] = 1
        elif(data['MTM'][i] <= 0):
            data['MTM'][i] = -1
        
        if(i != 0):
            if(data['close'][i] > data['close'][i-1]):
                data['label'][i] = 1
            elif(data['close'][i] <= data['close'][i-1]):
                data['label'][i] = 0

    data['MACD'] = MACD_copy
    data['W_R'] = W_R_copy
    data['RSI'] = RSI_copy
    data['CCI'] = CCI_copy
    data['AD'] = AD_copy
    return data
data = disperse(data)
# print(data)
data = data[['10_SMA','WMA','MACD','W_R','RSI','CCI','AD','MTM','label']]
data.to_csv('600690_trait_data.csv')#导出数据
# data.to_csv('000799_trait_data.csv')#导出数据
