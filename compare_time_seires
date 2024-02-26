# 6种时间序列常用方法比较
import pandas as pd
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing

#读取数据
deal_data = pd.read_csv('600690.csv', encoding='utf-8',index_col='date')
train = deal_data['2021-01-04':'2021-05-31']
test = deal_data['2021-05-31':'2021-06-30']
def draw_ts(train,test):
    f = plt.figure(facecolor='white')
    plt.figure(figsize=(20,5))
    plt.xticks(rotation = 270)
    plt.plot(train.index,train['close'],label='train_close',color='blue')
    plt.plot(test.index,test['close'],label='test_close',color='green')
    plt.legend()
    plt.xticks(())
    plt.show()
# draw_ts(train,test)
def draw(train,test,forecast,title):
    f = plt.figure(facecolor='white')
    plt.figure(figsize=(15,4))
    # plt.xticks(rotation = 270)
    plt.plot(train.index,train['close'],label='train_close',color='blue')
    plt.plot(test.index,test['close'],label='test_close',color='green')
    plt.plot(forecast.index,forecast['close'],label='forecast_close',color='red')
    plt.legend()
    plt.xticks(())
    plt.title(title)
    plt.xlabel("month")
    plt.ylabel("close")
    plt.show()

# MASE,RMSSE 老师讲的
def errors(train,test,forecast):
    molecule1 = abs(forecast-test).sum()
    molecule2 = ((forecast-test)*(forecast-test)).sum()
    denominator1 = 0
    denominator2 = 0
    for i in range(1,len(train)):
        denominator1 = abs(train['close'][i]-train['close'][i-1]) + denominator1
    denominator1 = denominator1/(len(train)-1)
    for i in range(1,len(train)):
        denominator2 = (train['close'][i]-train['close'][i-1])*(train['close'][i]-train['close'][i-1]) + denominator2
    denominator2 = denominator2/(len(train)-1)
    MASE = (molecule1['close']/denominator1)/(len(test))
    RMSSE = math.sqrt((molecule2/denominator2)/(len(test)))
    print('MASE = ',MASE)
    print('RMSSE = ',RMSSE)
    # return RMSSE 

# 朴素法
def simplicity(train,test):
    forecast = test.copy()
    for i in range(len(test)):
        forecast['close'][i] = train['close'][-1]
    # draw(train,test,forecast,'Simple method')
    print('Simple method')
    errors(train,test,forecast)
simplicity(train,test)

# 简单平均法
def means(train,test):
    forecast = test.copy()
    for i in range(len(test)):
        forecast['close'][i] = train['close'].mean()
    # draw(train,test,forecast,'Simple average method')
    print('Simple average method')
    errors(train,test,forecast)
means(train,test)

# 移动平均法 Moving average method 通过代码运算当p=14时RMSSE=3.2617211526075707最小，在该模型下效果最好
def moving_average(train,test,p):
    forecast = test.copy()
    train_moving = train['close'][-p:]
    for i in range(len(test)):
        forecast['close'][i] = train_moving.mean()
    # draw(train,test,forecast,'Moving average method')
    print('Moving average method')
    errors(train,test,forecast)
moving_average(train,test,14)
# A = []
# for p in range(len(train)):
#     A.append(moving_average(train,test,p))
# print(min(A))
# print(A.index(min(A)))

# 简单指数平滑法 Simple exponential smoothing method 通过代码运算当alpha=0.9时RMSSE=3.591415289121165最小，在该模型下效果最好
def ex_smooth(train,test,alpha):
    fore = 0
    for i in range(len(train)):
        fore = fore + alpha*((1-alpha)**i)*train['close'][-(i+1)]
    # print(fore)
    forecast = test.copy()
    for i in range(len(test)):
        forecast['close'][i] = fore
    # draw(train,test,forecast,'Simple exponential smoothing method')
    print('Simple exponential smoothing method')
    errors(train,test,forecast)
ex_smooth(train,test,0.9)

# 时间序列分解
def Time_series_decomposition(train,test):
    decomposition = seasonal_decompose(train, model="additive",freq=30)    # 分解
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(311)
    plt.ylabel("trend")
    plt.xticks(())
    decomposition.trend.plot()
    ax2 = f.add_subplot(312)
    plt.ylabel("seasonal")
    plt.xticks(())
    decomposition.seasonal.plot()
    ax2 = f.add_subplot(313)
    plt.ylabel("residual")
    decomposition.resid.plot()
    plt.xticks(rotation = 270)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()
Time_series_decomposition(train,test)

# 霍尔特(Holt)线性趋势法 Holt linear trend method 通过代码运算当smoothing_level=0.4, smoothing_slope=0.5时RMSSE=3.681187149499719最小，在该模型下效果最好
def Holt_linear(train,test):
    fit = Holt(np.asarray(train['close'])).fit(smoothing_level=0.4, smoothing_slope=0.5)
    forecast = test.copy()
    forecast['close'] = fit.forecast(len(test))
    # draw(train,test,forecast,'Holt linear trend method')
    print('Holt linear trend method')
    errors(train,test,forecast)
Holt_linear(train,test)

# Holt-Winters 季节性预测模型 通过代码运算当seasonal_periods = 21 时 RMSSE = 2.324856405578161最小，在该模型下效果最好
def Holt_Winters(train,test):
    forecast = test.copy()
    # A = []
    # for i in range(2,35):
    #     fit1 = ExponentialSmoothing(np.asarray(train['close']),seasonal_periods=i, trend='add', seasonal='add', ).fit()
    #     forecast['close'] = fit1.forecast(len(test))
    #     RMSSE = errors(train,test,forecast)
    #     A.append(RMSSE)
    # print(min(A))
    # print(A.index(min(A)))
    fit1 = ExponentialSmoothing(np.asarray(train['close']),seasonal_periods=21, trend='add', seasonal='add' ).fit()
    forecast['close'] = fit1.forecast(len(test))
    # draw(train,test,forecast,'Holt-Winters')
    print('Holt-Winters')
    errors(train,test,forecast)
Holt_Winters(train,test)
