# 时间序列 ARIMA 以及用SVM优化ARIMA
import pandas as pd
import math
import tushare as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import warnings
import datetime
import numpy as np
from pandas import Series,DataFrame
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import copy
from sklearn import svm

#读取数据
deal_data = pd.read_csv('600690.csv', encoding='utf-8',index_col='date')
# deal_data = pd.read_csv('000799.csv', encoding='utf-8',index_col='date')
deal_datas = deal_data['2021-01-04':'2021-06-30']
train = deal_data['2021-01-04':'2021-05-31']
test = deal_data['2021-05-31':'2021-06-30']

# 移动平均图
def draw_trend(timeSeries,size,parameter):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries[parameter].rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = timeSeries[parameter].ewm(span=size).mean()
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()
# draw_trend(deal_datas,10,'close')
# 单独显示股票每日数据图
def draw_ts(timeSeries,title):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.title(title)
    plt.show()
# draw_ts(deal_datas,'Daily data chart of raw data')
'''
　　单位根检验：ADF的零假设是存在一个单位根，另一种选择是没有单位根。也就是说p值越大，我们就越有理由断言存在单位根
'''
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
dfoutput = testStationarity(deal_datas)
# print('原始数据的单位根检验：\n',dfoutput)
# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts,title,lags=31):
    print(title)
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()
# draw_acf_pacf(deal_datas,'原始数据的自相关图和偏相关图')
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

'''平稳处理'''
def smooth(ts,size,parameter):
    ts_log = np.log(ts)
    draw_ts(ts_log,'Daily data chart of log transformed data')    # 对数变换
    draw_trend(ts_log,size,parameter)    # 平滑处理（移动平均法和指数平均法）
    diff_12 = ts_log.diff(size)    # 差分
    diff_12.dropna(inplace=True)
    diff_12_1 = diff_12.diff(1)
    diff_12_1.dropna(inplace=True)
    draw_ts(diff_12_1,'Daily data chart after ninth order difference')
    dfoutput1 = testStationarity(diff_12_1)
    # return(dfoutput1['p-value'])
    print('数据9阶差分后的单位根检验：\n',dfoutput1)
    # decomposition = seasonal_decompose(ts_log, model="additive",freq=30)    # 分解
    # trend = decomposition.trend
    # seasonal = decomposition.seasonal
    # residual = decomposition.resid
# smooth(deal_datas,9,'close') # 通过代码算出当进行九阶差分时p-value = 3.384039e-07最小，效果见生成图
# p_min=[]
# for i in range(1,21):
#     p = smooth(deal_datas,i,'close')
#     p_min.append(p)
# print(p_min.index(min(p_min)))
# print(min(p_min))

# 差分处理 这里只做一阶差分时间序列就接近平稳
def difference(ts,parameter):
    # ts_log = np.log(ts)
    diff = ts.diff(1)
    diff.dropna(inplace=True)
    dfoutput1 = testStationarity(diff)
    # print('数据1阶差分后的单位根检验：\n',dfoutput1)
    # draw_ts(diff,'Daily data chart after first-order difference')
    # draw_acf_pacf(diff,'Autocorrelation diagram and partial correlation diagram of data after first-order difference')
    return diff
diff = difference(deal_datas,'close')

# 当考虑使用ARIMA拟合时间序列数据时，需要找到最优的参数值。
# 这里我们使用“网格搜索”来迭代地探索参数的不同组合。
# 对于参数的每个组合，我们使用statsmodels模块的SARIMAX()函数拟合一个新的季节性ARIMA模型，并评估其整体质量。 
# 一旦我们探索了参数的整个范围，我们的最佳参数集将是我们感兴趣的标准产生最佳性能的参数。

#找合适的p d q
#初始化 p d q
p=d=q=range(0,2)
print("p=",p,"d=",d,"q=",q)
#产生不同的pdq元组,得到 p d q 全排列
pdq=list(itertools.product(p,d,q))
print("pdq:\n",pdq)
seasonal_pdq=[(x[0],x[1],x[2],12) for x in pdq]
print('SQRIMAX:{} x {}'.format(pdq[1],seasonal_pdq[1]))

# 模型选择的网格搜索（或超参数优化）。
'''
在评估和比较配备不同参数的统计模型时，可以根据数据的适合性或准确预测未来数据点的能力，对每个参数进行排序。我们将使用AIC进行衡量。 
在使用大量功能的情况下，适合数据的模型将被赋予比使用较少特征以获得相同的适合度的模型更大的AIC得分。 因此，我们有兴趣找到产生最低AIC值的模型。
下面的代码块通过参数的组合来迭代，并使用SARIMAX函数来适应相应的季节性ARIMA模型。
这里， order参数指定(p, d, q)参数，而seasonal_order参数指定季节性ARIMA模型的(P, D, Q, S)季节分量。 在安装每个SARIMAX()模型后，代码打印出其各自的AIC得分。
'''
def Grid_search(pdq,seasonal_pdq,diff):
    get_pdq = []
    get_aic = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(diff,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                get_aic.append(results.aic)
                get_pdq.append(param)
            except:
                continue
    print(get_aic.index(min(get_aic)))
    print(min(get_aic))
    print(get_pdq[get_aic.index(min(get_aic))])

diff.index = pd.DatetimeIndex(diff.index).to_period('D')
# Grid_search(pdq,seasonal_pdq,diff) # 得到最优参数（0,0,0）x（0,1,1,12）,AIC=180.6937133921199

# Configuration model 配置模型 这里选择最优参数（0,0,0）x（0,1,1,12）,AIC=180.6937133921199
def Configuration_model(diff):
    mod = sm.tsa.statespace.SARIMAX(diff,
                                order=(0, 0, 0),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    # 快速生成模型诊断并调查任何异常行为
    # results.plot_diagnostics(figsize=(15, 12))
    # plt.show()
    return results
results = Configuration_model(diff)

# #进行验证预测
def predict_close(results,train,test):
    pred=results.get_prediction(start=pd.to_datetime('2021-05-31'),dynamic=False)
    # pred_ci=pred.conf_int()
    # print("pred ci:\n",pred_ci) # 获得的是一个预测范围，置信区间
    # print("pred:\n",pred) # 为一个预测对象
    # print("pred mean:\n",pred.predicted_mean) # 为预测的平均值
    predict = pred.predicted_mean
    # 一阶差分还原
    predict_shift = predict.shift(1)
    predict_score = test.copy() 
    for i in range(1,len(predict_shift)):
        predict_score['close'][i] = predict_score['close'][i-1] + predict_shift[i]
    # print(predict_score)
    # 进行绘制预测图像
    draw(train,test,predict_score,'ARIMA')
    print('ARIMA检验')
    errors(train,test,predict_score)
    return predict_score
predict = predict_close(results,train,test)

'''
    计算残差，进而用SVM优化ARIMA模型
'''
# 计算残差
def sub(test,predict):
    subs = test.copy()
    subs = test - predict
    # print(subs)
    data_subs = subs.copy()
    data_subs.columns = ['label1']
    data_subs['label2'] = 0.1
    data_subs['label3'] = 0.1
    data_subs['label4'] = 0.1
    data_subs['label'] = 0.1
    for i in range(1,len(data_subs)):
        data_subs['label2'][i] = data_subs['label1'][i-1]
    for i in range(2,len(data_subs)):
        data_subs['label3'][i] = data_subs['label2'][i-1]
    for i in range(3,len(data_subs)):
        data_subs['label4'][i] = data_subs['label3'][i-1]
    for i in range(4,len(data_subs)):
        data_subs['label'][i] = data_subs['label4'][i-1]
    data_subs = data_subs[4:]
    # print(data_subs)
    # data_list = DataFrame(index = getEveryDay('2021-01-01','2021-05-31'),columns = {'close1':''})
    return data_subs
data_subs = sub(test,predict)

# SVM预测残差
def svm_sub(deal_datas,subs,predict,n):
    x = subs.values
    y = subs['label'].values
    y_train = y[:-3]
    for i in range(len(y_train)):
        y_train[i] = int(y_train[i] * n)
    X_train = x [:-3,:-1]
    y_test = y[-3:]
    for i in range(len(y_test)):
        y_test[i] = int(y_test[i] * n)
    X_test = x [-3:,:-1]
    
    # print(y_train)
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = y_pred/n
    train1 = deal_datas['2021-01-04':'2021-06-25']
    test1 = deal_datas['2021-06-25':'2021-06-30']
    predict1 = test1.copy()
    
    for i in range(3):
        predict1['close'][-3+i] = predict['close'][-3+i] + y_pred[-3+i]
    draw(train1,test1,predict1,'Arima test after SVM optimization')
    print('SVM优化后的ARIMA检验')
    errors(train1,test1,predict1)
svm_sub(deal_datas,data_subs,predict,10)
