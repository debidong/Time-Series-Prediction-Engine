import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

from file.models import File

def AR(pk: int, target, order, window, step):
    """自回归模型预测时序数据
    pk: 数据集主键
    order: 自回归模型阶数
    window: 预测的数据窗口的大小，即用前多少个数据点来构建自回归模型
    step: 预测数据步长，即使用模型预测后续多少个数据点
    """
    file = File.objects.get(pk=pk)
    data = pd.read_csv(file.path)
    
    window = int(window)
    order = int(order)
    step = int(step)
    
    training_window = data[target].values[-window:]
    ar_model = AutoReg(training_window, lags=order)
    ar_result = ar_model.fit()
    
    forcast = ar_result.predict(start=len(training_window), end=len(training_window)+step-1)
    print(forcast)
    return 0

def ARIMA_model(pk: int, target, order, window, step):
    """ARIMA模型预测时序数据
    pk: 数据集主键
    target: 目标变量名称
    order: ARIMA模型的阶数，例如(1,1,1)表示AR阶数为1，差分阶数为1，MA阶数为1
    window: 预测的数据窗口的大小，即用前多少个数据点来构建ARIMA模型
    step: 预测数据步长，即使用模型预测后续多少个数据点
    """
    file = File.objects.get(pk=pk)
    data = pd.read_csv(file.path)
    order = tuple(map(int, order.split(',')))
    order = (1,1,1)
    window = int(window)
    step = int(step)
    
    differenced_series = data[target].diff().dropna()
    
    arima_model = ARIMA(differenced_series, order=order, seasonal_order=None)
    arima_result = arima_model.fit()
    
    forecast = arima_result.predict(start=len(differenced_series), end=len(differenced_series)+step-1, typ='levels')
    
    print(forecast)
    return 0