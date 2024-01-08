import re
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from utils.storage import RESULT_PATH

from forecast.models import File

def AR(pk: int, target, order, window, step) -> np.ndarray:
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
    
    forecast = ar_result.predict(start=len(training_window), end=len(training_window)+step-1)

    plt.switch_backend('Agg')

    plt.clf()
    plt.plot(list(range(window, window+step)), forecast, label='Prediction', linestyle='--')
    plt.plot(list(range(window)), training_window, label='Training Data', linestyle='-')
    plt.xlabel('Serial')
    plt.ylabel(target)
    plt.title('AR Model Prediction')
    plt.legend()

    path = 'result/forecast_AR_' + file.name + '.png'
    plt.savefig(path)

    forecast_df = pd.DataFrame({target: forecast})
    csv_path = RESULT_PATH + '/forecast_AR_' + file.name + '_result.csv'
    forecast_df.to_csv(csv_path, index=False)

    return forecast, path
    # [240.34873895 240.30165088 240.25839607 240.21866248 240.18216345
    # 240.14863568 240.11783728 240.08954607 240.06355793 240.0396854]

def ARIMA_model(pk: int, target, order, window, step) -> pd.core.series.Series:
    """ARIMA模型预测时序数据
    pk: 数据集主键
    target: 目标变量名称
    order: ARIMA模型的阶数，例如(1,1,1)表示AR阶数为1，差分阶数为1，MA阶数为1
    window: 预测的数据窗口的大小，即用前多少个数据点来构建ARIMA模型
    step: 预测数据步长，即使用模型预测后续多少个数据点
    """
    file = File.objects.get(pk=pk)
    data = pd.read_csv(file.path)
    order = ast.literal_eval(order)
    window = int(window)
    step = int(step)

    training_window = data[target].tail(window)
    arima_model = ARIMA(training_window, order=order)
    arima_result = arima_model.fit()
    
    forecast = arima_result.predict(start=len(training_window), end=len(training_window)+step-1, typ='levels')
    
    print(forecast.values)

    plt.switch_backend('Agg')

    plt.clf()
    plt.plot(list(range(window, window + step)), forecast.values, label='Prediction', linestyle='--')
    plt.plot(list(range(window)), training_window, label='Training Data', linestyle='-')
    plt.xlabel('Serial')
    plt.ylabel(target)
    plt.title('ARIMA Model Prediction')
    plt.legend()

    path = 'result/forecast_ARIMA_' + file.name + '.png'
    plt.savefig(path)

    forecast_df = pd.DataFrame({target: forecast})
    csv_path = RESULT_PATH + '/forecast_ARIMA_' + file.name + '_result.csv'
    forecast_df.to_csv(csv_path, index=False)

    return forecast, path
    # 101236    240.377907
    # 101237    240.377388
    # 101238    240.377376
    # 101239    240.377376
    # 101240    240.377376
    # 101241    240.377376
    # 101242    240.377376
    # 101243    240.377376
    # 101244    240.377376
    # 101245    240.377376
    # Name: predicted_mean, dtype: float64
    

def Fbprophet(pk: int, target, window, step, periods, freq) -> pd.core.frame.DataFrame:
    """Fbprophet模型预测时序数据
    pk: 数据集主键
    target: 目标变量名称
    window: 预测的数据窗口的大小，即用前多少个数据点来构建模型
    step: 预测数据步长，即使用模型预测后续多少个数据点
    periods: 预测数据的期数，即使用模型预测后续多少个数据点
    freq: 预测数据的频率
        D: 天
        H: 小时
        T/min: 分钟
        S: 秒
        L/ms: 毫秒
        U/us: 微秒
        N/ns: 纳秒
        B: 交易日
        W: 每周
        M: 月末
        Q: 季度末
        A/Y: 年末
    """
    window = int(window)
    step = int(step)

    file = File.objects.get(pk=pk)
    data = pd.read_csv(file.path)
    datetime_regex = datetime_regex = r'\b(\d{4}/\d{1,2}/\d{1,2}(?: \d{2}:\d{2}:\d{2})?)|(\d{4}-\d{1,2}-\d{1,2}(?: \d{2}:\d{2}:\d{2})?)\b'
    matched_column = ""
    for col in data.columns:
        if re.search(datetime_regex, str(data[col].iloc[0])):
            matched_column = col
            break

    data.rename(columns={
            matched_column: 'ds',
            target: 'y'
        }, inplace=True)
    
    training_window = data[['ds', 'y']].tail(window)
    
    model = Prophet()
    model.fit(training_window)
    future = model.make_future_dataframe(periods=int(periods), freq=freq)
    forecast = model.predict(future)[window:window+step]

    training_window['ds'] = pd.to_datetime(training_window['ds'])

    plt.switch_backend('Agg')

    plt.clf()
    plt.plot(list(range(window, window + len(forecast))), forecast['yhat'], label='Prediction', linestyle='--')
    plt.plot(list(range(window)), training_window['y'], label='Training Data', linestyle='-')
    plt.xlabel('Serial')
    plt.ylabel(target)
    plt.title('Fbprophet Model Prediction')
    plt.legend()

    path = 'result/forecast_fbprophet_' + file.name + '.png'
    plt.savefig(path)
    
    forecast_df = pd.DataFrame({target: forecast['yhat']})
    csv_path = RESULT_PATH + '/forecast_Fbprophet_' + file.name + '_result.csv'
    forecast_df.to_csv(csv_path, index=False)

    return forecast, path
    #                      ds        trend     yhat_lower  ...  multiplicative_terms_lower  multiplicative_terms_upper         yhat
    # 0   2023-04-23 08:51:43   239.238079     238.931432  ...                         0.0                         0.0   239.238079
    # 1   2023-04-23 08:56:43   239.245132     238.939292  ...                         0.0                         0.0   239.245132
    # 2   2023-04-23 09:01:42   239.252161     238.953928  ...                         0.0                         0.0   239.252161
    # 3   2023-04-23 09:06:43   239.259238     238.963208  ...                         0.0                         0.0   239.259238
    # 4   2023-04-23 09:11:43   239.266291     238.973073  ...                         0.0                         0.0   239.266291
    # ..                  ...          ...            ...  ...                         ...                         ...          ...
    # 995 2024-09-02 03:11:30  1792.750004 -136788.429805  ...                         0.0                         0.0  1792.750004
    # 996 2024-09-03 03:11:30  1795.879950 -137293.589742  ...                         0.0                         0.0  1795.879950
    # 997 2024-09-04 03:11:30  1799.009897 -137750.974159  ...                         0.0                         0.0  1799.009897
    # 998 2024-09-05 03:11:30  1802.139843 -138180.571122  ...                         0.0                         0.0  1802.139843
    # 999 2024-09-06 03:11:30  1805.269790 -138647.247467  ...                         0.0                         0.0  1805.269790

    # [1000 rows x 13 columns]