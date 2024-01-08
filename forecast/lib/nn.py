import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from analysis.models import Algorithm, Result
from forecast.models import File

def infer(pk_algo: int, pk_file: int):

    algo = Algorithm.objects.get(pk=pk_algo)
    file = File.objects.get(pk=pk_file)
    result = Result.objects.get(algo=algo)
    window = int(algo.window)
    step = int(algo.step)
    model = torch.load(result.model, map_location=torch.device('cpu'))
    data = pd.read_csv(file.path)  # 待预测数据集路径
    try:
        training_features = data[json.loads(result.algo.selected)].values  # 将选择的特征从数据集中抽取出来
        training_goal = data[result.algo.target].values  # 将目标从数据集中抽取出来
    except:
        return None
    # 数据标准化
    scaler_features = MinMaxScaler()
    scaler_goal = MinMaxScaler()

    features_normalized = scaler_features.fit_transform(training_features.reshape(-1, 1))
    goal_normalized = scaler_goal.fit_transform(training_goal.reshape(-1, 1))

    # 准备输入序列
    input_dim = features_normalized.shape[1]
    X_predict = features_normalized[-window:]  # 使用前 window 个数据
    X_predict = torch.tensor(np.array(X_predict, dtype=np.float32)).reshape(1, -1, input_dim)

    Y_predictions = []

    for _ in range(int(step)):
        Y_prediction = model(X_predict)
        Y_prediction = scaler_goal.inverse_transform(Y_prediction.detach().numpy())
        X_predict = torch.cat((X_predict[:, 1:], torch.tensor(Y_prediction[-1, :]).view(1, 1, -1)), dim=1)
        Y_predictions.append(Y_prediction.flatten())

    Y_predictions = np.array(Y_predictions)
    print(Y_predictions)


    plt.switch_backend('Agg')
    plt.clf()

    forecast = Y_predictions.flatten()
    training_window = training_goal[-window:]

    plt.plot(list(range(window, window + int(step))), forecast, label='Prediction', linestyle='--')
    plt.plot(list(range(window)), training_window, label='Training Data', linestyle='-')
    plt.xlabel('Serial')
    plt.ylabel(result.algo.target)
    plt.title(algo.neuralNetwork+' Model Prediction')
    plt.legend()

    path = "result/forecast_" + algo.neuralNetwork + "_" + file.name + '.png'
    plt.savefig(path)

    return forecast, path
