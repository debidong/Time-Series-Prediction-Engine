import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from analysis.models import Result
# from forecast.models import File
from file.models import File

def infer(pk_algo: int, pk_file: int, step: list):
    algo = Result.objects.get(pk=pk_algo)
    data = File.objects.get(pk=pk_file)
    result = Result.objects.get(algo=algo)
    model = torch.load(algo.model_path, map_location=torch.device('cpu'))
    data = pd.read_csv(result.model)  # 待预测数据集路径
    try:
        training_features = data[json.loads(result.algo.selected)].values  # 将选择的特征从数据集中抽取出来
        training_goal = data[result.algo.target].values  # 将目标从数据集中抽取出来
    except:
        # 特征或目标不存在
        return None
    scaler_features = MinMaxScaler()
    scaler_goal = MinMaxScaler()
    training_features_normalized = scaler_features.fit_transform(training_features)
    input_dim = training_features_normalized.shape[1]

    X_predict = training_features_normalized[-result.algo.window:]
    X_predict = torch.tensor(np.array(X_predict, dtype=np.float32))
    X_predict = X_predict.resize(1, -result.algo.window, input_dim)
    Y_predictions = []
    for s in step:
        for _ in range(s):
            Y_prediction = model(X_predict)
            Y_prediction = scaler_goal.inverse_transform(Y_prediction.detach().numpy())
            # 更新输入序列，将当前预测值添加到末尾，以供下一个时间步预测使用
            X_predict = torch.cat((X_predict[:, 1:], torch.tensor(Y_prediction[-1, :]).view(1, 1, -1)), dim=1)
        Y_predictions.append(Y_prediction.flatten())
    Y_predictions = scaler_goal.inverse_transform(Y_predictions.detach().numpy())
    print(Y_predictions)
    return Y_predictions