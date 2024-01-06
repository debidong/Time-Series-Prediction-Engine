import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from analysis.models import Result
from forecast.models import File

def infer(pk_result: int, pk_file: int, features: list):
    result = Result.objects.get(pk=pk_result)
    data = File.objects.get(pk=pk_file)

    model = torch.load(result.model)
    data = pd.read_csv()  # 待预测数据集路径
    
    training_features = data[json.loads(result.algo.selected)].values  # 将选择的特征从数据集中抽取出来
    training_goal = data[result.algo.target].values  # 将目标从数据集中抽取出来
        
    scaler_features = MinMaxScaler()
    scaler_goal = MinMaxScaler()
    training_features_normalized = scaler_features.fit_transform(training_features)
    input_dim = training_features_normalized.shape[1]

    X_predict = training_features_normalized[-result.algo.window:]
    X_predict = torch.tensor(np.array(X_predict, dtype=np.float32))
    X_predict = X_predict.resize(1, -result.algo.window, input_dim)
    Y_predictions = model(X_predict)
    Y_predictions = scaler_goal.inverse_transform(Y_predictions.detach().numpy())