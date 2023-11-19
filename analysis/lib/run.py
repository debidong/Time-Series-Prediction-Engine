import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from background_task import background

from .models import FullyConnectedNetwork, LSTMNetwork
from analysis.models import Algorithm, Result
from file.models import File
from utils.db import get_redis_client

def create_optimizer(optimizer_name, model_parameters, lr=0.01):
    '''创建优化器
    根据传入的优化器名称和学习率选择相应的优化器
    优化器可选择adam, sgd, rmsprop
    默认学习率为0.01
    '''
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer

def create_network(network_type: str, input_size, hidden_sizes, output_size, training_window):
    '''创建深度学习模型
    支持全连接神经网络 MLP 和长短时记忆网络 LSTM
    ''' 
    network_type = network_type.lower()
    if network_type == 'mlp':
        network = FullyConnectedNetwork(input_size * training_window, hidden_sizes, output_size)
    elif network_type == 'lstm':
        network = LSTMNetwork(input_size, hidden_sizes, output_size)
    else:
        raise ValueError(f"Unsupported network: {network_type}")
    return network

@background(schedule=0)
def train(dataset: File, algo: Algorithm, training_window=50):
    algo.status = "ING"
    selected_features = pd.read_csv(dataset.path).columns.to_list() #用来预测目标选择的特征
    data = pd.read_csv(dataset.path) #数据集路径
    training_goal = data[algo.target].values #将目标从数据集中抽取出来
    training_features = data[selected_features].values #将特征从数据集中抽取出来
    scaler_features = MinMaxScaler()
    scaler_goal = MinMaxScaler()
    training_features_normalized = scaler_features.fit_transform(training_features)
    training_goal_normalized = scaler_goal.fit_transform(training_goal.reshape(-1, 1))
    X, Y = [], []
    for i in range(len(training_features_normalized) - training_window):
        X.append(training_features_normalized[i:i+training_window])
        Y.append(training_goal_normalized[i+training_window])
    losses = []  # 用于记录每个epoch的loss

    X = torch.tensor(np.array(X, dtype=np.float32))
    Y = torch.tensor(np.array(Y, dtype=np.float32))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=algo.verificationRate, random_state=42)

    train_dataset = TensorDataset(X_train, Y_train)
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=algo.batchSize, shuffle=True, num_workers=4)

    input_dim = training_features_normalized.shape[1]
    hidden_dim = 50
    #model = LSTMModel(input_dim, hidden_dim)
    #model = LinearModel(input_dim * train_window, hidden_dim)
    model = create_network(algo.neuralNetwork, input_dim, [algo.neurons], 1, training_window)
    # 训练
    loss_function = nn.MSELoss()
    optimizer = create_optimizer(optimizer_name=algo.optimization, model_parameters=model.parameters(), lr=algo.learningRate)
    
    
    # 使用tqdm在终端中展示训练进度
    # 使用epoch / algo.epoch在前端中展示训练进度

    for epoch in tqdm(range(algo.epoch)):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            Y_pred = model(batch_X)
            loss = loss_function(Y_pred, batch_y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            progress = epoch / algo.epoch
            redis = get_redis_client()
            redis.set(algo.pk, progress)
    model.eval()

    with torch.no_grad():
        test_predictions = model(X_test)
    test_predictions = scaler_goal.inverse_transform(test_predictions.detach().numpy())
    Y_test_actual = scaler_goal.inverse_transform(Y_test.numpy())

    # 保持两张图片和MSE,RMSE,MAE三个数值
    # 展示前100个预测数据与真实数据的差异
    plt.plot(test_predictions[0:100], label='Predicted Values', color='blue', linestyle='dashed')
    plt.plot(Y_test_actual[0:100], label='Actual values', color='grey')
    plt.legend()
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Values')
    difference_path = '.result/compare_' + str(time.time()) + '.png'
    plt.savefig(difference_path)
    # 展示损失率
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    loss_path = '.result/training_loss_' + str(time.time()) + '.png'
    plt.savefig(loss_path)

    mse = np.mean((test_predictions - Y_test_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_predictions - Y_test_actual))
    result = Result(
                algo=algo,
                difference=difference_path,
                loss=loss_path,
                mse=mse,
                rmse=rmse,
                mae=mae,
                )
    result.save()
    algo.status = "FIN"
    return 0