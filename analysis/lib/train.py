import json
import os
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from utils.storage import MODEL_PATH, RESULT_PATH

from .models import FullyConnectedNetwork, LSTMNetwork
from analysis.models import Algorithm, Result
from utils.db import redis_conn

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

# TODO: 添加参数异常处理
# 目前已经添加了简单的try except处理，用于将中断的训练记录回滚至INI状态
@shared_task
def train(pk: int):
    try:
        channel_layer = get_channel_layer()
        group_name = "training_progress"

        algo = Algorithm.objects.get(pk=pk)
        dataset = algo.dataset

        training_window = algo.window
        T = algo.step
        selected_features = json.loads(algo.selected)
        data = pd.read_csv(dataset.path) #数据集路径
        training_goal = data[algo.target].values #将目标从数据集中抽取出来
        training_features = data[selected_features].values #将特征从数据集中抽取出来
        scaler_features = MinMaxScaler()
        scaler_goal = MinMaxScaler()
        training_features_normalized = scaler_features.fit_transform(training_features)
        training_goal_normalized = scaler_goal.fit_transform(training_goal.reshape(-1, 1))
        X, Y = [], []
        # 超参数
        for i in range(training_window, len(training_features_normalized) - T):
            X.append(training_features_normalized[i-training_window:i, :])
            Y.append(training_goal_normalized[i:i+T, 0])
        losses = []  # 用于记录每个epoch的loss

        X = torch.tensor(np.array(X, dtype=np.float32))
        Y = torch.tensor(np.array(Y, dtype=np.float32))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=algo.verificationRate, random_state=42)

        device = ''
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = 'cuda'
        else:
            device = 'cpu'

        # 将数据移动到GPU上
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        train_dataset = TensorDataset(X_train, Y_train)
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=algo.batchSize, shuffle=True, num_workers=0)

        input_dim = training_features_normalized.shape[1]
        neurons = ast.literal_eval(algo.neurons)
        neurons = [int(x) for x in neurons]
        model = create_network(algo.neuralNetwork, input_dim, neurons, T, training_window)
        model = model.to('cuda')

        # 训练
        loss_function = nn.MSELoss()
        optimizer = create_optimizer(optimizer_name=algo.optimization, model_parameters=model.parameters(), lr=algo.learningRate)
        epoches = algo.epoch
        algo.status = "ING"
        algo.save()
        
        # 训练开始时发送一回0%进度
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'send_training_progress',
                'algoID': pk,
                'progress': 0.0,
                'message': ""
            }
        )

        # 使用epoch / algo.epoch在前端中展示训练进度
        for epoch in range(epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                Y_pred = model(batch_X)
                loss = loss_function(Y_pred, batch_y)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                del batch_X, batch_y, Y_pred

            progress = round(epoch / algo.epoch,2)
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'send_training_progress',
                    'algoID': pk,
                    'progress': progress,
                    'message': ""
                }
            )
            redis_conn.set(pk, progress)
            torch.cuda.empty_cache()
        # 训练结束后发送一回100%进度
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'send_training_progress',
                'algoID': pk,
                'progress': 1.0,
                'message': ""
            }
        )
        redis_conn.set(pk, progress)
        model.eval()

        with torch.no_grad():
            test_predictions = model(X_test)
            test_predictions_cpu = test_predictions.cpu()
            test_predictions = scaler_goal.inverse_transform(test_predictions_cpu.numpy())
            Y_test_actual = scaler_goal.inverse_transform(Y_test.cpu().numpy())

        model_path = MODEL_PATH + algo.name + '.pth'
        torch.save(model, model_path)

        result_directory = RESULT_PATH
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        # 保持两张图片和MSE,RMSE,MAE三个数值
        # 展示至多前100个预测数据与真实数据的差异
        
        plt.clf()
        plt.plot(test_predictions.flatten()[:100], label='Predicted Values', color='blue', linestyle='dashed')
        plt.plot(Y_test_actual.flatten()[:100], label='Actual values', color='grey')
        plt.legend()
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Index')
        plt.ylabel('Values')
        difference_path = RESULT_PATH+'/figure/compare_' + algo.name + '.png'
        plt.savefig(difference_path)
        # 展示损失率
        plt.clf()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        loss_path = RESULT_PATH+'/figure/training_loss_' + algo.name + '.png'
        plt.savefig(loss_path)

        mse = np.mean((test_predictions - Y_test_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_predictions - Y_test_actual))
        result = Result(
                    algo=algo,
                    difference='./api'+difference_path[1:],
                    loss='./api'+loss_path[1:],
                    mse=mse,
                    rmse=rmse,
                    mae=mae,
                    model=model_path
                    )
        result.save()
        algo.status = "FIN"
        algo.save()
        redis_conn.delete(pk)
        return 0
    except:
        # 训练出现错误
        # TODO: 在多处引入try catch以便区分具体的错误
        algo.status = "ERR"
        algo.save()
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'send_training_progress',
                'algoID': pk,
                'progress': -1,
                'message': "内存溢出，尝试减小batchsize",
                'detail': {
                    'name': algo.name,
                }
            }
        )
        redis_conn.delete(pk)
        return -1
