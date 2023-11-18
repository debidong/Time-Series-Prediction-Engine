import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNetwork, self).__init__()
        # 构建所有隐藏层
        layers = []
        for i in range(len(hidden_sizes)):
            in_features = input_size if i == 0 else hidden_sizes[i - 1]
            out_features = hidden_sizes[i]
            layers.append(nn.Linear(in_features, out_features))
        # 添加输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # 将所有层存储为ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # 依次经过每个层，这里使用ReLU作为激活函数
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # 最后一层不加激活函数
        x = self.layers[-1](x)
        return x

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LSTMNetwork, self).__init__()
        # 定义LSTM层
        lstm_layers = []
        for i in range(len(hidden_sizes)):
            lstm_input_size = input_size if i == 0 else hidden_sizes[i - 1]
            lstm_layers.append(nn.LSTM(lstm_input_size, hidden_sizes[i], batch_first=True))
        self.lstm_layers = nn.ModuleList(lstm_layers)
        # 定义全连接层
        linear_layers = [nn.Linear(hidden_sizes[-1], output_size)]
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, x):
        # 经过LSTM层
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        # 经过全连接层
        x = x[:, -1, :]  # 取LSTM最后一个时间步的输出
        for linear in self.linear_layers:
            x = linear(x)
        return x

# class LinearModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim=1):
#         super(LinearModel, self).__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         print(x.size())
#         out = self.linear1(x)
#         out = self.relu(out)
#         out = self.linear2(out)
#         return out