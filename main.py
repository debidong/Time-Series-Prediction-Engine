import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import MinMaxScalerp
def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
    super(LSTMModel, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(hidden_dim, output_dim)

def forward(self, x):
    out, _ = self.lstm(x)
    out = self.relu(out[:, -1, :])
    out = self.linear(out)
    return out


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
class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        print(x.size())
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


def create_optimizer(optimizer_name, model_parameters, lr=0.01):
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

def create_network(network_type, input_size, hidden_sizes, output_size, training_window):
    network_type = network_type.lower()
    if network_type == 'mlp':
        network = FullyConnectedNetwork(input_size * training_window, hidden_sizes, output_size)
    elif network_type == 'lstm':
        network = LSTMNetwork(input_size, hidden_sizes, output_size)
    else:
        raise ValueError(f"Unsupported network: {network_type}")
    return network

def main(network_type, optimize_method, layer, hidden_state, learning_rate, training_epoch, training_window, batch_size, ratio):
    goal = 'ucn_' #训练目标
    selected_features = ['uan_', 'ubn_', 'ia_', 'ib_', 'ic_', 'pt_w', 'wpa_'] #用来预测目标选择的特征
    data = pd.read_csv('training_data/time_series_data/10000027440006.csv') #数据集路径
    training_goal = data[goal].values #将目标从数据集中抽取出来
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

    train_dataset = TensorDataset(X_train, Y_train)
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    input_dim = training_features_normalized.shape[1]
    hidden_dim = 50
    #model = LSTMModel(input_dim, hidden_dim)
    #model = LinearModel(input_dim * train_window, hidden_dim)
    model = create_network(network_type, input_dim, hidden_state, 1, training_window)
    # 训练
    loss_function = nn.MSELoss()
    optimizer = create_optimizer(optimizer_name=optimize_method, model_parameters=model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(training_epoch)):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            Y_pred = model(batch_X)
            loss = loss_function(Y_pred, batch_y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
    model.eval()

    with torch.no_grad():
        test_predictions = model(X_test)
    test_predictions = scaler_goal.inverse_transform(test_predictions.detach().numpy())
    Y_test_actual = scaler_goal.inverse_transform(Y_test.numpy())

    #结果展示需要展示下面保存的两张图片，和MSE,RMSE,MAE三个数值，保存到数据库中。
    #展示前100个预测数据与真实数据的差异
    plt.plot(test_predictions[0:100], label='Predicted Values', color='blue', linestyle='dashed')
    plt.plot(Y_test_actual[0:100], label='Actual values', color='grey')
    plt.legend()
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.savefig('compare_' + str(time.time()) + '.png')
    plt.show()
    plt.clf()

    mse = np.mean((test_predictions - Y_test_actual) ** 2)
    # 计算RMSE
    rmse = np.sqrt(mse)
    # 计算MAE
    mae = np.mean(np.abs(test_predictions - Y_test_actual))
    print("均方误差 (MSE):", mse)
    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss_' + str(time.time()) + '.png')

    plt.show()


if __name__ == '__main__':
    network_type = 'MLP' #网络选择
    optimize_method = 'adam' #优化算法
    layer = 2 #网络层数
    hidden_state = [20] #神经元数目
    learning_rate = 0.01 #学习率
    training_epoch = 10 #训练轮数
    training_window = 50 #窗口大小
    batch_size = 256 #batch_size
    ratio = 0.2 #验证保留集比例
    main(network_type, optimize_method,
         layer, hidden_state, learning_rate, training_epoch, training_window, batch_size, ratio)