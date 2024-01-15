import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from pprint import pprint
import random
from models.BiLSTM import LSTMModel
from models.Transformer import TransformerTimeSeries
from models.CNN_Transformer import Cnn_TransformerTimeSeries
from utils.sliding_window import sliding_window
import argparse
import os
import copy


def load_data():
    # 读取 CSV 文件
    data = pd.read_csv('ETTh1.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    data.drop('date', axis=1, inplace=True)
    data_old = pd.read_csv('ETTh1.csv')
    data_old.drop('date', axis=1, inplace=True)

    train_data = pd.read_csv('data/train_set.csv')
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data['year'] = train_data['date'].dt.year
    train_data['month'] = train_data['date'].dt.month
    train_data['day'] = train_data['date'].dt.day
    train_data['hour'] = train_data['date'].dt.hour
    train_data.drop('date', axis=1, inplace=True)
    train_data_old = pd.read_csv('data/train_set.csv')
    train_data_old.drop('date', axis=1, inplace=True)

    valid_data = pd.read_csv('data/validation_set.csv')
    valid_data['date'] = pd.to_datetime(valid_data['date'])
    valid_data['year'] = valid_data['date'].dt.year
    valid_data['month'] = valid_data['date'].dt.month
    valid_data['day'] = valid_data['date'].dt.day
    valid_data['hour'] = valid_data['date'].dt.hour
    valid_data.drop('date', axis=1, inplace=True)
    valid_data_old = pd.read_csv('data/validation_set.csv')
    valid_data_old.drop('date', axis=1, inplace=True)

    test_data = pd.read_csv('data/test_set.csv')
    test_data['date'] = pd.to_datetime(test_data['date'])
    test_data['year'] = test_data['date'].dt.year
    test_data['month'] = test_data['date'].dt.month
    test_data['day'] = test_data['date'].dt.day
    test_data['hour'] = test_data['date'].dt.hour
    test_data.drop('date', axis=1, inplace=True)
    test_data_old = pd.read_csv('data/test_set.csv')
    test_data_old.drop('date', axis=1, inplace=True)
    # return data, train_data, valid_data, test_data
    return data, data_old, train_data, train_data_old, valid_data, valid_data_old, test_data, test_data_old


# # 定义随机种子固定的函数
# def get_random_seed(seed):
#     random.seed(seed)  # 确保每次划分的训练集、验证集、测试集一致
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def de_standardize_draw(target, predict, model_name):
    header = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    # index = random.choice(range(32))
    #
    # target = target[index].reshape(-1, 7).cpu().numpy()
    # predict = predict[index].reshape(-1, 7).cpu().numpy()
    # # (96, 7)
    # target = self.standard.inverse_transform(target)
    # predict = self.standard.inverse_transform(predict)

    for i in range(7):
        x = range(target.shape[0])
        trg = target[:, i]
        pred = predict[:, i]

        plt.figure()
        plt.plot(x, trg, label='target', color='orange')
        plt.plot(x, pred, label='predict', color='green')
        plt.legend()

        plt.xlabel('hour')
        plt.ylabel('value')
        plt.title(f'{header[i]}')
        plt.savefig(f'pic/{header[i]}_{model_name}.png')
        plt.close()


def main(args):
    seed_everything(args.seed)
    # seed = args.seed
    # torch.seed(seed)
    # random.seed(seed)

    # data, train_data, valid_data, test_data = load_data()
    data, data_old, train_data, train_data_old, valid_data, valid_data_old, test_data, test_data_old = load_data()

    # 参数设置
    input_window = args.input_window
    print("input_window:", input_window)
    output_window = args.output_window
    output_column = [i for i in range(7)]
    # valid_ratio = args.valid_ratio
    # test_ratio = args.test_ratio
    #
    # # 划分训练集和测试集
    # data_len = len(data)
    # valid_size = int(data_len * valid_ratio)
    # test_size = int(data_len * test_ratio)

    # X_train, X_valid, X_test = data[:-(test_size + valid_size)], data[-(test_size + valid_size):-test_size], data[-test_size:]

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    train_data = scaler.transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)

    scaler_old = StandardScaler()
    data_old = scaler_old.fit_transform(data_old)


    X_train_sliding, y_train_sliding = sliding_window(train_data, input_window, output_window, output_column)
    X_valid_sliding, y_valid_sliding = sliding_window(valid_data, input_window, output_window, output_column)
    X_test_sliding, y_test_sliding = sliding_window(test_data, input_window, output_window, output_column)

    print("X_train_sliding")
    print(X_train_sliding.shape, y_train_sliding.shape)

    # X_data_sliding, y_data_sliding = sliding_window(data, input_window, output_window, output_column)
    # data_sliding = list(zip(X_data_sliding, y_data_sliding))
    #
    # random.shuffle(data_sliding)
    # X_data_sliding[:], y_data_sliding[:] = zip(*data_sliding)
    # X_train_sliding, y_train_sliding = X_data_sliding[:-(test_size + valid_size)], y_data_sliding[:-(test_size + valid_size)]
    # X_valid_sliding, y_valid_sliding = X_data_sliding[-(test_size + valid_size):-test_size], y_data_sliding[-(test_size + valid_size):-test_size]
    # X_test_sliding, y_test_sliding = X_data_sliding[-test_size:], y_data_sliding[-test_size:]

    # print(X_train_sliding[0])

    # # 数据归一化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_valid = scaler.transform(X_valid)
    #
    # test_scaler = StandardScaler()
    # X_test = test_scaler.fit_transform(X_test)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_sliding, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_sliding, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid_sliding, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_sliding, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_sliding, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_sliding, dtype=torch.float32)
    # print(X_train_tensor[0])
    # print(y_valid_tensor.size())

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    cuda = args.device
    # 初始化模型、优化器和损失函数
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    input_size = data.shape[1]
    print("input_size:", input_size)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    # output_size = output_window
    output_size = output_window * args.output_size
    print("output_size", output_size)
    bidirectional = True

    if args.model == "LSTM":
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional).to(device)
    elif args.model == "Transformer":
        model = TransformerTimeSeries(input_size, output_size, output_size).to(device)
    else:
        model = Cnn_TransformerTimeSeries(input_size, output_size, output_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    L1criterion = nn.L1Loss(size_average=True)

    losses = []
    # 训练模型
    num_epochs = args.epochs
    patience = args.patience
    p = 0
    min_loss = float('inf')
    model_path = args.model_path

    # Train
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # [bz * 96 * features_num]
            optimizer.zero_grad()
            outputs = model(inputs)  # [bz * 672]
            # loss = criterion(outputs, targets.reshape(targets.size(0), -1))
            loss = criterion(outputs.reshape(outputs.size(0), targets.size(1), -1), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            losses.append(loss.item())
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}', end=' ')

        # dev
        model.eval()
        with torch.no_grad():
            X_valid_tensor = X_valid_tensor.to(device)
            y_valid_tensor = y_valid_tensor.to(device)
            predictions = model(X_valid_tensor)
            # predictions = predictions.reshape(predictions.shape[0], 96, -1)
            dev_loss = criterion(predictions, y_valid_tensor.reshape(y_valid_tensor.size(0), -1))
            print(f"Dev Loss: {dev_loss}")

        if dev_loss > min_loss:
            p += 1
            if p == patience:
                print(f"Epoch {epoch - 9} is best")
                print(f"Best dev loss = {min_loss}")
                print("Finish Training")
                break
        else:
            print("Saving Model")
            torch.save(model.state_dict(), f"model_files/{model_path}.best.pth")
            min_loss = dev_loss
            p = 0
        print("Saving Temp Model")
        torch.save(model.state_dict(), f"model_files/{model_path}.tmp.pth")


    # 滑窗预测
    if args.model == "LSTM":
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional).to(device)
    elif args.model == "Transformer":
        model = TransformerTimeSeries(input_size, output_size, output_size).to(device)
    else:
        model = Cnn_TransformerTimeSeries(input_size, output_size, output_size).to(device)
    print("Loading Model")
    model.load_state_dict(torch.load(f"model_files/{model_path}.best.pth"))
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        predictions = model(X_test_tensor).cpu()
        print(predictions.size())
        predictions = predictions.reshape(predictions.shape[0], output_window, -1)
        test_mse_loss = criterion(predictions, y_test_tensor)
        test_mae_loss = L1criterion(predictions, y_test_tensor)
        print("MSE_Loss:", test_mse_loss)
        print("MAE_Loss:", test_mae_loss)

    # print(results)
    prediction = predictions[0]
    prediction = scaler_old.inverse_transform(prediction)
    # print(prediction)
    label = scaler_old.inverse_transform(y_test_tensor[0])
    # print(label)
    de_standardize_draw(label, prediction, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='随机数种子')
    parser.add_argument('--input_window', default=96, type=int, help='输入的时间步长')
    parser.add_argument('--output_window', default=96, type=int, help='预测的时间步长')
    parser.add_argument('--valid_ratio', default=0.2, type=float, help='验证集比例')
    parser.add_argument('--test_ratio', default=0.2, type=float, help='测试集比例')
    parser.add_argument('--batch_size', default=256, type=int, help='batch大小')
    parser.add_argument('--device', default=4, type=int, help='GPU')
    parser.add_argument('--hidden_size', default=512, type=int, help='隐藏层维度')
    parser.add_argument('--num_layers', default=2, type=int, help='LSTM层数')
    parser.add_argument('--output_size', default=7, type=int, help='输出的特征数量')
    parser.add_argument('--lr', default=0.0001, type=float, help='学习率')
    parser.add_argument('--epochs', default=1000, type=int, help='迭代轮数')
    parser.add_argument('--patience', default=10, type=int, help='早停')
    parser.add_argument('--model_path', default="BiLSTM.v", type=str, help='模型保存路径')
    parser.add_argument('--model', default="LSTM", type=str, help='使用的模型')

    args = parser.parse_args()
    main(args)
