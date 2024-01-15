import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(LSTMModel, self).__init__()
        self.bidirectional_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                          bidirectional=bidirectional)
        # self.bidirectional_lstm_2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True,
        #                                   bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out_bidirectional, _ = self.bidirectional_lstm(x)
        out = F.tanh(out_bidirectional[:, -1, :])
        out = self.fc(out)
        return out
