import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.CNN import context_embedding


class Cnn_TransformerTimeSeries(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Cnn_TransformerTimeSeries, self).__init__()
        self.cnn = context_embedding(input_size, 512, 9)
        self.fc = torch.nn.Linear(input_size, 512)
        self.positional_embedding = torch.nn.Embedding(512, 512)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)

        self.fc1 = torch.nn.Linear(512*96, output_size)

    def forward(self, x):
        z = x.permute(0, 2, 1)
        z_embedding = self.cnn(z).permute(2, 0, 1)
        pos_encoder = (
            torch.arange(0, 96, device="cuda:0")
            .unsqueeze(0)
            .repeat(z_embedding.size(1), 1)
        )
        positional_embeddings = self.positional_embedding(pos_encoder).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding)
        transformer_embedding = transformer_embedding.permute(1, 0, 2)

        output = self.fc1(transformer_embedding.reshape(transformer_embedding.size(0), -1))

        return output