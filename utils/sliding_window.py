import numpy as np


def sliding_window(data, input_window, output_window, output_column):
    train, label = [], []
    for i in range(len(data) - input_window - output_window + 1):
        train.append(data[i:i + input_window])
        label.append(data[i + input_window:i + input_window + output_window, output_column])
    return np.array(train), np.array(label)
