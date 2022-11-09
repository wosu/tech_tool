import random
import time
from abc import ABC

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader,IterableDataset
from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class LibSvmDataset(Dataset):
    def __init__(self,X,Y,feature_num:int):
        # X为csr_matrix, Y为ndarray
        # X,Y = load_svmlight_file(data_path)
        self.sample_size = X.shape[0]
        self.feature_size = X.shape[1]
        assert self.feature_size == feature_num
        # X:shape(batch_size,num_features)  Y:shape(batch_size,1)
        self.X = pd.DataFrame(X.toarray(),dtype=np.float32)
        self.Y = Y.astype(np.float32)
        self.Y = self.Y.reshape((self.Y.shape[0],1))
        print(self.X)
        print(self.Y)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        labels = self.Y[idx]
        features = self.X.iloc[idx].to_numpy()
        return features,labels


class LibSvmIterableDataset(IterableDataset):
    def __init__(self,data_path,feature_num):
        self.data_path = data_path
        self.feature_num = feature_num

    def process_oneline(self,line):
        line = line.split(' ')
        label, values = int(line[0]), line[1:]
        value = torch.zeros((self.feature_num))
        for item in values:
            idx, val = item.split(':')
            value[int(idx) - 1] = float(val)
        return value,label

    def __iter__(self):
        with open(self.data_path,mode='r') as reader:
            for line in reader:
                yield self.process_oneline(line.strip())
    # def __iter__(self):
    #     shuffle_buffer = []
    #     with open(self.file_path, 'r') as fp:
    #         index = 0
    #         for line in fp:
    #             shuffle_buffer.append(self.process_line(line.strip("\n")))
    #             index += 1
    #             if index > self.buffer_size:
    #                 break
    #
    #     with open(self.file_path, 'r') as fp:
    #         for line in fp:
    #             evict_idx = random.randint(0, self.buffer_size - 1)
    #             yield shuffle_buffer[evict_idx]
    #             shuffle_buffer[evict_idx] = self.process_oneline(line.strip("\n"))


if __name__ == "__main__":
    data_path = r"../model_datas/sex/v9_train_datas.libsvm"
    X, Y = load_svmlight_file(data_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)
    train_dataset = LibSvmDataset(X_train, Y_train, 281)
    test_dataset = LibSvmDataset(X_test, Y_test, 281)
