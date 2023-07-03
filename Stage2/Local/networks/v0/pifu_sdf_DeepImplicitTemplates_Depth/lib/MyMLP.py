import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MyMLP, self).__init__()
        self.hidden0 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, 512)
        self.hidden2 = torch.nn.Linear(512, 256)
        self.predict = torch.nn.Linear(256, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden0(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.predict(x)
        return x