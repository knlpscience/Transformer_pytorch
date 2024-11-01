import torch.nn as nn

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2, dr_rate=0.0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x