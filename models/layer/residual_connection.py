import torch.nn as nn

class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm, dr_rate=0.0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x, sub_layer):
        normalized_x = self.norm(x)
        
        sub_layer_out = sub_layer(normalized_x)
        sub_layer_out = self.dropout(sub_layer_out)
        
        return x + sub_layer_out