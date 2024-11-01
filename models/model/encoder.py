import torch.nn as nn
import copy

class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(n_layer)])
        self.norm = norm

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return self.norm(src)