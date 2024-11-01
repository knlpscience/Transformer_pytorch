import torch.nn as nn
import copy
from models.layer.residual_connection import ResidualConnectionLayer

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff, norm, dr_rate=0.0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residual_self_attention = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual_position_ff = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, src, src_mask):
        # Self-attention + Residual 연결
        src = self.residual_self_attention(src, lambda src: self.self_attention(query=src, key=src, value=src, mask=src_mask))
        
        # Position-wise Feed-Forward Network + Residual 연결
        src = self.residual_position_ff(src, self.position_ff)
        
        return src