import torch.nn as nn
import copy
from models.layer.residual_connection import ResidualConnectionLayer

class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0.0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residual_self_attention = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual_cross_attention = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual_position_ff = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # Self-attention + Residual 연결
        tgt = self.residual_self_attention(tgt, lambda tgt: self.self_attention(query=tgt, key=tgt, value=tgt, mask=tgt_mask))
        
        # Cross-attention + Residual 연결
        tgt = self.residual_cross_attention(tgt, lambda tgt: self.cross_attention(query=tgt, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        
        # Position-wise Feed-Forward Network + Residual 연결
        tgt = self.residual_position_ff(tgt, self.position_ff)
        
        return tgt