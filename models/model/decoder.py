import torch.nn as nn
import copy

class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(n_layer)])
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, encoder_out, tgt_mask, src_tgt_mask)
        
        return self.norm(tgt)