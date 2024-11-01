import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=4096):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        
        encoding = torch.zeros(max_len, d_embed)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        # 모델에서 학습되지 않도록 requires_grad=False 설정
        self.register_buffer('positional_encoding', encoding.unsqueeze(0))

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.positional_encoding[:, :seq_len, :].to(x.device)
        
        return x + pos_embed