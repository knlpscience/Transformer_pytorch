import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0.0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h            # 각 헤드의 차원
        self.q_fc = copy.deepcopy(qkv_fc)  # Query FC 레이어
        self.k_fc = copy.deepcopy(qkv_fc)  # Key FC 레이어
        self.v_fc = copy.deepcopy(qkv_fc)  # Value FC 레이어
        self.out_fc = out_fc               # 출력 FC 레이어
        self.dropout = nn.Dropout(p=dr_rate)

    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (batch_size, h, seq_len, d_k)
        # mask: (batch_size, 1, seq_len, seq_len)
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K^T, (batch_size, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(self.d_k)       # scaling
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)  # 마스킹
        attention_prob = F.softmax(attention_score, dim=-1)           # (batch_size, h, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value)                     # (batch_size, h, seq_len, d_k)
        return out

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_embed)
        # mask: (batch_size, seq_len, seq_len)
        # return value: (batch_size, seq_len, d_embed)
        batch_size = query.size(0)

        def transform(x, fc_layer):
            out = fc_layer(x)                                 # (batch_size, seq_len, d_model)
            out = out.view(batch_size, -1, self.h, self.d_k)  # (batch_size, seq_len, h, d_k)
            out = out.transpose(1, 2)                         # (batch_size, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc)  # (batch_size, h, seq_len, d_k)
        key = transform(key, self.k_fc)      # (batch_size, h, seq_len, d_k)
        value = transform(value, self.v_fc)  # (batch_size, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask)                    # (batch_size, h, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        out = self.out_fc(out)  # (batch_size, seq_len, d_embed)
        return out