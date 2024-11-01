import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_src_mask(self, src):
        return self.make_pad_mask(src, src)

    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        return pad_mask & seq_mask

    def make_src_tgt_mask(self, src, tgt):
        return self.make_pad_mask(tgt, src)

    def make_pad_mask(self, query, key, pad_idx=1):
        # query와 key의 시퀀스 길이를 저장
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # key에 있는 패딩 토큰에 대해 마스크를 생성 (패딩이 아닌 위치는 True)
        # key 텐서를 확장하여 (batch_size, 1, 1, key_seq_len) 형태로 만들기 위해 unsqueeze를 두 번 사용
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, key_seq_len)

        # key_mask를 query의 시퀀스 길이에 맞게 확장하여 (batch_size, 1, query_seq_len, key_seq_len) 형태로 변환
        key_mask = key_mask.expand(-1, 1, query_seq_len, -1)  # (batch_size, 1, query_seq_len, key_seq_len)

        # query에 있는 패딩 토큰에 대해 마스크를 생성 (패딩이 아닌 위치는 True)
        # query 텐서를 확장하여 (batch_size, 1, query_seq_len, 1) 형태로 만들기 위해 unsqueeze를 두 번 사용
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, query_seq_len, 1)

        # query_mask를 key의 시퀀스 길이에 맞게 확장하여 (batch_size, 1, query_seq_len, key_seq_len) 형태로 변환
        query_mask = query_mask.expand(-1, 1, -1, key_seq_len)  # (batch_size, 1, query_seq_len, key_seq_len)

        # key_mask와 query_mask를 결합하여 두 텐서 모두에서 패딩 위치에 대해 False가 되도록 만듦(cross-attention을 위해 + 일관성 확보)
        mask = key_mask & query_mask
        return mask  # (batch_size, 1, query_seq_len, key_seq_len)


    def make_subsequent_mask(self, query, key):
        # query와 key의 시퀀스 길이를 저장
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        
        # 하삼각 행렬을 생성하여 자기 자신 및 이전 위치까지만 접근할 수 있도록 함
        # torch.tril()을 사용해 query와 key의 시퀀스 길이에 맞는 하삼각 행렬을 생성
        mask = torch.tril(torch.ones(query_seq_len, key_seq_len, dtype=torch.bool, device=query.device))
        return mask  # (query_seq_len, key_seq_len)