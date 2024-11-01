import torch
import torch.nn as nn
import copy

from models.model.transformer import Transformer
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock
from models.layer.multi_head_attention import MultiHeadAttentionLayer
from models.layer.feedforward import PositionWiseFeedForwardLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding


def set_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("cuda"),
                max_len=256,
                d_embed=512,
                n_layer=6,
                d_model=512,
                h=8,
                d_ff=2048,
                dr_rate=0.1,
                norm_eps=1e-5):

    # 소스와 타겟을 위한 토큰 임베딩
    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    tgt_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=tgt_vocab_size)

    # 소스와 타겟 임베딩을 위한 위치 인코딩
    pos_embed = PositionalEncoding(d_embed=d_embed, max_len=max_len)

    # 소스와 타겟을 위한 Transformer 임베딩과 드롭아웃 설정
    src_embed = TransformerEmbedding(token_embed=src_token_embed, pos_embed=copy.deepcopy(pos_embed), dr_rate=dr_rate)
    tgt_embed = TransformerEmbedding(token_embed=tgt_token_embed, pos_embed=copy.deepcopy(pos_embed), dr_rate=dr_rate)

    # 멀티헤드 어텐션과 피드 포워드 레이어
    attention = MultiHeadAttentionLayer(
        d_model=d_model,
        h=h,
        qkv_fc=nn.Linear(d_embed, d_model),
        out_fc=nn.Linear(d_model, d_embed),
        dr_rate=dr_rate
    )
    
    position_ff = PositionWiseFeedForwardLayer(
        fc1=nn.Linear(d_embed, d_ff),
        fc2=nn.Linear(d_ff, d_embed),
        dr_rate=dr_rate
    )

    # 레이어 정규화
    norm = nn.LayerNorm(d_embed, eps=norm_eps)

    # 인코더와 디코더 블록
    encoder_block = EncoderBlock(
        self_attention=copy.deepcopy(attention),
        position_ff=copy.deepcopy(position_ff),
        norm=copy.deepcopy(norm),
        dr_rate=dr_rate
    )

    decoder_block = DecoderBlock(
        self_attention=copy.deepcopy(attention),
        cross_attention=copy.deepcopy(attention),
        position_ff=copy.deepcopy(position_ff),
        norm=copy.deepcopy(norm),
        dr_rate=dr_rate
    )

    # 지정된 레이어 수를 가진 인코더 및 디코더 스택
    encoder = Encoder(encoder_block=encoder_block, n_layer=n_layer, norm=copy.deepcopy(norm))
    decoder = Decoder(decoder_block=decoder_block, n_layer=n_layer, norm=copy.deepcopy(norm))

    # 출력 생성기 레이어
    generator = nn.Linear(d_model, tgt_vocab_size)

    # 모든 구성 요소를 포함한 Transformer 모델 생성
    model = Transformer(
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        encoder=encoder,
        decoder=decoder,
        generator=generator
    ).to(device)

    model.device = device  # 모델에 디바이스 정보를 저장

    return model