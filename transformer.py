import torch
import torch.nn as nn

from components.feed_fwd import FeedForward
from components.input_embedding import InputEmbeddings
from components.multihead_attention import MultiheadAttention
from components.positional_encoding import PositionalEncoding
from components.projection_layer import ProjectionLayer
from decoder.decoder import Decoder, DecoderBlock
from encoder.encoder import Encoder, EncoderBlock

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)
    
    @staticmethod
    def init_transformer(src_vocab_size: int, 
                         tgt_vocab_size: int, 
                         src_sequence_length: int, 
                         tgt_sequence_length: int, 
                         d_model: int = 512,
                         N: int = 6,
                         num_heads: int = 8,
                         dropout: float = 0.1,
                         d_ff: int = 2048):
        src_embed = InputEmbeddings(d_model, src_vocab_size)
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        src_pe = PositionalEncoding(d_model, src_sequence_length, dropout)
        tgt_pe = PositionalEncoding(d_model, tgt_sequence_length, dropout)

        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiheadAttention(d_model, num_heads, dropout)
            feed_fwd_block = FeedForward(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_fwd_block, dropout)
            encoder_blocks.append(encoder_block)

        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiheadAttention(d_model, num_heads, dropout)
            decoder_cross_attention_block = MultiheadAttention(d_model, num_heads, dropout)
            feed_fwd_block = FeedForward(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_fwd_block, dropout)
            decoder_blocks.append(decoder_block)

        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pe, tgt_pe, projection_layer)
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer


    