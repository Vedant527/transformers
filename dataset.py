from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    SOS_EOS_LEN = 2
    SOS_LEN = 1

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
    
        enc_input_tok = self.tokenizer_src.encode(src_text).ids
        dec_input_tok = self.tokenizer_tgt.encode(tgt_text).ids

        enc_pad_size = self.seq_len - len(enc_input_tok) - self.SOS_EOS_LEN
        dec_pad_size = self.seq_len - len(dec_input_tok) - self.SOS_LEN

        if (enc_pad_size < 0 or dec_pad_size < 0):
            raise ValueError('Input sentence too large')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tok, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_pad_size, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tok, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_pad_size, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tok, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_pad_size, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0