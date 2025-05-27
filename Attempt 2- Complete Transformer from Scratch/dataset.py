import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,
                 ds,
                 tokenizer_src,
                 tokenizer_tgt,
                 src_lang,
                 tgt_lang,
                 seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token_src = tokenizer_src.token_to_id('[SOS]')
        self.eos_token_src = tokenizer_src.token_to_id('[EOS]')
        self.pad_token_src = tokenizer_src.token_to_id('[PAD]')

        self.sos_token_tgt = tokenizer_tgt.token_to_id('[SOS]')
        self.eos_token_tgt = tokenizer_tgt.token_to_id('[EOS]')
        self.pad_token_tgt = tokenizer_tgt.token_to_id('[PAD]')

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # for SOS only in decoder input, EOS in label

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat([
            torch.tensor([self.sos_token_src], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_src], dtype=torch.int64),
            torch.full((enc_num_padding_tokens,), self.pad_token_src, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            torch.tensor([self.sos_token_tgt], dtype=torch.int64),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.full((dec_num_padding_tokens,), self.pad_token_tgt, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_tgt], dtype=torch.int64),
            torch.full((dec_num_padding_tokens,), self.pad_token_tgt, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token_src).unsqueeze(0).unsqueeze(0).int()
        seq_len = self.seq_len
        causal = causal_mask(seq_len).unsqueeze(0)  # [1, seq_len, seq_len], dtype=bool
        pad_mask = (decoder_input != self.pad_token_tgt).unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len], dtype=bool

        decoder_mask = causal & pad_mask  # final shape: [1, seq_len, seq_len], dtype=bool

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1)
    return ~mask  # lower-triangular: True below or on diagonal, False above

# class BilingualDataset(Dataset):
#     def __init__(self,
#                  ds,
#                  tokenizer_src,
#                  tokenizer_tgt,
#                  src_lang,
#                  tgt_lang,
#                  seq_len):
#         super().__init__()

#         self.ds = ds
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
#         self.seq_len = seq_len

#         self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
#         self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
#         self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

#     def __len__(self,):
#         return len(self.ds)
    
#     def __getitem__(self,index: any) -> any:
#         src_target_pair = self.ds[index]
#         src_text = src_target_pair['translation'][self.src_lang]
#         tgt_text = src_target_pair['translation'][self.tgt_lang]

#         enc_input_tokens = self.tokenizer_src.encode(src_text).ids
#         dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

#         enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
#         dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

#         if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
#             raise ValueError("Sentence is too long")
        
#         # adding SOS and EOS to the sorce text
#         encoder_input = torch.cat(
#             [
#                 self.sos_token,
#                 torch.tensor(enc_input_tokens, dtype=torch.int64),
#                 self.eos_token,
#                 torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         decoder_input = torch.cat(
#             [
#                 self.sos_token,
#                 torch.tensor(dec_input_tokens, dtype=torch.int64),
#                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         label = torch.cat(
#             [
#                 torch.tensor(dec_input_tokens, dtype = torch.int64),
#                 self.eos_token,
#                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         assert encoder_input.size(0) == self.seq_len
#         assert decoder_input.size(0) == self.seq_len
#         assert label.size(0) == self.seq_len

#         return {
#             "encoder_input": encoder_input,
#             "decoder_input": decoder_input,
#             "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # 1,1,seq_len
#             "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # 1,seq_len & 1,seq_len,seq_len
#             "label":label,
#             "src_text":src_text,
#             "tgt_text":tgt_text
#         }
    

# def causal_mask(size):
#     mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
#     return mask == 0