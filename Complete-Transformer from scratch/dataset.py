# pylint: disable=no-member
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        sos_id = tokenizer_src.token_to_id('[SOS]')
        print('-----',sos_id)
        eos_id = tokenizer_src.token_to_id('[EOS]')
        print('-----',eos_id)
        pad_id = tokenizer_src.token_to_id('[PAD]')
        print('-----',pad_id)
        # # initialising the special tokens
        # self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])[0]], dtype = torch.long)
        # self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])[0]], dtype = torch.long)
        # self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])[0]], dtype = torch.long)
        # Convert to torch tensors
        self.sos_token = torch.tensor([sos_id], dtype=torch.long)
        self.eos_token = torch.tensor([eos_id], dtype=torch.long)
        self.pad_token = torch.tensor([pad_id], dtype=torch.long)

    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # converting extracted text into tokens and coverting it into ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # reason for -2 : since apart from sentence we will also be adding EOS and SOS token 
        # in the sentence thus -2 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # here will be adding only 1 special token that is SOS token
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # adding SOS and EOS to the source text

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.long)
            ]
        )
        
        # adding SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.long),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.long)
            ]
        )

        # add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.long)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            "encoder_input":encoder_input, # (seq_len),
            "decoder_input":decoder_input, # (seq_len)
            # since we are increasing the len of the input tokens by adding PAD tokens
            # but we don't want to include them in training and ignore them, thus encoder_mask
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1, Seq_Len)
            # for DECODER we need SPECIAL MASK, that is each word can only look previous words and non padding words only
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_len) & (1, Seq_len, Seq_len)
            "label": label, #(Seq_len)
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
