# pylint: disable=no-member
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# this is the only thing for which we will beusing external library like 
# hugging face, just to take dataset for training as we can't build very
# large dataset for the required task
from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config,get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import  WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def get_all_sentences(ds, lang):
    """
    getting all the sentences to prepare dataset of the particular language

    """

    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    This Tokenizer code is taken from hugging face library

    config : input configuiration for our model
    ds : dataset to build tokenizer on
    lang : the language of the dataset
    """
    # path of tokenizer file and it depends upon the input language
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        # pre tokenizer stand that we are spliting the tokens depending upon white space
        tokenizer.pre_tokenizer = Whitespace()
        #building traineer to train our tokenizer
        trainer  = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]", "[SOS]", "[EOS]"],
                                min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Loading datasert which takes the configuration of the model
    """
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}',
                    split='train')

    # building tokenizer 
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # spllitng data to 90% training and 10% to validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # method random_split is spliting method from pytorch to split
    # the dataset into train and validation size mentioned
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src,tokenizer_tgt,
                        config['lang_src'],config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src,tokenizer_tgt,
                        config['lang_src'],config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f"Max length of source sentence : {max_len_src}")
    print(f"Max length of target sentence : {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'],config['seq_len'],config['d_model'])
    return model

def train_model(config):
    # define the device on which you want to put your tensors on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    # loading dataset
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr= config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # adding padding token to ignore to add to loss function
    # performing label smooting in order to smooth the probability curve 
    # to make the model less prone to overfitting
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)









