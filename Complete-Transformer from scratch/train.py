import torch
import torch.nn as nn

# this is the only thing for which we will beusing external library like 
# hugging face, just to take dataset for training as we can't build very
# large dataset for the required task

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import  WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

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

