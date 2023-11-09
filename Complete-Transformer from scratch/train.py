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

