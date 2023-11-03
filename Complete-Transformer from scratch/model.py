import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    INPUT:
    d_model : dimension of embeddings
    vocab_size : size of vocabulary
    """
    def __init(self, d_model:int,vocab_size:int):
        # super() is used to give access to methods and properties
        # of a parent or sibling class.
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        #embedding layer from Torch to generate embedding for a given number of
        # d_model dimension
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        # according to paper "attention is all you need" embeddings vectors are
        # multiplied by square root of embedding dimension.
        return self.embedding(x) * math.sqrt(self.d_model)