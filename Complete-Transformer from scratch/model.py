# pylint: disable=no-member
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

class PositionalEncoding(nn.Module):
    """
    d_model : dimension of embedding vector
    seq_len : length of the sentence, also one vector for each position
    dropout : to make the model less overfit
    ----------------------
    -> To create the positional encoding we will be using the PE formula from the original paper
    and these are as follows:

    1. PE(pos,2i) = sin( pos/10000 ** ( 2i/d_model ) )
    2. PE(pos,2i + 1) = cos( pos/10000 ** ( 2i/d_model ) )

    Where, first fomula is applied to all the words in odd position and 
    the second formula is applied to all the words in even position

    Note: in our case we are calculting positonal encoding in log space for numerical stability

    ----------------------------
    For REGISTER_BUFFER()
    If you have parameters in your model,
    which should be saved and restored in the state_dict, 
    but not trained by the optimizer, you should register them as buffers.
    Buffers won’t be returned in model.parameters(), 
    so that the optimizer won’t have a change to update them.

    """

    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # creating a matrix(positional encoding) of shape (seq_len,d_model)
        pe = torch.zeros(seq_len, d_model)
        # creating a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        # applying the sin to even odd posiont , IMPORTANT: python index starts from 0 thus
        # by looking it, confusion can occur as formula is applied to 0,2 ...
        pe[:, 0::2] = torch.sin(position * div_term)
        # applying the sin to even odd posiont , IMPORTANT: python index starts from 0 thus
        # by looking it, confusion can occur as formula is applied to 0,2 ...
        pe[:, 1::2] = torch.cos(position * div_term)

        # modifying positional encoding to take care of batch of dimension
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        # register_buffer => Tensor which is not a parameter,
        #  but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        self.register_buffer()

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
