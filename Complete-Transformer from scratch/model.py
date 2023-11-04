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
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# let's first build layer normalisation
class LayerNormalisation(nn.Module):
    """
    This will normalise the values of vectors between 0 and 1.
    Also, apart from standard normalisation process of finding mean (mu) and variance (sigma),
    ----------
    we  also introduce two parameters, usually called "gamma"(multiplicative) and beta (additive)
    that introduce some fluctuations in the data, because maybe having all values between 0 and 1 may
    be too restrivtive for the netword. the network will learn to tune these
    parameters to introduce fluctuations when necessary.

    epsilon: this parameter is there for the purpose of numerical stability and to avoid
            division by 0


    """

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # mulitplies=d
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    d_model : Dimension of embeddings
    d_ff : embedding dimension to which input d_model embedding is mapped to.

    Since next layer also takes d_model as input so the flow becomes as :

            d_model --> d_ff --> d_model

    """
    def __init__(self, d_model: int, d_ff: int, droput) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.droput = nn.Dropout(droput)
        self.linear_2 = nn.Linear(d_ff,d_model) # W2 and B2

    def forward(self, x):
        """
        Input sentence of shape :
        
        ( Batch, Seq_Len, d_model )
                    |
                    |
        ( Applying Linear Layer )
                    |
                    |        
        ( Batch, Seq_Len, d_ff )
                    |
                    |        
        ( Applying Linear Layer )
                    |
                    |        
        ( Batch, Seq_Len, d_model ) 
        """
        return self.linear_2(self.droput(torch.relu(self.linear_1(x))))

# Most Important Block - Building Multihead Attention
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // heads

        #defining weight parameter metrics
        self.w_q = nn.Linear(d_model, d_model) # query metrics
        self.w_k = nn.Linear(d_model, d_model) # weight metrics
        self.w_v = nn.Linear(d_model, d_model) # value metrics

        self.w_o = nn.Linear(d_model, d_model) # Ouput weight metrics, used after heads concatination
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h , Seq_len, d_k) --> (batch, h , Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        mask is there to hide some words so that they do not 
        interact other words(usefull in algorithms like BERT), so it will guess those words
        compoare the output with actual values and adjust the weights accordingly

        Also, we will be spliting query,key and value into number of heads, and while
        splitting: 
        we will be keeping 'batch_dimension' i.e., query.shape[0] UNCHANGED
        we will keep 'Seq_len' i.e., query.shape[1] UNCHANGED

        only, spliting 'd_model' into (Heads x d_k)
        """
        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)   # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len,heads, d_k) --> (Batch, heads, Seq_len, d_k) 

        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)


        # (Batch, heads, seq_len, d_k) --> (batch, seq_len, heads, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)



