from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import code

# code.interact(local=locals())
# torch.set_default_dtype(torch.half)
class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 ### if error, look for it
        # regulatisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not a 'bias' actually but is a 'mask' instaed, but since following OpenAI/HF naming 
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)

        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B,nh,T,hs)
        y = F.scaled_dot_product_attention(q,k,v , is_causal = True, )

        # output projection
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
        


class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
import inspect
class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self,idx, targets = None):
        # idx is of shape (B,T)
        B,T = idx.size()
        assert T <= self.config.block_size,f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # shape(T)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # if master_process:
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls,model_type):
        """Loads pretrained GPT-2 model weights from the huggingface"""
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2' : dict(n_layer = 12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium' : dict(n_layer = 24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large' : dict(n_layer = 36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl' : dict(n_layer = 48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        #create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config=config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # initalizzing hugging face/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring  all of the paramters are aligned and mathc in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    

# simple run :
    # python train_gpt2.py
# DDP launch for e.g in my case 2 GPUs
    # torchrun --standalone --nproc_per_node = 2 train_gpt2.py
from torch.distributed import init_process_group,destroy_process_group
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# setting up ddp (Distributed Data Parallel)
# torchrun command sets the env variable RANK, Local_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK',-1)) != -1 # is this a ddp run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to the rand
    assert torch.cuda.is_available(),"For now we need cuda for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #this process will do logging,checkpoint
else:
    #vanilla , non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attenpt to auto detect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends,'mps') and torch.backend.mps.is_available():
        device = 'mps'
    print(f'Using device : {device}')




#____________________________________________________________________
#code for autodetect the training device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device : {device}")
    
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# creating______________________ DATASET _______ for the training
import tiktoken
import time
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt','r') as f:
#     text = f.read()
# text = text[:10000]
# tokens = enc.encode(text)
# B, T = 4,32
# buf = torch.tensor(tokens[:B*T + 1])
# x = buf[:-1].view(B,T)
# y = buf[1:].view(B,T)
runpod_absolute_path = "/root/transformer_from_scratch/GPT-2 Reproducing Andrej Kaparthy/input.txt"
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_process
        
        # at init load tokens from disk and store them in memory
        # with open('input.txt','r') as f:
        with open(runpod_absolute_path,'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # making changes in below code to accomodate the DDP and MultiGPU training
        # data splitting
        self.current_position = self.B * self.T * self.process_rank # for each process it's batch will start at rank times B times T

    def next_batch(self):
        # as well as makinng the changes in below code to always load the data on corresponding GPU accordingly 
        # and current position is advanced in such a way that it get's diffent data from every other GPU always
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        # buf.to(dtype = torch.float16)
        x = (buf[:-1]).view(B,T) # inputs
        y = (buf[1:]).view(B,T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x,y


# gradient accumulation step
total_batch_size = 524288 # 2 ** 19, ~0.5M, in number of tokens
B = 8
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0 , "make sure total batch is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size : {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print("I am GPU", ddp_rank)
# print("Testing completed")
# import sys;sys.exit(0)

train_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_process = ddp_world_size)

# to set floating point calculation change in order to reduce training time
torch.set_float32_matmul_precision('high')
#defaul= highest, high and medium also available


##################  

#___________TESTING
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
model = GPT(GPTConfig(vocab_size=50304)) # 50304 in order to convert it to nice number that is 
# number for the power of 2 that is adding fake tokens
model.to(device)

# tested with RunPod Linux server and this command is working in RunPOD and efficiently reducing computation time
#=================
model = torch.compile(model) 
#======================

# Now after setting up DDP it is required and mandatory to wrap the model in DDP
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iter steps
    if it<warmup_steps:
        return max_lr * (it + 1) / warmup_steps
       
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    
    #3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to zero
    return min_lr + coeff * (max_lr - min_lr)


#optmizer step
# optimizer = torch.optim.AdamW(model.parameters(),
#                               lr = 3e-4, 
#                               betas = (0.9,0.95),
#                               eps = 1e-8)

optimizer = raw_model.configure_optimizers(weight_decay = 0.1,
                                       learning_rate = 6e-4,
                                       device_type = device)

from tqdm.auto import tqdm
for step in range(max_steps):
    t0 = time.time()
    loss_accum = 0.0    
    optimizer.zero_grad()

    # gradient accumulation over the micro_steps
    for micro_steps in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        # below calculation is for the purpose of reducing the precision
        # point calculation opertaions to speed up the process
        # and precision used is bfloat 16 and not float16 as it requires the
        # gradient optimization techincquies to perform in float16 data types
        with torch.autocast(device_type=device, dtype = torch.bfloat16):
            logits, loss = model(x.to(device),y.to(device))
        # print(loss)
        loss = loss / grad_accum_steps
        # print(loss)
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_steps == grad_accum_steps - 1)
        loss.backward() # backward
    
    if ddp: # only for the loss accumulated not for synchronising gradients which is done automatically in above line of if ddp:
        dist.all_reduce(loss_accum,op = dist.ReduceOp.AVG)


    # import code; code.interact(local=locals())
    # gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    # finiding learning rate for each iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1-t0)
    if master_process:
        print(f"\nStep {step + 1}| lr {lr:.4e} | loss: {loss_accum.item():.4f} | in time : {dt:.2f}ms\
 | tokens/sec: {tokens_per_sec:.2f} | norm : {norm :.4f}")

if ddp:
    destroy_process_group()

import sys
sys.exit(0)

####_____________TESTING END

num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2') # using the weights of pretrained 
                                        # GPT2 model downloaded from the Hugging face

model = GPT(GPTConfig())                                      # using our skelton model created using the random
                                        # weights to inference the model,
print("Model didn't crasged yet")
model.eval()
model.to(device)



#prefix toekns
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype = torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) #(5,8)
# x = tokens.to(device)


# generateing, right now x is (B,T) where B = 5, T= 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits,loss = model(x) # B,T, Vocab_size
        # taking the logits at the last position
        logits = logits[:, -1, :] # B, vocab_size
        # get the probabilities
        probs = F.softmax(logits, dim = -1)
        # doing the top-k sampling og 50 which is by default in hugingface pipeline
        # topk_probs here becomes (5,50), tok_inidzes is 5,50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # selecting a token from tok-k probabilities
        ix = torch.multinomial(topk_probs,1) # B,1
        # gather the corresponding indices
        xcol = torch.gather(topk_indices,-1,ix)
        # append to the sequence
        x = torch.cat((x,xcol), dim =1)

# priniting the generated texts
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
