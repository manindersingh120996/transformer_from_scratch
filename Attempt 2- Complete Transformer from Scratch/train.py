import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm
from pathlib import Path
import torchmetrics
import inspect
import math
from datasets import load_dataset, load_from_disk, DatasetDict

config = get_config()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    print("SOS index:", sos_idx)
    print("EOS index:", eos_idx)

    # computeing the encoder output once and using it for all the required times during decodeing from decoder
    encoder_output = model.encode(source,source_mask)
    # initialising the decoder input with the SOS token, as it is always the first token of decoder to generate
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        # if length of output reaches max_len of generation
        if decoder_input.size(1) == max_len:
            break

        # Building mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #calculating the output of the decoder
        out = model.decode(encoder_output,source_mask,decoder_input, decoder_mask)

        # get the next token based on previous tokens
        prob = model.project(out[:,-1])

        # sleecting the token with maximum probabaility (as it is greedy search), but there exists other methods as well
        # and here top_p, top_k, temperature like parameters can be introduced
        _, next_word  = torch.max(prob, dim = 1)
        print("Predicted token index:", next_word.item())

        decoder_input = torch.cat(
            [decoder_input,
             torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],
             dim=1)
        decoded_tokens = decoder_input.squeeze(0).tolist()
        decoded_text = tokenizer_tgt.decode(decoded_tokens)
        print("Current partial prediction:", decoded_text)


        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
    

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step,
                   writer, num_examples = 2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # size of the control window (just use a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch Size must be 1 for the Validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask,tokenizer_src,tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            # print_msg('-'*console_width)
            # print_msg(model_out)
            # print_msg(type(model_out))
            # print_msg('-'*console_width)
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            #printing to the consle
            print_msg('-'*console_width)
            print_msg(f"SOURCE : {source_text}")
            print_msg(f"TARGET : {target_text}")
            print_msg(f"PREDICTED : {model_out_text}")

            if count == num_examples:
                print_msg("-"*console_width)
                break

        if writer:
            # evaluate the character error rate
            # compute the char error rate
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()
                
def get_filtered_dataset(config, tokenizer_src, tokenizer_tgt):
    cache_path = Path("cached_filtered_dataset")

    if cache_path.exists():
        print("🔁 Loading pre-filtered dataset from disk...")
        ds_filtered = load_from_disk(cache_path)
    else:
        print("📥 Loading dataset from HuggingFace...")
        ds_raw = load_dataset("philomath-1209/english-to-hindi-high-quality-training-data", split="train")
        # ds_raw = load_dataset("cfilt/iitb-english-hindi", split="train")

        print("🔍 Filtering dataset (this may take time)...")
        seq_limit = config['seq_len'] - 10

        def is_valid(example):
            src_ids = tokenizer_src.encode(example['translation'][config['lang_src']]).ids
            tgt_ids = tokenizer_tgt.encode(example['translation'][config['lang_tgt']]).ids
            return len(src_ids) <= seq_limit and len(tgt_ids) <= seq_limit

        ds_filtered = ds_raw.filter(is_valid)

        print("📦 Saving filtered dataset to disk...")
        ds_filtered.save_to_disk(cache_path)

    return ds_filtered

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds,lang,):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    print("Starting building TOkenizer for ->", lang)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("Tokenizer Building Completed")
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('cfilt/iitb-english-hindi',split='train')
    print("Dataset Loaded...")

    #building tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])
    seq_limit = config['seq_len'] - 10

    # Filter sentences that are too long
    def is_valid(example):
        src_ids = tokenizer_src.encode(example['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(example['translation'][config['lang_tgt']]).ids
        return len(src_ids) <= seq_limit and len(tgt_ids) <= seq_limit

    # ds_raw = ds_raw.filter(is_valid)
    ds_raw = get_filtered_dataset(config, tokenizer_src, tokenizer_tgt)
    filtered_data = list(ds_raw)
    filtered_data = filtered_data[:100]
    print(f"Dataset Filtered for sentences having length less then {config['seq_len']}")
    print(f"Len of Data Set : {len(filtered_data)}")
    writer = SummaryWriter(config['experiment_name'])
    writer.add_scalar("Dataset Used for training Size : " ,len(filtered_data))

    # train val split 90:10
    train_ds_size = int(0.9 * len(filtered_data))
    val_ds_size = len(filtered_data) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(filtered_data, [train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    print("Dataset devided into train and eval")


    max_len_src = 0
    max_len_tgt = 0

    print(f"In a step to detect the max length of source and target sentences...\nIt can take some time...")
    for item in tqdm(filtered_data):
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
    print('\n')

    print(f'Max length of soruce sentence: {max_len_src}')
    print(f'Max length of soruce sentence: {max_len_tgt}')
    writer.add_scalar("Max length of soruce sentence:" ,max_len_src)
    writer.add_scalar("Max length of target sentence:" ,max_len_tgt)

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],#len(train_ds),#config['batch_size'],
                                   shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1, shuffle=True)
    print("Data loader, Source tokenizer and target tokenizer created...")

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

    
def get_model(config,
              vocab_src_len,
              vocab_tgt_len,):
    print("Started Creating model...")
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'], 
                              config['seq_len'], config['d_model'],config['N'],
                              config['head'],0.0,config['d_ff'])
    print("Model created and loaded successfully using 'build_transformer' function...")
    return model

def configure_optimizers(model,weight_decay, learning_rate, device_type):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
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
# to include function to have a warmup step and a variable learning rate as per origninal paper
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = config['num_epochs'] // 4
max_steps = config['num_epochs'] # 

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

def train_model(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model = torch.compile(model)
    # writing to tensor board
    writer = SummaryWriter(config['experiment_name'])

    # optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], eps=1e-9)
    optimizer = configure_optimizers(model,
                                    weight_decay = 0.1,
                                    learning_rate = 6e-4,
                                    device_type = device)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    print(f"Started model training...")
    writer.add_text("Model Training Paramters and details",str(config))
    writer.flush()
    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            # torch.cuda.empty_cache()
            

            encoder_input = batch['encoder_input'].to(device) # (B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device) # (B, Seq_Len)
            # encoder mask to mask padding tokens
            encoder_mask = batch['encoder_mask'].to(device) #(B,1,1 Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, Seq_len, Seq_len)

            # passing the sentences through the transformer model
            encoder_output = model.encode(encoder_input, encoder_mask) # B, seq_len, d_model
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # b, seq_len, d_model
            
            # converting to get the output token reppresentation
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device) # B, seq_len

            # b, seq_len, tgt_vocab_size -> b* seq_len, tgt_vocab_size
            # labels with raw score and not one hot encoded
            # projecting the output to the target vocab size and then calculating the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss :" : f"{loss.item():6.3f}"})

            # logging the loss to tensorboard
            writer.add_scalar('traininig loss' , loss.item(), global_step)
            
            
            # loss backpropagation
            loss.backward()
            lr = get_lr(epoch)
            # print('learning rate : ', lr)
            writer.add_scalar('learning rate', lr, global_step)
            writer.flush()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'],
                       device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # saving the model at the end of each epoch
        if (epoch+1) % config['save_every'] == 0:
            model_filename = get_weights_file_path(config, f'{epoch+1:02d}')
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)







