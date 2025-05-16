import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm
from pathlib import Path


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

    ds_raw = ds_raw.filter(is_valid)
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

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1, shuffle=True)
    print("Data loader, Source tokenizer and target tokenizer created...")

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

    
def get_model(config,
              vocab_src_len,
              vocab_tgt_len,):
    print("Started Creating model...")
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'], config['seq_len'], config['d_model'])
    print("Model created and loaded successfully using 'build_transformer' function...")
    return model

def train_model(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # writing to tensor board
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    print(f"Started model training...")
    writer.add_text("Model Training Paramters and details",str(config))
    writer.flush()
    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
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
            writer.flush()

            # loss backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # saving the model at the end of each epoch
        if epoch % config['save_every'] == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)







