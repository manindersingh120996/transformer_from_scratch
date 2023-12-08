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
import warnings
 
from tqdm import tqdm

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Runs encoder only once.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculating the output
        out = model.decoder(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token

        prob = model.project(out[:,-1])
        # selecting the token with the maximum probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim= 1)

        # concatinating output to deoderinput an it becomes input for the next iteration
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)







def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # size of the control window(using the default value)
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src,
                                        tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print to the console
            # we are using print_msg provided by tqdm instead of python's inbuilt print
            # because we are using tqdm progress bar and using python's print will
            # interfere in the tqdm progress bar
            # printing to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

    # if write





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

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'Preprocessing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) #(B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device) #(B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device) #(B,1,1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device) #(B, 1, Seq_Len, Seq_Len)

            # Run the Tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_Len,d_model)
            proj_output = model.project(decoder_output) #(B, Seq_Len, tgt_vocab_size)

            label = batch['label'].to(device) #(B, Seq_Len)
            # below code transforms the (B, Seq_Len, tgt_vocab_size) --> (B * Seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # to show the progress bar
            batch_iterator.set_postfix({f"loss": f"{loss.item() : 6.3f}"})

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()


            # Backpropagate the loss
            loss.backward()

            # update the weights of the model
            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'],device,
                            lambda msg: batch_iterator.write(msg), global_step, writer )

            global_step += 1

        # saving teh model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # it's always suggested to save the optimizer state along with the weights of the model
        # it helps in having resumable training on saved model
        torch.save(
            {
               'epoch' : epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict' : optimizer.state_dict(),
               'global_step': global_step
            }, model_filename
        )

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
 








