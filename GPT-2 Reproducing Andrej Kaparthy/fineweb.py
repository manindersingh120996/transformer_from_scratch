# """
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

# Downloads and tokenizes the data and saves data shards to disk.
# Run simply as:
# $ python fineweb.py

# Will save shards to the local directory "edu_fineweb10B"
# """

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
train_shard_size = 60000000  # 60M tokens per training shard
val_shard_size = train_shard_size // 2  # Validation shard is half the size of a training shard
num_train_shards = 3
num_val_shards = 1

data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(data_cache_dir, exist_ok=True)

# Load dataset
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True
)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Process and create shards
total_shards = num_train_shards + num_val_shards
shard_sizes = [train_shard_size] * num_train_shards + [val_shard_size] * num_val_shards

token_count = 0
shard_index = 1
shard_tokens = np.empty((shard_sizes[shard_index - 1],), dtype=np.uint16)
progress_bar = tqdm(total=sum(shard_sizes), unit="tokens", desc="Creating shards")

for doc in fw:
    tokens = tokenize(doc)
    for token in tokens:
        if token_count < shard_sizes[shard_index - 1]:
            shard_tokens[token_count] = token
            token_count += 1
            progress_bar.update(1)
        if token_count == shard_sizes[shard_index - 1]:
            # Save the shard
            shard_type = "train" if shard_index <= num_train_shards else "val"
            filename = os.path.join(data_cache_dir, f"edufineweb_{shard_type}_{shard_index:06d}.npy")
            write_datafile(filename, shard_tokens[:token_count])
            print(f"Shard saved to {filename}")
            
            # Move to next shard if available
            shard_index += 1
            if shard_index > total_shards:
                break
            
            token_count = 0
            shard_tokens = np.empty((shard_sizes[shard_index - 1],), dtype=np.uint16)

progress_bar.close()

# # Configuration for a single shard
# local_dir = "edu_fineweb10B"
# remote_name = "sample-10BT"
# # shard_size = int(1e8)  # 100M tokens per shard

# # Create the local directory
# DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
# os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# # Load only a small subset of the dataset (enough for one shard)
# fw = load_dataset(
#     "HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True
# )

# # Activate the below code to use Hindi dataset for training as well
# #code for fineweb-hindi model 
# # fw = load_dataset(
# #     "KathirKs/fineweb-edu-hindi", split="train", streaming=True
# # )


# # Roughly Limiting the dataset to ~100MB worth of tokens
# shard_size = 60000000 # roughly this much tokens will be there
# subset = fw.take(shard_size)  # Assuming 60000000 samples will fit in 100MB; adjust as needed.

# # Initialize the tokenizer
# enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens["<|endoftext|>"]  # End of text token

# # Tokenizer function
# def tokenize(doc):
#     tokens = [eot]
#     tokens.extend(enc.encode_ordinary(doc["text"]))
#     tokens_np = np.array(tokens)
#     assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Tokens too large for uint16"
#     return tokens_np.astype(np.uint16)

# # Write shard to disk
# def write_datafile(filename, tokens_np):
#     np.save(filename, tokens_np)

# # Process the subset and create a single shard
# all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # Preallocate buffer
# token_count = 0

# # code for creating progress bar and code execution
# progress_bar = tqdm(total=shard_size, unit="tokens", desc="Creating single shard")

# for doc in subset:
#     tokens = tokenize(doc)
#     if token_count + len(tokens) <= shard_size:
#         # Add tokens to the shard
#         all_tokens_np[token_count : token_count + len(tokens)] = tokens
#         token_count += len(tokens)
#         # updating progress bar
#         progress_bar.update(len(tokens))
#     else:
#         # Stop after completing the first shard
#         remainder = shard_size - token_count
#         all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
#         progress_bar.update(remainder)
#         break

# progress_bar.close()

# # Save the single shard
# filename = os.path.join(DATA_CACHE_DIR, "edufineweb_train_000001")
# write_datafile(filename, all_tokens_np[:token_count])
# print(f"Shard saved to {filename}")
