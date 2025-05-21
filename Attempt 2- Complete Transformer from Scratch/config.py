from pathlib import Path
def get_config():
    return {
        "batch_size": 2,
        "num_epochs" : 100,
        "lr": 10**-3,
        "seq_len" : 64,
        "d_model" : 128,
        "N":2,
        "d_ff":256,
        "head":4,
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "save_every" : 10
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)