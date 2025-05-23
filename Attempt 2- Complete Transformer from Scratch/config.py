from pathlib import Path
def get_config():
    return {
        "batch_size": 256,
        "num_epochs" : 200,
        "lr": 10**-3,
        "seq_len" : 128,
        "d_model" : 256,
        "N":4,
        "d_ff":512,
        "head":4,
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "save_every" : 10,
        "warmup_steps": 4000,
        "weight_decay": 0.01,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)