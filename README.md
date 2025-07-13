# Readme for Attempt 2 - Complete Transfomer from Scratch 

This repository contains a complete implementation of the Transformer model, built from scratch following the seminal "Attention Is All You Need" paper.</br> 
The codebase is designed for clarity and modularity, allowing for easy understanding and experimentation with different aspects of the Transformer architecture.

## Project Structure

The project is organized into several distinct files, each responsible for a specific part of the Transformer pipeline:

-   **`model.py`**: This file encapsulates the core components of the Transformer model. It includes the implementation of input embeddings, positional encodings, multi-head attention mechanisms, and the encoder-decoder blocks,and all other major componenets form the original paper thus building complete encoder-decoder transfomer

-   **`train.py`**: This script orchestrates the entire training and validation process. It handles data loading `(using datasets.py file)`, initializes the Transformer model`(using model.py file)` with configurations defined in `config.py`, and manages the training loop, including optimization and evaluation.

-   **`config.py`**: A centralized location for all model hyperparameters and training settings. This includes parameters such as the number of layers, hidden unit sizes, learning rate, dropout rates, and other configurable options, making it straightforward to modify and experiment with different setups.

-   **`datasets.py`**: This file is dedicated to data loading and preprocessing. It currently supports an English-Hindi parallel dataset , the dataset which we will be using is `"philomath-1209/english-to-hindi-high-quality-training-data` and preparing it for use by the Transformer model.
-   the format in which dataset is currently expected is can be checked by visting the relevant hugging-face dataset repository, but the dataset file can also be modified easily to work with data stored in other formats.

## Getting Started

To run the Transformer model and begin training, simply execute the following command from the root of the repository:

```bash
python -m train
