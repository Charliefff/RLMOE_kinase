import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config


# protein | selfies

class TrainingScript:
    def __init__(self, 
                config: dict,
                selfies_path: str, 
                protein_path: str, 
                output_dir: str,
                run_name: str):
        
        self.selfies_path = selfies_path
        self.protein_path = protein_path
        self.output_dir = output_dir
        self.run_name = run_name
        
        self.SEED = config.seed
        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.train_epochs
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay
        self.WARMUP_STEPS = config.warmup_steps
        self.N_LAYER = config.n_layer
        self.N_HEAD = config.n_head
        self.MAX_MOLECULE_LENGTH = config.max_molecule_length
        self.MAX_PROTEIN_LENGTH = config.max_protein_length
        self.MAX_LENGTH = config.max_length
        self.N_EMBD = config.n_embd
        
        
        # self.config = GPT2Config(add_cross_attention=True, is_decoder=True, n_layer=self.N_LAYER, n_head=self.N_HEAD, n_embd=self.N_EMBD)
