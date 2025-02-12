from transformers import (
    PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, 
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
import wandb 
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from dataclasses import dataclass, field
import torch
import os


import warnings
warnings.simplefilter("ignore", category=FutureWarning)


@dataclass
class BertEmbedding:
    selfies_tokenizer_path: str
    train_path: str
    dataset_path: str = None
    tokenizer: PreTrainedTokenizerFast = field(init=False)
    dataset: Dataset = field(init=False)
    model: BertForMaskedLM = field(init=False)

    def __post_init__(self):
        self.tokenizer = self._load_tokenizer(self.selfies_tokenizer_path)
        if self.dataset_path is None:
            self.dataset = self._dataset()
        else:
            self.dataset = Dataset.load_from_disk(self.dataset_path)
        
        self.model = BertForMaskedLM(self._bertmodel())

    def training(self, 
                 output_dir: str = "./bert-bpe-selfies",
                 gradient_accumulation_steps: int = 4,
                 per_device_train_batch_size: int = 32,
                 num_train_epochs: int = 10,
                 save_strategy: str = "steps",
                 eval_steps: int = 10000):
        
        wandb.init(
            project="BERT-Selfies-Embedding",   # 這裡改成你的專案名稱
            name="bert_embedding_test0",  # 這是這次訓練的名稱
            config={
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "per_device_train_batch_size": per_device_train_batch_size,
                "num_train_epochs": num_train_epochs,
                "save_strategy": save_strategy,
                "eval_steps": eval_steps,
            }
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            save_strategy=save_strategy,
            save_steps=eval_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            logging_dir="./logs",
            logging_steps=5000,
            fp16=True,
            report_to="wandb",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.dataset
        )
        print("Start Training")
        trainer.train()
        wandb.finish()

    def testing(self, 
                test_selfies: str):
        input_ids = self.tokenizer(test_selfies, 
                                                return_tensors="pt",
                                                max_length=64,
                                                padding="max_length", 
                                                truncation=True)["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits 
        return logits

    
    def evaluate(self):
        print("Evaluating Model...")
        results = self.model.evaluate(self.dataset)
        return results

    def embed(self, 
              selfies: str):
        input_ids = self.tokenizer(selfies, 
                                return_tensors="pt", 
                                max_length=64,
                                padding="max_length", 
                                truncation=True)["input_ids"].to(self.model.device)

        with torch.no_grad():
            embeddings = self.model.bert(input_ids).last_hidden_state.mean(dim=1)

        return embeddings

    def _load_tokenizer(self, 
                        max_length=64):
        return PreTrainedTokenizerFast.from_pretrained(self.selfies_tokenizer_path, 
                                    model_max_length=max_length, 
                                    padding=True, 
                                    truncation=True)



    def _dataset(self):
        
        dataset_path = "dataset_ape_selfies"
        if os.path.exists(dataset_path):
            return Dataset.load_from_disk(dataset_path)
        else:
            
            dataset = load_dataset("text", data_files=self.train_path, split="train")

            def tokenize_function(examples):
                return self.tokenizer(examples["text"],
                                    max_length=64, 
                                    padding="max_length", 
                                    truncation=True)
            
            dataset = dataset.map(tokenize_function, batched=True, num_proc=8)

            dataset = dataset.remove_columns(["text"])
            dataset.save_to_disk(dataset_path)
            return dataset


    def _bertmodel(self, 
                   hidden_size: int = 256, 
                   num_hidden_layers: int = 8, 
                   num_attention_heads: int = 8, 
                   intermediate_size: int = 512):
        config = BertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
        return config


def main():
    selfies_tokenizer_path = "/data/tzeshinchen/RLMOE_kinase_inhibitor/src/data_preprocessing/kinase_ape_tokenizer"
    train_path = "/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset/extracted_file.txt"

    bert_emb = BertEmbedding(selfies_tokenizer_path, train_path)
    bert_emb.training()

    test_selfies = "[C][=C][C][=C][Ring1][=Branch1]"
    print("Testing : ", bert_emb.testing(test_selfies))
    embedding = bert_emb.embed(test_selfies)
    print("SELFIES Embedding:", embedding.shape)


if __name__ == '__main__':
    main()
