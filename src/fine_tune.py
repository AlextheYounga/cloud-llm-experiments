from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from .hyperparameters import *
import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

INPUT_FILE = 'data/kungfu.txt'
TEXT = open(INPUT_FILE, 'r', encoding='utf-8').read()
PRETRAINED_MODEL = 'models/TheBloke_Llama-2-7B-Chat-GGML'
TUNED_MODEL_SAVE_DIRECTORY = 'models/kungfuai'


class FineTune():
    def __init__(self, model_family=None):
        self.model_family = model_family

    def load_dataset(self, file_path, tokenizer):
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
        )
        return dataset

    def load_data_collator(self, tokenizer):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )
        return data_collator

    def train(self):
        if self.model_family == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL)
            model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL)
        else:
            # You may run into problems if your GPU has 16GB memory or less
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL)

        tokenizer.save_pretrained(TUNED_MODEL_SAVE_DIRECTORY)

        train_dataset = self.load_dataset(INPUT_FILE, tokenizer)

        data_collator = self.load_data_collator(tokenizer)

        model.save_pretrained(TUNED_MODEL_SAVE_DIRECTORY)

        training_args = TrainingArguments(
            output_dir=TUNED_MODEL_SAVE_DIRECTORY,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model()


# Train
tune = FineTune()

tune.train()
