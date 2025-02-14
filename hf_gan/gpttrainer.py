import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import argparse
from utils.utils import onepass, chunk_text_datset

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train Qwen model on TinyStories dataset")
parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained weights')
args = parser.parse_args()

# Load the tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if args.load_pretrained:
    # Load the model with pre-trained weights
    model = AutoModelForCausalLM.from_pretrained(model_name)
else:
    # Create a configuration for the model and initialize from scratch
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

# Load the dataset
dataset = chunk_text_dataset(load_dataset("roneneldan/TinyStories",tokenizer), max_length=512)

# Calculate and print the size of the model
model_size = sum(p.numel() for p in model.parameters())
print(f"Model size: {model_size} parameters")

# sanity test
#onepass(model,tokenizer,dataset)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,  # Enable bf16
    logging_dir="./logs",
    lr_scheduler_type="cosine",  # Use cosine learning rate schedule
    warmup_steps=500,  # Number of warmup steps
)

# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to True if using masked language modeling
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

# Train the model
trainer.train()