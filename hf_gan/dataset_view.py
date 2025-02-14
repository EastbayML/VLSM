from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from utils.utils import batch_and_pad, tokenize_and_chunk_causal_lm,dataset_cache_names
from functools import partial
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding,DataCollatorForLanguageModeling

# Load the tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the GSM8K dataset
dataset_name="roneneldan/TinyStories" # "gsm8k"
dataset = load_dataset(dataset_name)#, "main")
print(dataset)
print(dataset['train'])
dataset = dataset["train"].select(range(100))
print(f"after select {len(dataset)=}")
print(dataset)
for i,row in enumerate(dataset):
    print(row)
    if i>1:
        break

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True,num_proc=8)

for i,row in enumerate(tokenized_datasets):
    print(row)
    if i>1:
        break
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False) 
dataloader = DataLoader(tokenized_datasets, batch_size=4, collate_fn=data_collator)

print("after collatorforcausallm")

for i,batch in enumerate(dataloader):
    print(f"{type(batch)=} {batch.keys()}")
    #print(f"{batch['question']}\n\n{batch['answer']}\n\n\n")
    if i>1:
        exit()
