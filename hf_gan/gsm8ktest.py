from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from utils.utils import batch_and_pad, tokenize_and_chunk_causal_lm,dataset_cache_names
from functools import partial
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding

# Load the tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the GSM8K dataset
dataset_name="gsm8k"
dataset = load_dataset(dataset_name, "main")
print(dataset)
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 
dataloader = DataLoader(dataset["test"], batch_size=8)
for i,batch in enumerate(dataset['test']):
    print(f"{type(batch)=} {batch.keys()}")
    print(f"{batch['question']}:{batch['answer']}")
    if i>10:
        exit()




#tokenized_datasets = dataset.map(tokenize_function, batched=True,batch_size=4,num_proc=4,cache_file_names={split:f"/home/rkstager/repos/VSLM/.cache/{dataset_name}_{split}.mapped" for split in dataset.keys()})
dataset = load_dataset("roneneldan/TinyStories")
# tokenized_datasets = dataset.map(partial(tokenize_and_chunk_causal_lm,tokenizer=tokenizer,max_length=512),
#                       batched=True,batch_size=4,cache_file_names=dataset_cache_names(dataset,dataset_name))#num_proc=4,

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Now use the collator
dataloader = DataLoader(dataset["train"], batch_size=8, collate_fn=data_collator)
for i,batch in enumerate(dataloader):
    print(f"{type(batch)=}")
    if i>10:
        exit()

tokenized_datasets = dataset.map(
    partial(tokenize_and_chunk_causal_lm, tokenizer=tokenizer, max_length=512),
    batched=True,
    batch_size=4, # Set your desired batch size here
    #remove_columns=dataset.column_names["test"],
    num_proc=None # Or set to a specific number
)

# tokenized_datasets = tokenized_datasets.map(
#     partial(batch_and_pad, tokenizer=tokenizer, max_length=512),
#     batched=True, # Batch at the chunk level
#     batch_size=4, # Set your desired batch size here
#     #remove_columns=tokenized_datasets.column_names["test"],
#     num_proc=4 # Or set to a specific number
# )





model = AutoModelForCausalLM.from_pretrained(model_name)

# Set training arguments for evaluation
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets['train'],
)

# Perform batched inference
predictions = []
for batch in trainer.get_eval_dataloader():
    inputs = batch["input_ids"].to(trainer.args.device)
    attention_mask = batch["attention_mask"].to(trainer.args.device)
    outputs = model.generate(inputs, attention_mask=attention_mask, max_length=256)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions.extend(decoded_outputs)

# Print the predictions
for i, prediction in enumerate(predictions):
    print(f"Question: {dataset[i]['question']}")
    print(f"Prediction: {prediction}")
    print(f"Answer: {dataset[i]['answer']}")
    print()