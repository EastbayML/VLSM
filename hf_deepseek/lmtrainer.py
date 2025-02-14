from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gc
import os
import torch
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel
import wandb
from utils.utils import AsciiTokenizer, DynamicEvalCallback, preprocess_and_cache_dataset, save_prior_version, save_tokenizer_model
import numpy as np

# Define the evaluation function
def compute_metrics(eval_pred):
    print("Entered compute metrics")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    loss = np.mean((predictions - labels) ** 2)  # Example loss computation
    return {"eval_loss": loss}

def train(dataset, tokenizer, model,
            rundir="runs",
            logging_dir="./logs",
            num_epochs=1,
            per_device_batch_size=32,
            gradient_accumulation_steps=1,
            bf16=True,
            name="{repr(model)}_{config.dataset_name}",
            project='newmodel',
            eval_steps=500,
            max_length=512,
            run="{model_name}_{lef}",
            ):



    # Training arguments
    training_args = TrainingArguments(
        output_dir="temp",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=500,  # Adjust as needed
        weight_decay=0.01,  # Adjust as needed
        logging_dir=logging_dir,
        logging_steps=100,
        save_steps=1000,
        bf16=bf16,
        dataloader_num_workers=4,  # Adjust as needed
        # eval_strategy="steps",  # Set evaluation strategy to steps
        # eval_steps=eval_steps, 
        report_to=['wandb'],
        remove_unused_columns=False,
        metric_for_best_model = "eval_loss",
        # load_best_model_at_end=True,
        save_total_limit=2,
    )

    #print(f"{training_args}")
    config = training_args.to_dict()
    wrapped_model=model.module if hasattr(model, 'module') else model
    #print(f"{wrapped_model=}")
    config.update({
        #"max_length": wrapped_model.n_positions,
        "tokenizer_name": tokenizer.__class__.__name__,
        "model_name": wrapped_model.__class__.__name__,
        "dataset_name": dataset.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 256, # wrapped_model.config.n_embd,
        "num_layers": 16, # wrapped_model.config.n_layer,
        "num_heads": 8, # wrapped_model.config.n_head,        
    })

    # Initialize wandb
    try:
        lef="{num_layers}_{embed_dim}_{num_heads}".format(**config)
        name = run.format(**config,lef=lef)
    except Exception as e:
        name = f"{config.model_name}_{lef}"
    print(f"run name {name}")

    #wandb.init(project=project, config=config, name=name)

    # manage the output directory
    outdir=os.path.join(rundir,name)
    save_prior_version(outdir)

    # save config of tokenizer and model for reconstruction
    save_tokenizer_model(outdir, tokenizer, model)
    
    # This is a little tricky because we want the logs to be in a directory with the run name
    # but the run name is created using the training_args. So we create a new trainging_args
    # with the updated output_dir
    training_args = TrainingArguments(**{**training_args.to_dict(), "output_dir": outdir})

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # Access the 'train' split
        eval_dataset=dataset["validation"],  # Access the 'validation' split
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[DynamicEvalCallback()],
        #compute_metrics=compute_metrics,  
    )

    # Start training
    trainer.train()

    # Free up memory
    print("Clean up after training")
    wandb.finish()



def run_experiment(n_layer, n_embd, n_head,batchsize):
    print(f'run experiment {n_layer=},{n_embd=},{n_head=}')
    max_length=1024
    tokenizer=AsciiTokenizer()
    dataset = preprocess_and_cache_dataset("roneneldan/TinyStories",tokenizer)

    from model import ModelArgs,Transformer
    torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_device("cuda")
    #torch.manual_seed(0)

    args=ModelArgs()
    args.max_batch_size=batchsize
    args.vocab_size=tokenizer.vocab_size
    model = Transformer(args)


    # config = GPT2Config(vocab_size=tokenizer.vocab_size,
    #                     n_embd=n_embd,
    #                     n_layer=n_layer,
    #                     n_head=n_head,
    #                     n_positions=max_length
    #                     )
    # model = GPT2LMHeadModel(config)

    # Move the model to the primary GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Wrap the model with DataParallel
    model = torch.nn.DataParallel(model)
    print(f"cuda {next(model.parameters()).is_cuda}")


    # Perform a single forward pass
    print("forward pass test")
    # Sample a batch from the dataset
    x = next(iter(dataset['train']))['input_ids']
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    collated_batch = data_collator([{'input_ids': x}])
    with torch.no_grad():
        print(f"{collated_batch['input_ids'].shape=}")
        outputs = model(collated_batch['input_ids'].to(device))
        try:
            print(f"Forward pass output shape: {outputs.logits.size()}")
        except:
            print(f"Forward pass output shape: {outputs=}")     

    print(f" train {n_layer=},{n_embd=},{n_head=}")
    train(dataset,tokenizer,model,per_device_batch_size=batchsize,max_length=512)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Running experiments")
    tokenizer = AsciiTokenizer()
    dataset_name="roneneldan/TinyStories"
    dataset = preprocess_and_cache_dataset(dataset_name,tokenizer)

    experiments = [
        (2,128,2,128),
        (3,192,3,128),
        (4,256,4,128),
        (5,320,5,128),
        (6,384,6,128),
        (7,448,7,32),
        (8,512,8,32),
        (9,576,9,32),
        (10,640,10,32),
        (11,704,11,16),
        (12,768,12,16),
        (13,832,13,16),
        (14,896,14,8),
        (15,960,15,8),
        (16,1024,16,8),
    ]

    for i, (n_layer, n_embd, n_head,bs) in enumerate(experiments):
        run_experiment(n_layer, n_embd, n_head, bs)
    #     for future in futures:
    # # Run experiments in parallel using ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(run_experiment, n_layer, n_embd, n_head, i % 2) for i, (n_layer, n_embd, n_head) in enumerate(experiments)]
    #     for future in futures:
    #         future.result()