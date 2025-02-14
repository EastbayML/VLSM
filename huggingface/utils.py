import importlib
import json
import os
import string
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
import shutil
from transformers import TrainerCallback, TrainerState, TrainerControl

class AsciiTokenizer(PreTrainedTokenizerFast):
    """
    A custom tokenizer class that extends PreTrainedTokenizerFast for tokenizing ASCII characters.

    This tokenizer uses a Unigram model and includes special tokens for various purposes such as
    unknown tokens, padding, beginning of sequence, end of sequence, and more. It normalizes text
    using NFKC normalization and pre-tokenizes text at the byte level.

    Args:
        **kwargs: Additional keyword arguments passed to the PreTrainedTokenizerFast initializer.

    Attributes:
        tokenizer (Tokenizer): The underlying Tokenizer object used for tokenization.
    """
    def __init__(self, **kwargs):
        special_tokens = {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "bot_token": "<thinking>",
            "eot_token": "</thinking>",
            "cls_token":"<cls>",
            "sep_token":"<sep>",
            "mask_token":"<mask>",
        }

        # Create the tokenizer
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Define the vocabulary, including all printable characters
        vocab = list(special_tokens.values()) + list(string.printable)
        tokenizer.add_special_tokens(list(special_tokens.values()))
        tokenizer.add_tokens(vocab[len(special_tokens):])  # Add regular tokens


        print(f"{tokenizer.get_vocab_size()=}")
        # Initialize the PreTrainedTokenizerFast
        super().__init__(tokenizer_object=tokenizer, padding_side="right", **special_tokens, **kwargs)
        self.tokenizer = tokenizer


    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def _tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_token(index)
    
def preprocess_and_cache_dataset(dataset_name, tokenizer,reload=False,max_length=512):
    """
    Preprocesses and caches a dataset.

    This function loads a dataset, tokenizes it using the provided tokenizer, and caches the
    processed dataset to disk. If the cached dataset already exists and `reload` is False,
    it loads the dataset from the cache. Otherwise, it processes the dataset and saves it
    to the cache.

    Args:
        dataset_name (str): The name of the dataset to be processed.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing the dataset.
        reload (bool, optional): If True, forces reprocessing of the dataset even if a cached
                                 version exists. Defaults to False.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 512.

    Returns:
        Dataset: The processed and tokenized dataset.

    Raises:
        FileNotFoundError: If the cached dataset file is not found and `reload` is False.
        OSError: If there is an error creating the cache directory.
    """

    cached_fname = os.path.expanduser(f"~/.cache/datasets/{dataset_name}")
    dataset = None
    if not reload:
        try:
            dataset = load_from_disk(cached_fname)
        except FileNotFoundError:
            pass

    if dataset is None:
        print(f"Generating {cached_fname} now.")

        os.makedirs(os.path.dirname(cached_fname), exist_ok=True)
        dataset = load_dataset(dataset_name)

        def tokenize_function(examples):
            tokenized_output = tokenizer(
                examples["text"], 
                padding=True, 
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            print(f"{len(tokenized_output['input_ids'][0])=} {len(examples['text'])=}")
            return {
                "input_ids": tokenized_output["input_ids"],
                "attention_mask": tokenized_output["attention_mask"]
            }

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=os.cpu_count() - 1,  # Adjust as needed
        )
        print("Save dataset to disk")
        dataset.save_to_disk(cached_fname)
    return dataset



def save_prior_version(file_path):
    """
    Save the prior version of a file by renaming it with an incremented version number.

    If the specified file exists, it will be renamed by appending an incremented version number
    to its base name. For example, if the file is named 'example.txt', it will be renamed to
    'example_1.txt', 'example_2.txt', etc., depending on the existing versions.

    Args:
        file_path (str): The path to the file to be versioned.

    Returns:
        str: The new file path with the version number appended, or None if the file does not exist.

    Raises:
        OSError: If the file cannot be renamed.
    """
    if not os.path.exists(file_path):
        return
       
    base, ext = os.path.splitext(file_path)
    version = 1
    new_file_path = f"{base}_{version}{ext}"
    
    while os.path.exists(new_file_path):
        version += 1
        new_file_path = f"{base}_{version}{ext}"
    
    shutil.move(file_path, new_file_path)
    return new_file_path

def save_tokenizer_model(outdir,tokenizer,model):
    """
    Save the tokenizer and model class information to the specified directory.

    Args:
        outdir (str): The directory where the model and tokenizer information will be saved.
        tokenizer (PreTrainedTokenizer): The tokenizer to be saved.
        model (PreTrainedModel): The model to be saved.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created.
    """
    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Save the model class information
    model_info = {
        "tokenizer_class_name": tokenizer.__class__.__name__,
        "tokenizer_module": tokenizer.__class__.__module__,
        "model_class_name": model.__class__.__name__,
        "model_module": model.__class__.__module__,
    }

    with open(os.path.join(outdir, "model_info.json"), "w") as f:
        json.dump(model_info, f)

def load_tokenizer_model_class(outdir):
    """
    Load the tokenizer and model classes from the specified directory.

    Args:
        outdir (str): The directory containing the model and tokenizer information.

    Returns:
        tuple: A tuple containing the tokenizer class and model class.

    Raises:
        FileNotFoundError: If the model_info.json file is not found in the specified directory.
        ImportError: If the module or class specified in model_info.json cannot be imported.
    """
    # Load the model class information
    with open(os.path.join(dir, "model_info.json"), "r") as f:
        model_info = json.load(f)

    # Dynamically import the model class
    module = importlib.import_module(model_info["model_module"])
    model_class = getattr(module, model_info["model_class_name"])

    # Dynamically import the model class
    module = importlib.import_module(model_info["tokenizer_module"])
    tokenizer_class = getattr(module, model_info["tokenizer_class_name"])
 
    return tokenizer_class,model_class

def load_tokenizer_model(outdir, checkpoint=None):
    """
    Load the tokenizer and model from the specified directory.

    Args:
        outdir (str): The directory containing the model and tokenizer.
        checkpoint (str, optional): The specific checkpoint to load. If None, the most recent checkpoint is used.

    Returns:
        tuple: A tuple containing the loaded tokenizer and model.

    Raises:
        FileNotFoundError: If no checkpoints are found in the specified directory.
    """
    # Load the model class information
    if checkpoint is None:
        checkpoints = [os.path.join(outdir, d) for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))]
        checkpoints.sort(key=os.path.getmtime, reverse=False)  # Sort by modification time, most recent first
        checkpoint = checkpoints[-1]  # Only use the most recent checkpoint
    else:
        checkpoint = os.path.join(outdir,checkpoint)

    tokenizer_class, model_class = load_tokenizer_model_class(outdir)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    model = model_class.from_pretrained(checkpoint)
    return tokenizer, model

class DynamicEvalCallback(TrainerCallback):
    """
    A custom callback to adjust the evaluation steps dynamically during training.

    Args:
        ratio (float): The ratio by which to multiply the eval_steps at each evaluation.
        first_epoch_only (bool): If True, only adjust eval_steps during the first epoch and switch to epoch-based evaluation afterwards.

    """
    def __init__(self, ratio=1.25, first_epoch_only=True):
        self.ratio = ratio
        self.first_epoch_only = first_epoch_only

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        args.eval_steps = int(args.eval_steps*self.ratio)
        if self.first_epoch_only and state.epoch > 0 and args.evaluation_strategy != "epoch":
            args.evaluation_strategy = "epoch"
            args.eval_steps = 0
        return control
