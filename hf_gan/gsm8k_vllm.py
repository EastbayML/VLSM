from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from vllm import LLM, SamplingParams

# Load the tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the GSM8K dataset
dataset_name = "gsm8k"
dataset = load_dataset(dataset_name, "main")

# Initialize the LLM
llm = LLM(model_name)

# Define the sampling parameters
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a DataLoader
dataloader = DataLoader(dataset["test"], batch_size=8, collate_fn=data_collator)

# Function to run inference and calculate the GSM8K metric
def run_gsm8k_metric(dataloader, tokenizer, llm, sampling_params):
    predictions = []
    references = []

    for batch in dataloader:
        questions = batch["question"]
        reference_answers = batch["answer"]

        # Tokenize the questions
        inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)

        # Run inference
        outputs = llm.generate(inputs["input_ids"], sampling_params=sampling_params)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predictions.extend(decoded_outputs)
        references.extend(reference_answers)

    # Calculate the GSM8K metric
    correct = 0
    for pred, ref in zip(predictions, references):
        if pred.strip() == ref.strip():
            correct += 1

    accuracy = correct / len(references)
    return accuracy

# Run the GSM8K metric
accuracy = run_gsm8k_metric(dataloader, tokenizer, llm, sampling_params)
print(f"GSM8K Accuracy: {accuracy:.2f}")