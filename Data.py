from datasets import load_dataset
from transformers import AutoTokenizer

def construct_input_for_batch(batch):
    """Construct input strings from a batch."""
    source = [' '.join(concepts) for concepts in batch["concepts"]]
    target = batch["target"]
    return source, target

def batch_tokenize(batch, tokenizer, max_length=32):
    """Construct the batch (source, target) and run them through a tokenizer."""
    source, target = construct_input_for_batch(batch)
    res = {
        "input_ids": tokenizer(source)["input_ids"],
        "labels": tokenizer(
            target,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )["input_ids"],
    }
    return res

#Load CommonGen Dataset
DATASET_NAME = "common_gen"
data = load_dataset("gem", DATASET_NAME)

#Tokenize Dataset
MODEL_NAME = "facebook/bart-base"
MAX_LENGTH = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_data_tokenized = data['train'].map(
    lambda batch: batch_tokenize(batch, tokenizer, max_length=MAX_LENGTH),
    batched=True
)
valid_data_tokenized = data['validation'].map(
    lambda batch: batch_tokenize(batch, tokenizer, max_length=MAX_LENGTH),
    batched=True
)