import torch
from transformers import  Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
from Data import MAX_LENGTH, train_data_tokenized, valid_data_tokenized, tokenizer
from Model import model

rouge_scorer = load_metric("rouge")
def rouge_metric_builder(tokenizer):
    def compute_rouge_metrics(pred):
        """Utility to compute ROUGE during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        # All special tokens are removed.
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        # Compute the metric.
        rouge_results = rouge_scorer.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge2", "rougeL"],
            #use_agregator=True,
            use_stemmer=False,
        )
        return {
            "rouge2": round(rouge_results['rouge2'].mid.fmeasure, 4),
            "rougeL": round(rouge_results['rougeL'].mid.fmeasure, 4),
        }
    return compute_rouge_metrics

rouge_metric_fn = rouge_metric_builder(tokenizer)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
BEAM_SIZE = 4

model = model.to(DEVICE)

train_args = Seq2SeqTrainingArguments(
    output_dir="BART-commongen",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    # optimization args, the trainer uses the Adam optimizer
    # and has a linear warmup for the learning rate
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=1e-04,
    num_train_epochs=3,
    warmup_steps=1000,
    # misc args
    seed=RANDOM_SEED,
    disable_tqdm=False,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
    # generation
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_data_tokenized,
    eval_dataset=valid_data_tokenized,
    tokenizer=tokenizer,
    compute_metrics=rouge_metric_fn,
)

trainer._max_length = MAX_LENGTH
trainer._num_beams = BEAM_SIZE