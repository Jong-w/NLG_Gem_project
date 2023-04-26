from transformers import AutoModelForSeq2SeqLM
from Data import MODEL_NAME

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)