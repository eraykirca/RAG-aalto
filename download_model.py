from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "google/flan-t5-xl"

print(f"Downloading model: {MODEL_NAME} ...")

# Download and cache model & tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Model and tokenizer downloaded successfully.")