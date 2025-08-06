from transformers import T5Tokenizer, T5ForConditionalGeneration
from retriever import Retriever
import torch, textwrap

MODEL_NAME = "google/flan-t5-large" # CPU‑only
tokenizer  = T5Tokenizer.from_pretrained(MODEL_NAME)
model      = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to("cpu")
retriever  = Retriever()

# helper: keep context ≤ 900 tokens
def safe_join(chunks, max_tokens=900):
    joined = ""
    for c in chunks:
        if len(tokenizer.encode(joined + c)) > max_tokens:
            break
        joined += c + "\n"
    return joined

# main QA function
def answer(question):
    chunks = retriever.get_chunks(question)
    context = safe_join(chunks)      # token‑budgeted
    prompt  = textwrap.dedent(f"""
        You are a helpful assistant who does not hallucinate. 
        Using only the information in Context, write a clear answer to the Question with maximum 3 sentences.
        If Context is empty or very irrelevant to the Question, ONLY reply with "I don't know."
        
        Try to get the meaning behind the questions like:
        Q: what new plans are in store for the company?
        Reasoning: The user is not referring to "plans in store" literally but rather "ideas for the future"
        
        Context:
        {context}

        Question: {question}
        Answer:
    """)

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(
        inp.input_ids,
        max_length   = 120,
        num_beams=4,
        min_length   = 40,        # ensure substance
        do_sample    = False,      # sampling instead of beam
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    reply = text.split("Answer:")[-1].strip()

    if reply.lower().startswith("i don't know"):
        return "I don't know."
    return reply

# CLI loop
if __name__ == "__main__":
    print("Type a question (exit to quit)")
    while True:
        q = input("Q: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("A:", answer(q))
