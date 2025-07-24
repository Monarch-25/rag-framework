#imports
import torch
import transformers
from transformers import AutoTokenizer  


#Takes a set of document embeddings and a query embedding, and returns the top k most similar documents using a reranker model.
def rerank_documents(doc_embeddings, query_embedding, k=5): 
    # Load the reranker model and tokenizer
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

    # Prepare the input for the reranker
    inputs = []
    for doc_embedding in doc_embeddings:
        inputs.append((query_embedding, doc_embedding))

    # Tokenize the inputs
    tokenized_inputs = tokenizer(
        [q for q, _ in inputs],
        [d for _, d in inputs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Get the reranker scores
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        scores = outputs.logits.squeeze().tolist()

    # Get the top k indices based on scores
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [doc_embeddings[i] for i in top_k_indices]