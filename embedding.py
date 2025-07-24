#takes a huggingface embedding model and creates an embedding for each text chunk
from transformers import AutoTokenizer, AutoModel
import torch    
def create_embedding(text, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the mean of the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings
