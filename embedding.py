#takes an embedding model and return the embedding of a document and a query
def get_embedding(document, query): 
    # This function would typically use an embedding model to convert the document and query into embeddings.
    # For now, we will return dummy embeddings.
    document_embedding = [0.1, 0.2, 0.3]  # Dummy embedding for the document
    query_embedding = [0.4, 0.5, 0.6]      # Dummy embedding for the query
    return document_embedding, query_embedding