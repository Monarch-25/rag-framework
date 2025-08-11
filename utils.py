"""
Utility functions for managing session-based documents in a Redis vector store.
This module handles document hashing for de-duplication, adding documents with a TTL,
and explicit deletion.
"""
import redis
import hashlib
from redis.commands.search.query import Query
from langchain.docstore.document import Document
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from typing import List
import logging

_logger = logging.getLogger(__name__)

def document_exists(raw_client: redis.Redis, index_name: str, session_id: str, doc_hash: str) -> bool:
    """Checks if a document with the same hash already exists in the session."""
    try:
        query = Query(f'(@session_id:{{{session_id}}}) (@doc_hash:{{{doc_hash}}})')
        query = query.return_fields('id').paging(0, 1)
        result = raw_client.ft(index_name).search(query)
        return result.total > 0
    except Exception as e:
        _logger.error(f"Error checking if document exists for session '{session_id}': {e}")
        # Fail safely: assume it doesn't exist to allow ingestion to proceed.
        return False

def add_documents_for_session(
    raw_client: redis.Redis,
    vectorstore: RedisVectorStore,
    index_name: str,
    user_id: str,
    session_id: str,
    documents: List[Document],
    ttl_seconds: int = 14400
) -> List[str]:
    """
    Adds unique documents to the index, tagging them with user and session IDs,
    and setting a TTL. Skips documents that are already indexed in the session.
    """
    _logger.info(f"Processing {len(documents)} document chunks for user '{user_id}' in session '{session_id}'.")
    
    docs_to_add = []
    for doc in documents:
        content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
        if not document_exists(raw_client, index_name, session_id, content_hash):
            doc.metadata["user_id"] = user_id
            doc.metadata["session_id"] = session_id
            doc.metadata["doc_hash"] = content_hash
            docs_to_add.append(doc)
        else:
            _logger.info(f"Skipping duplicate document chunk with hash {content_hash[:8]}... for session '{session_id}'.")

    if not docs_to_add:
        _logger.info("No new unique documents to add.")
        return []

    _logger.info(f"Adding {len(docs_to_add)} new, unique document chunks.")
    keys_added = vectorstore.add_documents(docs_to_add)
    
    if keys_added:
        _logger.info(f"Setting TTL for {len(keys_added)} new keys.")
        try:
            pipeline = raw_client.pipeline()
            for key in keys_added:
                pipeline.expire(key, ttl_seconds)
            pipeline.execute()
        except Exception as e:
            _logger.error(f"Failed to set TTL for new keys in session '{session_id}': {e}")
    
    _logger.info(f"Successfully added and set TTL for {len(keys_added)} chunks.")
    return keys_added

def session_has_documents(raw_client: redis.Redis, index_name: str, session_id: str) -> bool:
    """Checks if any documents already exist for a given session."""
    try:
        query = Query(f'@session_id:{{{session_id}}}').return_fields('id').paging(0, 1)
        result = raw_client.ft(index_name).search(query)
        return result.total > 0
    except Exception as e:
        _logger.error(f"Error checking for documents in session '{session_id}': {e}")
        return False

def delete_documents_for_session(raw_client: redis.Redis, index_name: str, session_id: str):
    """Deletes all documents associated with a specific session_id. For explicit cleanup."""
    _logger.warning(f"Executing manual deletion for all documents in session: {session_id}")
    
    try:
        query = Query(f'@session_id:{{{session_id}}}').nocontent()
        results = raw_client.ft(index_name).search(query)
        if not results.docs:
            _logger.info(f"No documents found for session '{session_id}' to delete.")
            return

        keys_to_delete = [doc.id for doc in results.docs]
        _logger.info(f"Found {len(keys_to_delete)} document keys to delete for session '{session_id}'.")
        
        pipeline = raw_client.pipeline()
        for key in keys_to_delete:
            pipeline.delete(key)
        pipeline.execute()
        _logger.info(f"✅ Successfully deleted documents for session '{session_id}'.")
    except Exception as e:
        _logger.error(f"Error deleting documents for session '{session_id}': {e}")
