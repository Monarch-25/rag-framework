"""Utility functions for interacting with the Redis/MemoryDB backend,
independent of the core RAG logic.
"""
import redis
from redis.commands.search.query import Query

def delete_documents_for_user(raw_client: redis.Redis, index_name: str, user_id: str):
    """
    Deletes all documents associated with a specific user_id from the index. 
    
    Args:
        raw_client: The raw redis-py client instance.
        index_name: The name of the search index.
        user_id: The ID of the user whose documents should be deleted.
    """
    print(f"\n--- Deleting all documents for user: '{user_id}' ---")
    
    # Redis-py query to find all keys for the given user_id tag.
    # We don't need the content, so we use NOCONTENT for performance.
    query = Query(f"@user_id:{{{user_id}}}").nocontent()
    
    try:
        results = raw_client.ft(index_name).search(query)
        if not results.docs:
            print(f"No documents found for user '{user_id}'. Nothing to delete.")
            return 0

        keys_to_delete = [doc.id for doc in results.docs]
        print(f"Found {len(keys_to_delete)} document keys to delete.")

        # Delete the keys in a single command
        num_deleted = raw_client.delete(*keys_to_delete)
        print(f"✅ Successfully deleted {num_deleted} entries for user '{user_id}'.")
        return num_deleted
    
    except Exception as e:
        print(f"❌ Error during deletion for user '{user_id}': {e}")
        return 0
