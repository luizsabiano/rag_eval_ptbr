import chromadb


def create_collection(collections_dir: str, collection_name: str, model_embedding_function):

    client = chromadb.PersistentClient(path=collections_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
        embedding_function=model_embedding_function
    )

    return collection
