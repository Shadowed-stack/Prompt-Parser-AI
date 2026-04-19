import chromadb
from shared.config import settings

client = chromadb.Client()
collection = client.get_or_create_collection(settings.chroma_collection)

def add_traits(traits):
    for i, t in enumerate(traits):
        collection.add(
            ids=[str(i)],
            documents=[t],
            embeddings=[embed(t)]
        )
