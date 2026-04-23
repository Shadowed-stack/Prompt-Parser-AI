import chromadb
from shared.config import settings
from search_engine.embeddings import embed

client = chromadb.Client()
collection = client.get_or_create_collection(settings.chroma_collection)

def populate():
    traits = [
        "heat tolerance",
        "drought resistance",
        "deep root system",
        "high yield efficiency",
        "waxy leaf coating"
    ]

    for i, t in enumerate(traits):
        collection.add(
            ids=[str(i)],
            documents=[t],
            embeddings=[embed(t).tolist()]
        )
