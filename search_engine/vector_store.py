"""
ChromaDB wrapper.

• Uses a persistent client when chroma_persist_dir is set (recommended for production).
• Falls back to in-memory when the dir is blank ("").
• populate() seeds the collection with a meaningful set of agricultural traits
  so similarity_search returns useful results even without Kartik's Parquet data.
"""
import chromadb
from shared.config import settings
from search_engine.embeddings import embed_batch

# ── Client + Collection ───────────────────────────────────────────────────────

def _make_client() -> chromadb.ClientAPI:
    if settings.chroma_persist_dir:
        return chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return chromadb.EphemeralClient()


_client: chromadb.ClientAPI | None = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = _make_client()
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Seed data ────────────────────────────────────────────────────────────────
# Organised by domain so Jay can later extend this from real Parquet files.

SEED_TRAITS: list[dict] = [
    # Heat / drought tolerance
    {"id": "t001", "text": "heat shock protein HSP70 expression increases survival above 45°C",  "domain": "biology"},
    {"id": "t002", "text": "drought resistance via stomata closure and reduced transpiration",     "domain": "biology"},
    {"id": "t003", "text": "high canopy temperature tolerance in wheat cultivar HD3086",           "domain": "biology"},
    {"id": "t004", "text": "osmotic adjustment through proline accumulation under water deficit",  "domain": "biology"},
    {"id": "t005", "text": "deep root architecture reaching 120cm for subsoil moisture access",    "domain": "biology"},
    # Yield / efficiency
    {"id": "t006", "text": "high harvest index above 0.5 for improved grain yield",               "domain": "biology"},
    {"id": "t007", "text": "nitrogen use efficiency via optimised nitrate transporter expression", "domain": "biology"},
    {"id": "t008", "text": "photosynthetic efficiency under high radiation and heat stress",       "domain": "biology"},
    # Soil / environment
    {"id": "t009", "text": "salinity tolerance in sandy loam soils pH 7.5-8.5 Rajasthan",        "domain": "environment"},
    {"id": "t010", "text": "alkaline soil adaptation with pH buffering root exudates",             "domain": "environment"},
    {"id": "t011", "text": "low rainfall crop performance 200-400mm annual precipitation",        "domain": "environment"},
    {"id": "t012", "text": "arid zone adaptation for Jodhpur Barmer Bikaner regions India",       "domain": "environment"},
    # Disease / pest
    {"id": "t013", "text": "rust resistance genes Lr34 Yr36 for wheat leaf and yellow rust",      "domain": "biology"},
    {"id": "t014", "text": "aphid resistance via trichome density and antibiosis",                "domain": "biology"},
    # Structural / material traits
    {"id": "t015", "text": "waxy leaf cuticle coating reduces UV radiation and water loss",       "domain": "materials"},
    {"id": "t016", "text": "stem lodging resistance through increased culm wall thickness",       "domain": "materials"},
    {"id": "t017", "text": "thick seed coat protects embryo from thermal and osmotic stress",     "domain": "materials"},
    # Chemistry
    {"id": "t018", "text": "Arrhenius rate adaptation enzymes stable 40-55°C temperature range", "domain": "chemistry"},
    {"id": "t019", "text": "antioxidant enzyme SOD CAT activity elevated under oxidative stress", "domain": "chemistry"},
    {"id": "t020", "text": "phytohormone ABA signalling pathway regulates drought response",      "domain": "chemistry"},
    # Growth
    {"id": "t021", "text": "early vigour germination within 48h at 35-45°C soil temperature",    "domain": "biology"},
    {"id": "t022", "text": "short crop duration 90-100 days fits single monsoon season",          "domain": "biology"},
    {"id": "t023", "text": "tiller production maintained under moderate heat stress conditions",  "domain": "biology"},
    {"id": "t024", "text": "grain filling rate stable between 35°C and 42°C ambient temperature","domain": "biology"},
    {"id": "t025", "text": "high biomass accumulation in poor fertility low-input arid soils",    "domain": "environment"},
]


def populate(force: bool = False) -> None:
    """
    Seed the vector store.  Skips if data is already present (unless force=True).
    Call once at startup (main.py does this).
    """
    col = get_collection()
    if col.count() > 0 and not force:
        return  # already populated

    texts = [t["text"] for t in SEED_TRAITS]
    ids   = [t["id"]   for t in SEED_TRAITS]
    metas = [{"domain": t["domain"]} for t in SEED_TRAITS]

    vectors = [v.tolist() for v in embed_batch(texts)]

    col.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    print(f"[vector_store] Seeded {len(SEED_TRAITS)} traits into '{settings.chroma_collection}'.")
