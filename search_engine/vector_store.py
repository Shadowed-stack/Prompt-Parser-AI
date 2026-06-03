"""
ChromaDB wrapper with a richer, multi-crop seed dataset.

Fix over v1:
  • Added sugarcane, rice, maize, bajra, mustard, cotton trait entries
  • Tagged every trait with crop + domain metadata for filtered search
  • populate() is idempotent (upsert, not add)
"""
import chromadb
from shared.config import settings
from search_engine.embeddings import embed_batch


def _make_client() -> chromadb.ClientAPI:
    if settings.chroma_persist_dir:
        return chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return chromadb.EphemeralClient()


_client = None
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


# ── Seed traits (multi-crop) ──────────────────────────────────────────────────

SEED_TRAITS: list[dict] = [

    # ── WHEAT ──────────────────────────────────────────────────────────────
    {"id": "w001", "crop": "wheat", "domain": "biology",
     "text": "heat shock protein HSP70 expression increases wheat survival above 45°C"},
    {"id": "w002", "crop": "wheat", "domain": "biology",
     "text": "drought resistance in wheat via stomata closure and reduced transpiration"},
    {"id": "w003", "crop": "wheat", "domain": "biology",
     "text": "high canopy temperature tolerance in wheat cultivar HD3086"},
    {"id": "w004", "crop": "wheat", "domain": "biology",
     "text": "grain filling rate in wheat stable between 35°C and 42°C ambient temperature"},
    {"id": "w005", "crop": "wheat", "domain": "biology",
     "text": "tiller production in wheat maintained under moderate heat stress conditions"},
    {"id": "w006", "crop": "wheat", "domain": "biology",
     "text": "rust resistance genes Lr34 Yr36 for wheat leaf and yellow rust"},
    {"id": "w007", "crop": "wheat", "domain": "biology",
     "text": "deep root architecture in wheat reaching 120cm for subsoil moisture access"},
    {"id": "w008", "crop": "wheat", "domain": "biology",
     "text": "osmotic adjustment in wheat through proline accumulation under water deficit"},

    # ── SUGARCANE ──────────────────────────────────────────────────────────
    {"id": "s001", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane heat tolerance variety Co0238 performs well at 35-42°C in Uttar Pradesh"},
    {"id": "s002", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane sucrose accumulation and juice quality maintained under moderate heat stress 37-40°C"},
    {"id": "s003", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane ratoon crop regeneration and stool vigour under high temperature conditions"},
    {"id": "s004", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane waterlogging tolerance important for Uttar Pradesh monsoon flooding"},
    {"id": "s005", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane drought tolerance through deep fibrous root system and osmotic regulation"},
    {"id": "s006", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane red rot resistance Colletotrichum falcatum fungal pathogen management"},
    {"id": "s007", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane high CCS sugar content varieties for UP subtropical growing conditions"},
    {"id": "s008", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane early maturing variety 10-month crop cycle for North Indian plains"},
    {"id": "s009", "crop": "sugarcane", "domain": "environment",
     "text": "sugarcane alluvial soil preference Gangetic plains Uttar Pradesh high fertility loamy"},
    {"id": "s010", "crop": "sugarcane", "domain": "environment",
     "text": "sugarcane cultivation Uttar Pradesh Meerut Muzaffarnagar belt subtropical climate"},
    {"id": "s011", "crop": "sugarcane", "domain": "chemistry",
     "text": "sugarcane photosynthesis C4 pathway efficient at high temperatures 30-40°C and high radiation"},
    {"id": "s012", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane smut resistance Sporisorium scitamineum whip emergence control"},
    {"id": "s013", "crop": "sugarcane", "domain": "biology",
     "text": "sugarcane high tillering ability after planting improves yield in North India"},
    {"id": "s014", "crop": "sugarcane", "domain": "materials",
     "text": "sugarcane bagasse and trash mulching conserves soil moisture reduces evaporation"},

    # ── RICE ───────────────────────────────────────────────────────────────
    {"id": "r001", "crop": "rice", "domain": "biology",
     "text": "rice submergence tolerance Sub1A gene flood survival 2 weeks waterlogged"},
    {"id": "r002", "crop": "rice", "domain": "biology",
     "text": "rice heat tolerance during flowering pollen viability above 38°C"},
    {"id": "r003", "crop": "rice", "domain": "biology",
     "text": "rice salinity tolerance Saltol QTL for coastal and saline soils"},
    {"id": "r004", "crop": "rice", "domain": "environment",
     "text": "rice paddy cultivation kharif season high rainfall alluvial soils India"},

    # ── MAIZE / CORN ───────────────────────────────────────────────────────
    {"id": "m001", "crop": "maize", "domain": "biology",
     "text": "maize drought tolerance stay-green trait delayed senescence under water stress"},
    {"id": "m002", "crop": "maize", "domain": "biology",
     "text": "maize heat stress tolerance pollen germination maintained above 38°C"},
    {"id": "m003", "crop": "maize", "domain": "chemistry",
     "text": "maize C4 photosynthesis high efficiency at elevated temperatures and CO2"},

    # ── BAJRA / PEARL MILLET ───────────────────────────────────────────────
    {"id": "b001", "crop": "bajra", "domain": "biology",
     "text": "bajra pearl millet extreme heat tolerance above 42°C arid Rajasthan Gujarat"},
    {"id": "b002", "crop": "bajra", "domain": "environment",
     "text": "bajra sandy soil low fertility drought adaptation Rajasthan desert conditions"},

    # ── MUSTARD ────────────────────────────────────────────────────────────
    {"id": "mu01", "crop": "mustard", "domain": "biology",
     "text": "mustard heat tolerance during pod filling Rajasthan Haryana rabi season"},
    {"id": "mu02", "crop": "mustard", "domain": "biology",
     "text": "mustard aphid resistance high glucosinolate content natural pest deterrent"},

    # ── COTTON ────────────────────────────────────────────────────────────
    {"id": "c001", "crop": "cotton", "domain": "biology",
     "text": "cotton heat tolerance boll development above 40°C Maharashtra Gujarat Telangana"},
    {"id": "c002", "crop": "cotton", "domain": "biology",
     "text": "Bt cotton bollworm resistance Cry1Ac gene expression pest management"},

    # ── SHARED PHYSICS / CHEMISTRY / ENVIRONMENT ──────────────────────────
    {"id": "p001", "crop": "general", "domain": "chemistry",
     "text": "Arrhenius rate adaptation enzymes stable 40-55°C temperature range crop metabolism"},
    {"id": "p002", "crop": "general", "domain": "biology",
     "text": "antioxidant enzyme SOD CAT activity elevated under oxidative heat stress all crops"},
    {"id": "p003", "crop": "general", "domain": "chemistry",
     "text": "phytohormone ABA abscisic acid signalling pathway regulates drought response"},
    {"id": "p004", "crop": "general", "domain": "biology",
     "text": "early vigour germination within 48h at 35-45°C soil temperature for crop establishment"},
    {"id": "p005", "crop": "general", "domain": "materials",
     "text": "waxy leaf cuticle coating reduces UV radiation and water loss in heat stress"},
    {"id": "e001", "crop": "general", "domain": "environment",
     "text": "arid zone adaptation Jodhpur Barmer Bikaner Rajasthan low rainfall desert crops"},
    {"id": "e002", "crop": "general", "domain": "environment",
     "text": "alluvial fertile soil Gangetic plains Uttar Pradesh Bihar high productivity agriculture"},
    {"id": "e003", "crop": "general", "domain": "environment",
     "text": "salinity tolerance sandy loam soils pH 7.5-8.5 Rajasthan alkaline conditions"},
    {"id": "e004", "crop": "general", "domain": "environment",
     "text": "high humidity subtropical climate 70-80% monsoon season North Indian plains"},
    {"id": "e005", "crop": "general", "domain": "environment",
     "text": "photosynthetic efficiency under high solar radiation and heat stress tropical crops"},
    {"id": "e006", "crop": "general", "domain": "biology",
     "text": "thick seed coat protects embryo from thermal and osmotic stress germination"},
    {"id": "e007", "crop": "general", "domain": "biology",
     "text": "nitrogen use efficiency optimised nitrate transporter expression high yield potential"},
]


def populate(force: bool = False) -> None:
    """Seed the vector store.  Skips if already populated (unless force=True)."""
    col = get_collection()
    if col.count() > 0 and not force:
        return

    texts   = [t["text"]   for t in SEED_TRAITS]
    ids     = [t["id"]     for t in SEED_TRAITS]
    metas   = [{"domain": t["domain"], "crop": t["crop"]} for t in SEED_TRAITS]
    vectors = [v.tolist() for v in embed_batch(texts)]

    col.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    print(f"[vector_store] Seeded {len(SEED_TRAITS)} traits "
          f"({len({t['crop'] for t in SEED_TRAITS})} crops).")
