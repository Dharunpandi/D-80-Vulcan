# test_retrieval.py

import chromadb
from sentence_transformers import SentenceTransformer

# Connect to DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("products")

model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔍 Try a natural query
query = "lightweight black running shoes under 2000"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
git
print("🔎 Top matching products:\n")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("✅", meta["title"])
    print("   💸 Price: ₹", meta["price"])
    print("   🎨 Color:", meta["color"])
    print("   📏 Size:", meta["size"])
    print("   🔗 URL:", meta["url"])
    print("   🖼️ Image:", meta["image_url"])
    print()
print("✅ Retrieval test completed!")