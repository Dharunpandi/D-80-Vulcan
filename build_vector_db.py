# build_vector_db.py

import json
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------
# 1. Load product.json
# -----------------------
with open("products.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.dropna(subset=["title", "price"])

# Clean price (remove commas, ₹, etc.)
df["price"] = df["price"].replace({',': '', '₹': ''}, regex=True)

# -----------------------
# 2. Combine fields for embedding
# -----------------------
df["combined_text"] = (
    "Product: " + df["title"] +
    " | Price: ₹" + df["price"].astype(str) +
    " | Color: " + df.get("color", "") +
    " | Size: " + df.get("size", "")
)

# -----------------------
# 3. Store in ChromaDB
# -----------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("products")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Add products
for idx, row in df.iterrows():
    embedding = model.encode(row["combined_text"]).tolist()
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        documents=[row["combined_text"]],
        metadatas=[{
            "title": row["title"],
            "price": row["price"],
            "color": row.get("color", ""),
            "size": row.get("size", ""),
            "url": row["url"],
            "image_url": row.get("image_url", "")
        }]
    )

print("✅ All products stored successfully in ChromaDB!")
