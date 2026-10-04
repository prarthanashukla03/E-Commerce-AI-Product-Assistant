import pandas as pd
import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/products.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. CREATE PRODUCT DOCUMENTS
# ==========================================

documents = []

for _, row in df.iterrows():

    document = f"""
Product ID: {row['Product_ID']}
Product Name: {row['Product_Name']}
Category: {row['Category']}
Brand: {row['Brand']}
Price: ₹{row['Price']}
Rating: {row['Rating']}
Processor: {row['Processor']}
RAM: {row['RAM']}
Storage: {row['Storage']}
Display: {row['Display']}
GPU: {row['GPU']}

Description:
{row['Description']}

Features:
{row['Features']}
"""

    documents.append(document)


print("Total product documents:", len(documents))


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==========================================
# 4. CREATE EMBEDDINGS
# ==========================================

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

print("Embeddings created!")
print("Embedding shape:", embeddings.shape)


# ==========================================
# 5. CREATE FAISS VECTOR INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    embeddings.astype("float32")
)

print("FAISS index created!")
print("Total vectors:", index.ntotal)


# ==========================================
# 6. CREATE VECTOR DATABASE FOLDER
# ==========================================

os.makedirs("vector_db", exist_ok=True)


# ==========================================
# 7. SAVE FAISS INDEX
# ==========================================

faiss.write_index(
    index,
    "vector_db/products.index"
)


# ==========================================
# 8. SAVE PRODUCT DOCUMENTS
# ==========================================

with open(
    "vector_db/products.pkl",
    "wb"
) as f:

    pickle.dump(documents, f)


# ==========================================
# COMPLETE
# ==========================================

print()
print("==========================================")
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("==========================================")
print("FAISS index: vector_db/products.index")
print("Documents:   vector_db/products.pkl")
print("Total products:", len(documents))