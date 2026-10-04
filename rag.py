import faiss
import pickle
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. LOAD FAISS VECTOR DATABASE
# ==========================================

print("Loading FAISS database...")

index = faiss.read_index(
    "vector_db/products.index"
)


# ==========================================
# 2. LOAD PRODUCT DOCUMENTS
# ==========================================

with open(
    "vector_db/products.pkl",
    "rb"
) as f:

    documents = pickle.load(f)


print("Product documents loaded:", len(documents))


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# 4. RAG FUNCTION
# ==========================================

def generate_answer(query):

    print("\nUser Query:", query)

    # --------------------------------------
    # Convert user query into embedding
    # --------------------------------------

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # --------------------------------------
    # Search FAISS
    # --------------------------------------

    distances, indices = index.search(
        query_embedding,
        5
    )


    # --------------------------------------
    # Get relevant products
    # --------------------------------------

    retrieved_products = []

    for i in indices[0]:

        if i < len(documents):

            retrieved_products.append(
                documents[i]
            )


    print(
        "Retrieved products:",
        len(retrieved_products)
    )


    # --------------------------------------
    # Create context
    # --------------------------------------

    context = "\n\n".join(
        retrieved_products
    )


    # ======================================
    # 5. CREATE RAG PROMPT
    # ======================================

    prompt = f"""
You are an AI E-Commerce Product Assistant.

You must answer the user's question using ONLY
the product information provided in the context.

Do not invent product names, prices,
ratings, specifications or features.

If the requested information is not present
in the context, say:

"I could not find that information
in the available product database."

Use simple and clear language.

----------------------------------------
PRODUCT INFORMATION
----------------------------------------

{context}

----------------------------------------
USER QUESTION
----------------------------------------

{query}

----------------------------------------
ANSWER
----------------------------------------
"""


    # ======================================
    # 6. SEND CONTEXT TO LLAMA
    # ======================================

    response = ollama.chat(

        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # ======================================
    # 7. GET ANSWER
    # ======================================

    answer = response["message"]["content"]


    return answer, retrieved_products