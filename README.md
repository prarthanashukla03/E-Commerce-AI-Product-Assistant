# 🛒 E-Commerce AI Product Assistant

An AI-powered e-commerce product assistant that uses **Retrieval-Augmented Generation (RAG)** and a **Large Language Model (LLM)** to answer product-related questions using information from a product database.

The system uses **FAISS** for vector-based product retrieval, **Sentence Transformers** for generating embeddings, **Llama 3.2 3B** as the LLM, **Ollama** for running the LLM locally, and **Streamlit** for the web interface.

---

## 📌 Project Overview

Traditional e-commerce search systems mainly depend on keyword matching. They may return products but often cannot understand natural-language questions such as:

- "Suggest a laptop for programming."
- "Tell me about HP Pavilion 14."
- "Which products have 16 GB RAM?"
- "Suggest a laptop under ₹70,000."
- "Which laptop is suitable for students?"

This project addresses this problem by combining **RAG with an LLM**.

The system first retrieves relevant product information from a vector database and then provides that information as context to the LLM. The LLM uses the retrieved context to generate a clear and natural-language response.

---

## 🎯 Objectives

The main objectives of this project are:

1. To develop an AI-based product assistant for e-commerce applications.
2. To implement Retrieval-Augmented Generation (RAG).
3. To use FAISS for efficient similarity-based product retrieval.
4. To generate product embeddings using Sentence Transformers.
5. To integrate a Large Language Model for natural-language response generation.
6. To use Llama 3.2 3B as the local LLM.
7. To build an interactive user interface using Streamlit.
8. To reduce hallucination by grounding LLM responses in retrieved product information.
9. To provide users with a conversational way of searching product information.

---

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| Pandas | Dataset loading and processing |
| NumPy | Numerical operations |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Ollama | Local LLM execution |
| Llama 3.2 3B | Large Language Model |
| Streamlit | Web application interface |
| Pickle | Storing retrieved product documents |
| Git & GitHub | Version control and project hosting |

---

## 🏗️ System Architecture

The overall architecture of the project is:

```text
                         USER
                           │
                           ▼
                    User Question
                           │
                           ▼
              Sentence Transformer
                all-MiniLM-L6-v2
                           │
                           ▼
                    Query Embedding
                           │
                           ▼
                    FAISS Vector DB
                           │
                           ▼
                  Top Relevant Products
                           │
                           ▼
                Retrieved Product Context
                           │
                           ▼
              ┌─────────────────────────┐
              │          RAG            │
              │                         │
              │ Retrieval + Context     │
              └────────────┬────────────┘
                           │
                           ▼
                  Llama 3.2 3B
                       (LLM)
                           │
                           ▼
                  Generated Answer
                           │
                           ▼
                    Streamlit UI
                           │
                           ▼
                         USER
