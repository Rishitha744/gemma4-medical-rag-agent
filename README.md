# Medical RAG Agent with Gemma 🩺

A lightweight **Retrieval-Augmented Generation (RAG)** medical question-answering system built using Google's **Gemma**, Sentence Transformers, and ChromaDB.

The system retrieves relevant medical information from a document collection and provides the retrieved context to Gemma to generate concise, grounded answers.

## Overview

Large language models can generate convincing answers even when the underlying information is uncertain. This project demonstrates how **Retrieval-Augmented Generation (RAG)** can ground an LLM's response in a predefined medical knowledge base.

Instead of asking the language model to answer entirely from its internal knowledge, the system:

1. Converts the user's question into an embedding.
2. Searches a medical document collection.
3. Retrieves the most relevant passages.
4. Adds those passages to the LLM prompt.
5. Instructs Gemma to answer using only the retrieved context.
6. Returns a concise response with a source label.

---

## Architecture

```text
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
                 ChromaDB
                     │
              Similarity Search
                     │
                     ▼
          Top-2 Relevant Documents
                     │
                     ▼
              Prompt Builder
                     │
        Context + User Question
                     │
                     ▼
          Google Gemma 3 1B IT
                     │
                     ▼
          Grounded Medical Answer
```

---

## Technologies Used

* **Python**
* **Google Gemma 3**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **ChromaDB**
* **PyTorch**
* **Kaggle**
* **Hugging Face Hub**

### Models

**Generation Model**

```text
google/gemma-3-1b-it
```

**Embedding Model**

```text
all-MiniLM-L6-v2
```

---

## How the RAG Pipeline Works

### 1. Medical Knowledge Base

The current implementation contains a small collection of sample medical passages covering topics including:

* Type 2 diabetes
* Kidney disease
* Hypertension
* Heart attack
* Asthma
* Depression
* Anemia
* Migraine

These documents act as the knowledge base for the RAG system.

> This notebook is a prototype. The current knowledge base contains sample medical passages rather than a production medical dataset.

---

### 2. Generate Embeddings

The project uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

to convert the medical documents into dense vector representations.

```text
Medical Document
       ↓
Sentence Transformer
       ↓
Vector Embedding
```

---

### 3. Store Documents in ChromaDB

The medical documents and their embeddings are stored in a ChromaDB collection:

```python
collection = chroma_client.create_collection("medical_docs")
```

Each document receives a unique ID and corresponding embedding.

---

### 4. Embed the User Question

When a user asks a question, the same Sentence Transformer model converts the question into an embedding.

For example:

```text
"What are the symptoms of diabetes?"
```

becomes a query vector.

---

### 5. Retrieve Relevant Context

ChromaDB performs vector similarity search against the medical document collection.

The system retrieves the **top 2 most relevant documents**:

```python
results = collection.query(
    query_embeddings=question_embedding,
    n_results=2
)
```

This retrieved information becomes the context provided to Gemma.

---

### 6. Prompt Gemma

The retrieved passages are inserted into a prompt instructing Gemma to answer **only from the supplied context**.

Conceptually:

```text
You are a helpful medical assistant.

Answer the question using ONLY the context below.

Context:
[Retrieved Medical Document 1]
[Retrieved Medical Document 2]

Question:
[User Question]

Answer:
```

This helps reduce unsupported responses by grounding generation in retrieved information.

---

### 7. Generate the Answer

The project uses:

```text
google/gemma-3-1b-it
```

through Hugging Face Transformers.

Generation is deterministic in the current implementation:

```python
do_sample=False
```

with a maximum of:

```python
max_new_tokens=150
```

The generated response is cleaned to remove repeated lines and ends when the expected source line is encountered.

---

## Example Questions

The notebook tests the RAG agent with questions such as:

```text
What are the symptoms of diabetes?
```

```text
What are the signs of a heart attack?
```

```text
How is high blood pressure defined?
```

The system retrieves relevant medical passages before generating each answer.

---

## Installation

Install the required dependencies:

```bash
pip install transformers accelerate bitsandbytes sentence-transformers chromadb
```

---

## Hugging Face Authentication

The notebook loads Gemma from Hugging Face.

When running on Kaggle, configure your Hugging Face token as a Kaggle Secret:

```text
HF_token
```

The notebook retrieves it using:

```python
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_token")
```

**Never commit Hugging Face tokens or other credentials to GitHub.**

---

## Running the Project

The easiest way to run the project is through the included Jupyter/Kaggle notebook.

Run the cells in order to:

```text
1. Install dependencies
        ↓
2. Load Gemma
        ↓
3. Load the embedding model
        ↓
4. Create the medical knowledge base
        ↓
5. Generate document embeddings
        ↓
6. Store embeddings in ChromaDB
        ↓
7. Initialize the RAG agent
        ↓
8. Ask medical questions
```

Call the agent using:

```python
medical_rag_agent("What are the symptoms of diabetes?")
```

---

## Project Structure

```text
gemma-medical-rag-agent/
│
├── gemma4-medical-rag-agent.ipynb
├── README.md
└── .gitignore
```

---

## Current Limitations

This project is a prototype and currently:

* Uses a small manually defined medical document collection.
* Retrieves only the top 2 passages.
* Does not provide links to external medical sources.
* Does not include a dedicated evaluation pipeline.
* Does not include a production user interface.
* Is not intended for clinical diagnosis or medical decision-making.

---

## Future Improvements

Potential improvements include:

* Integrating PubMed or another authoritative medical corpus
* Processing larger collections of medical documents
* Adding document chunking and metadata filtering
* Adding reranking after vector retrieval
* Providing document-level citations
* Evaluating retrieval accuracy and answer faithfulness
* Building a Streamlit or Gradio interface
* Adding conversation history
* Comparing multiple embedding models
* Experimenting with larger Gemma models

---

## Disclaimer

This project is intended for **educational and research purposes only**.

The generated responses should not be considered medical advice, diagnosis, or treatment recommendations. Consult qualified healthcare professionals for medical concerns.

---

## License

This project is intended for educational and research use.
