## 🚀 Features

### 📄 PDF Document Processing

* Upload PDF documents through the Streamlit interface.
* Extract text using PyMuPDF.
* Split documents into overlapping chunks for efficient retrieval.

### 🔍 Semantic Search

* Generate embeddings using `all-MiniLM-L6-v2`.
* Store embeddings in a FAISS vector index.
* Retrieve semantically relevant document chunks.

### 🔁 Multi-Query Retrieval

* Reformulate user questions into multiple semantically similar queries using DeepSeek.
* Retrieve context for each reformulated query.
* Improve retrieval recall and reduce missed information.

### 🎯 Cross-Encoder Re-ranking

* Re-rank retrieved chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* Select the most relevant contexts before answer generation.

### 🛡️ Hallucination Reduction

* Includes an answerability check based on retrieval similarity scores.
* Detects when relevant information is not available in the document.

### 🤖 Grounded Answer Generation

* Uses Ollama with DeepSeek-R1 1.5B.
* Generates answers strictly from retrieved document context.
* Returns:

```text
I do not know based on the document.
```

when the answer cannot be found.

### 📚 Source Transparency

* Displays retrieved source chunks used for answer generation.
* Allows users to inspect supporting evidence.

### 🌐 Interactive Web Interface

* Built with Streamlit.
* Simple PDF upload and question-answer workflow.

---

## 🏗️ Architecture

```text
PDF Upload
      │
      ▼
Text Extraction (PyMuPDF)
      │
      ▼
Chunking
      │
      ▼
Embedding Generation
(Sentence Transformers)
      │
      ▼
FAISS Vector Store
      │
      ▼
User Question
      │
      ▼
Multi-Query Reformulation
(DeepSeek via Ollama)
      │
      ▼
Document Retrieval
      │
      ▼
Cross-Encoder Re-ranking
      │
      ▼
Answerability Check
      │
      ├── Relevant Context Found
      │         │
      │         ▼
      │    Generate Answer
      │
      └── No Relevant Context
                │
                ▼
      "I do not know based
         on the document"
```

---

## 📂 Project Structure

```text
project/
│
├── main.py
├── README.md
├── requirements.txt
│
├── articles/
│   └── Article.pdf
│
└── assets/
```

### Folder Description

| File/Folder          | Description                         |
| -------------------- | ----------------------------------- |
| main.py              | Main Streamlit Mini-RAG application |
| requirements.txt     | Project dependencies                |
| README.md            | Project documentation               |
| articles/Article.pdf | Sample document for testing         |
| assets/              | Screenshots and project resources   |

---

## 🎓 Assignment Requirement Coverage

This project implements a complete Mini-RAG pipeline without using LangChain, LlamaIndex, or any other RAG framework.

### Implemented Components

* PDF document ingestion
* Text chunking
* Sentence Transformer embeddings
* FAISS vector retrieval
* Multi-query retrieval
* Cross-encoder re-ranking
* Context-grounded answer generation
* Answerability detection
* "I do not know" fallback response
* Source chunk display

### Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Dense Vector Retrieval
* Re-ranking
* Prompt Engineering
* Hallucination Reduction
* Local LLM Deployment

---

## 📊 Current Retrieval Pipeline

```text
Question
    │
    ▼
Multi-Query Generation
    │
    ▼
FAISS Retrieval
    │
    ▼
Cross-Encoder Re-ranking
    │
    ▼
Answerability Check
    │
    ├── Answer Found
    │       │
    │       ▼
    │   DeepSeek Response
    │
    └── No Answer Found
            │
            ▼
      "I do not know"
```

---

## 🔮 Future Improvements

* Hybrid Search (BM25 + Dense Retrieval)
* Persistent Vector Database
* Multi-PDF Retrieval
* Citation Highlighting
* Chat Memory
* Streaming Responses
* RAG Evaluation (RAGAS)
* Metadata-Based Filtering
* Agentic Retrieval Pipeline
