import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# -----------------------------
# 1️⃣ PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_text = ""

    with fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    ) as doc:

        for page in doc:
            pdf_text += page.get_text("text")

    return pdf_text


# -----------------------------
# 2️⃣ Text Chunking
# -----------------------------
def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size - overlap
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


# -----------------------------
# 3️⃣ Vector Store Creation
# -----------------------------
@st.cache_resource
def create_vector_store(chunks):

    embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
    )

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index, chunks, embedder


# -----------------------------
# 4️⃣ Multi Query Retrieval
# -----------------------------
def retrieve_context_multiquery(
    query,
    embedder,
    index,
    chunks,
    k=3
):

    reform_prompt = f"""
You are a helpful assistant.

Generate 3 alternative versions of the user question.

Use different wording and synonyms.

create bullet points which are easy to read.

Question:
{query}

Return only the reformulated queries,
one per line.
"""

    try:

        response = ollama.generate(
            model="llama3.2:latest",
            prompt=reform_prompt,
            options={
                "temperature": 0.6
            }
        )

        reformulated_text = response["response"]

        alt_queries = []

        for line in reformulated_text.split("\n"):
            line = line.strip()

            if not line:
                continue

            if "alternative versions" in line.lower():
                continue

            line = re.sub(
                r'^[\d\.\-\•]+\s*',
                '',
                line
            )

            alt_queries.append(line)

    except Exception as e:

        st.warning(
            f"Query reformulation failed: {e}"
        )

        alt_queries = []

    all_queries = [query] + alt_queries

    st.write(
        "🔁 Reformulated Queries:",
        all_queries
    )

    retrieved = []

    for q in all_queries:

        q_emb = embedder.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        q_emb = q_emb.astype(
            np.float32
        )

        distances, indices = index.search(
            q_emb,
            k
        )

        for d, idx in zip(
            distances[0],
            indices[0]
        ):
            retrieved.append(
                (chunks[idx], float(d))
            )

    unique_results = {}

    for chunk, score in retrieved:

        if (
            chunk not in unique_results
            or score < unique_results[chunk]
        ):
            unique_results[chunk] = score

    return [
        (chunk, score)
        for chunk, score
        in unique_results.items()
    ]


# -----------------------------
# 5️⃣ Answerability Check
# -----------------------------
def is_answerable(
    retrieved_contexts,
    threshold=1.8
):

    if not retrieved_contexts:
        return False

    best_score = min(
        score
        for _, score
        in retrieved_contexts
    )

    return best_score < threshold


# -----------------------------
# 6️⃣ Re-ranking
# -----------------------------
@st.cache_resource
def load_reranker():

    return CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def rerank_contexts(
    query,
    contexts,
    top_n=5
):

    reranker = load_reranker()

    pairs = [
        (query, ctx)
        for ctx in contexts
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(contexts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        ctx
        for ctx, _
        in ranked[:top_n]
    ]


# -----------------------------
# 7️⃣ Remove Thinking Tags
# -----------------------------
def remove_think_tags(text):

    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )


# -----------------------------
# 8️⃣ Answer Generation
# -----------------------------
def generate_answer(
    query,
    contexts
):

    if not contexts:

        return (
            "I do not know based on "
            "the uploaded document."
        )

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
You are a question-answering assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
respond exactly:

I do not know based on the document.

Context:
{context_text}

Question:
{query}
"""

    response = ollama.generate(
    model="llama3.2:latest",
        prompt=prompt,
        options={
            "temperature": 0.2,
            "max_tokens": 1000
        }
    )

    return remove_think_tags(
        response["response"]
    )


# -----------------------------
# 9️⃣ Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Mini-RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title(
    "🧠 Mini-RAG PDF Chatbot"
)

st.markdown(
    """
Upload a PDF and ask questions about it.

Features:
- PDF Question Answering
- FAISS Vector Search
- Multi-Query Retrieval
- Cross-Encoder Re-ranking
- Hallucination Reduction
- "I Don't Know" Detection
"""
)

uploaded_file = st.file_uploader(
    "📤 Upload a PDF",
    type=["pdf"]
)

query = st.text_area(
    "💬 Ask a question:"
)

if uploaded_file:

    with st.spinner(
        "Processing PDF..."
    ):

        pdf_text = extract_text_from_pdf(
            uploaded_file
        )

        chunks = chunk_text(
            pdf_text
        )

        index, docs, embedder = (
            create_vector_store(
                chunks
            )
        )

    st.success(
        f"PDF processed successfully. "
        f"Created {len(chunks)} chunks."
    )

    if st.button(
        "🔍 Get Answer"
    ):

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching document..."
            ):

                retrieved = (
                    retrieve_context_multiquery(
                        query,
                        embedder,
                        index,
                        docs
                    )
                )

                if not is_answerable(
                    retrieved
                ):

                    answer = (
                        "I could not find "
                        "relevant information "
                        "in the uploaded document."
                    )

                    ranked_contexts = []

                else:

                    contexts = [
                        chunk
                        for chunk, _
                        in retrieved
                    ]

                    ranked_contexts = (
                        rerank_contexts(
                            query,
                            contexts,
                            top_n=5
                        )
                    )

                    answer = (
                        generate_answer(
                            query,
                            ranked_contexts
                        )
                    )

            st.markdown(
                "## 🧠 Answer"
            )

            st.write(answer)

            if ranked_contexts:

                st.markdown(
                    "## 📚 Retrieved Sources"
                )

                for i, chunk in enumerate(
                    ranked_contexts,
                    start=1
                ):

                    with st.expander(
                        f"Source Chunk {i}"
                    ):

                        st.write(chunk)

else:

    st.info(
        "Upload a PDF to begin."
    )
