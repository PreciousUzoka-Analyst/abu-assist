"""
ABU-Assist: A Grounded RAG Assistant for University Administrative Information
Streamlit Community Cloud version (free, permanent hosted link).
"""

import os
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="ABU-Assist: Grounded RAG Assistant")

DOCUMENTS = {
    "Student_Handbook_2023_2024.pdf": "https://abu.edu.ng/wp-content/uploads/2024/04/Updated-Student-Handbook-for-2023-2024-Session.pdf",
    "Examination_Regulations_2023.pdf": "https://abu.edu.ng/wp-content/uploads/2024/08/EXAMINATION-MANAGEMENT-EXAMINATION-REGULATIONS.pdf",
    "Quality_Assurance_Policy.pdf": "https://abu.edu.ng/wp-content/uploads/2023/08/abu-quality-assuarance-policy.pdf",
    "ICT_Policy_2020.pdf": "https://abu.edu.ng/wp-content/uploads/2025/11/ABU-ICT-Policy-December-2020-Corrected-23-04-25.pdf",
}

DISTANCE_THRESHOLD = 1.1


@st.cache_resource(show_spinner="Setting up ABU-Assist (first load only, ~2-3 minutes)...")
def build_index():
    os.makedirs("corpus", exist_ok=True)
    for filename, url in DOCUMENTS.items():
        path = os.path.join("corpus", filename)
        if not os.path.exists(path):
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

    def extract_text(pdf_path):
        reader = PdfReader(pdf_path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    def chunk_text(text, chunk_size=500, overlap=80):
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return [c.strip() for c in chunks if len(c.strip()) > 40]

    all_chunks = []
    for filename in DOCUMENTS.keys():
        for page_num, page_text in extract_text(os.path.join("corpus", filename)):
            for chunk in chunk_text(page_text):
                all_chunks.append({"text": chunk, "source": filename, "page": page_num})

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return all_chunks, embedder, index, tok, model


all_chunks, embedder, index, tok, model = build_index()


def retrieve(question, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({**all_chunks[idx], "distance": float(dist)})
    return results


def generate(prompt):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)


def answer_question(question):
    results = retrieve(question, top_k=3)
    relevant = [r for r in results if r["distance"] < DISTANCE_THRESHOLD]
    if not relevant:
        return ("I couldn't find this in the available ABU documents. "
                 "Please check with the relevant office directly."), []

    context = "\n\n".join(r["text"] for r in relevant)
    prompt = (f"Question: {question}\n"
              "Answer using only the context below. If the answer is not there, say you don't know.\n\n"
              f"Context: {context}\n\nAnswer:")
    answer = generate(prompt)
    sources = list(dict.fromkeys(f"{r['source']} (page {r['page']})" for r in relevant))
    return answer, sources


# ---------- UI ----------
st.title("ABU-Assist: Grounded RAG Assistant")
st.caption("Answers are grounded strictly in official ABU documents (Student Handbook, "
           "Examination Regulations, Quality Assurance Policy, ICT Policy). "
           "The system cites its source or says it doesn't know.")

question = st.text_input("Ask an ABU administrative question",
                          placeholder="e.g. What happens if I miss an exam?")

example_cols = st.columns(3)
examples = [
    "What happens if a student misses an examination?",
    "What does the ICT policy say about acceptable use?",
    "What is the capital of France?",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex):
        question = ex

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        answer, sources = answer_question(question)
    st.markdown(f"**Answer:** {answer}")
    if sources:
        st.markdown("**Sources:**")
        for s in sources:
            st.markdown(f"- {s}")
    else:
        st.markdown("**Sources:** None")
