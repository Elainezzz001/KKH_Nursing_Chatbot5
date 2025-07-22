import os
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import requests

# --- CONFIG ---
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
PDF_PATH = "data/KKH Information file.pdf"
FAISS_INDEX_PATH = "faiss_index.pkl"
CHUNKS_PATH = "chunks.pkl"
QUIZ_PATH = "quiz.pkl"

# --- PDF PARSING & CHUNKING ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- EMBEDDING & FAISS ---
def build_or_load_faiss(chunks, model, index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH):
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        with open(chunks_path, "rb") as f:
            metadatas = pickle.load(f)
        return index, metadatas
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    return index, chunks

def search_faiss(query, model, index, metadatas, top_k=3):
    emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(emb, top_k)
    return [metadatas[i] for i in I[0]]

# --- LLM CHAT ---
def call_zephyr(messages, system_prompt, context):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "zephyr-7b-beta",
        "messages": [
            {"role": "system", "content": system_prompt + "\nContext: " + context},
            *messages
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    resp = requests.post(LM_STUDIO_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# --- FLUID CALCULATOR ---
def calculate_fluid(weight, dehydration):
    if weight <= 10:
        total = weight * 100
    elif weight <= 20:
        total = 1000 + (weight - 10) * 50
    else:
        total = 1500 + (weight - 20) * 20
    if dehydration == "Mild":
        total *= 1.05
    elif dehydration == "Moderate":
        total *= 1.1
    elif dehydration == "Severe":
        total *= 1.2
    rate = total / 24
    return int(total), round(rate, 1)

# --- QUIZ ---
def load_or_generate_quiz(chunks, quiz_path=QUIZ_PATH):
    if os.path.exists(quiz_path):
        with open(quiz_path, "rb") as f:
            return pickle.load(f)
    # Simple auto-generation: pick 15 random sentences as questions
    import random
    questions = []
    for _ in range(15):
        chunk = random.choice(chunks)
        sentences = [s for s in chunk.split('.') if len(s.split()) > 6]
        if not sentences:
            continue
        q = random.choice(sentences).strip()
        options = random.sample(sentences, min(4, len(sentences)))
        if q not in options:
            options[0] = q
        random.shuffle(options)
        questions.append({
            "question": q,
            "options": options,
            "answer": options.index(q),
            "explanation": "See guide for details."
        })
    with open(quiz_path, "wb") as f:
        pickle.dump(questions, f)
    return questions

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="KKH Nursing Chatbot", layout="wide")
    st.title("🏥 KKH Nursing Chatbot")
    st.markdown("""<style>body {font-size: 18px;} .stButton>button {font-size: 18px;} .stTextInput>div>input {font-size: 18px;} </style>""", unsafe_allow_html=True)

    # Load models and data
    with st.spinner("Loading models and knowledge base..."):
        model = SentenceTransformer(EMBEDDING_MODEL)
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            with open(FAISS_INDEX_PATH, "rb") as f:
                index = pickle.load(f)
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
        else:
            text = extract_text_from_pdf(PDF_PATH)
            chunks = chunk_text(text)
            index, chunks = build_or_load_faiss(chunks, model)

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧮 Fluid Calculator", "🧪 Quiz"])

    # --- CHAT TAB ---
    with tab1:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.markdown("**Prompt buttons:**")
        cols = st.columns(3)
        if cols[0].button("How to treat fever?"):
            st.session_state.chat_input = "How to treat fever?"
        if cols[1].button("Fluid requirement for child?"):
            st.session_state.chat_input = "Fluid requirement for child?"
        if cols[2].button("Reset chat"):
            st.session_state.chat_history = []
        chat_input = st.text_input("You:", key="chat_input")
        if st.button("Send") and chat_input:
            context_chunks = search_faiss(chat_input, model, index, chunks)
            context = "\n".join(context_chunks)
            messages = st.session_state.chat_history + [{"role": "user", "content": chat_input}]
            system_prompt = "You are a helpful KKH nurse assistant. Answer based on the provided context."
            try:
                bot_reply = call_zephyr(messages, system_prompt, context)
            except Exception as e:
                bot_reply = f"[Error contacting LLM: {e}]"
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            st.session_state.chat_input = ""
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")

    # --- FLUID CALCULATOR TAB ---
    with tab2:
        st.header("Fluid Calculator")
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=100.0, value=10.0)
        dehydration = st.selectbox("Dehydration", ["None", "Mild", "Moderate", "Severe"])
        if st.button("Calculate Fluid"):
            total, rate = calculate_fluid(weight, dehydration)
            st.success(f"Total fluid: {total} ml\nInfusion rate: {rate} ml/hr")

    # --- QUIZ TAB ---
    with tab3:
        st.header("Nursing Quiz")
        if "quiz" not in st.session_state:
            st.session_state.quiz = load_or_generate_quiz(chunks)
            st.session_state.quiz_idx = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = []
        quiz = st.session_state.quiz
        idx = st.session_state.quiz_idx
        if idx < len(quiz):
            q = quiz[idx]
            st.write(f"Q{idx+1}: {q['question']}")
            option = st.radio("Options", q["options"], key=f"quiz_{idx}")
            if st.button("Submit Answer", key=f"submit_{idx}"):
                correct = q["options"].index(option) == q["answer"]
                st.session_state.quiz_answers.append((option, correct))
                if correct:
                    st.success("Correct!")
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"Incorrect. Correct answer: {q['options'][q['answer']]}")
                st.info(q["explanation"])
                st.session_state.quiz_idx += 1
        else:
            st.success(f"Quiz complete! Your score: {st.session_state.quiz_score}/{len(quiz)}")
            if st.button("Restart Quiz"):
                st.session_state.quiz = load_or_generate_quiz(chunks)
                st.session_state.quiz_idx = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []

if __name__ == "__main__":
    main()
