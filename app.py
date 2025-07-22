import streamlit as st
import faiss
import numpy as np
import fitz  # PyMuPDF
import requests
import json
import uuid
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any
import time
import pickle

# Configuration
LM_STUDIO_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Using a lighter model for better performance
PDF_PATH = "data/KKH Information file.pdf"
VECTOR_STORE_PATH = "vector_store.pkl"
CHUNKS_PATH = "text_chunks.pkl"

class NursingChatbot:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.text_chunks = []
        self.load_or_create_embeddings()
    
    @st.cache_resource
    def load_embedding_model(_self):
        """Load the sentence transformer model"""
        return SentenceTransformer(EMBEDDING_MODEL)
    
    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones from PDF"""
        self.embedding_model = self.load_embedding_model()
        
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(CHUNKS_PATH):
            # Load existing embeddings
            with open(VECTOR_STORE_PATH, 'rb') as f:
                self.vector_store = pickle.load(f)
            with open(CHUNKS_PATH, 'rb') as f:
                self.text_chunks = pickle.load(f)
        else:
            # Create new embeddings
            self.create_embeddings_from_pdf()
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract and chunk text from PDF"""
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found at {pdf_path}")
            return []
        
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split into smaller chunks (approximately 500 characters)
            sentences = text.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        doc.close()
        return [chunk for chunk in chunks if len(chunk) > 50]  # Filter out very short chunks
    
    def create_embeddings_from_pdf(self):
        """Create FAISS embeddings from PDF content"""
        with st.spinner("Processing nursing guide PDF..."):
            self.text_chunks = self.extract_text_from_pdf(PDF_PATH)
            
            if not self.text_chunks:
                st.error("No text chunks extracted from PDF")
                return
            
            # Create embeddings
            embeddings = self.embedding_model.encode(self.text_chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))
            
            # Save embeddings for future use
            with open(VECTOR_STORE_PATH, 'wb') as f:
                pickle.dump(self.vector_store, f)
            with open(CHUNKS_PATH, 'wb') as f:
                pickle.dump(self.text_chunks, f)
            
            st.success(f"Successfully processed {len(self.text_chunks)} text chunks from the nursing guide!")
    
    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant text chunks using semantic similarity"""
        if not self.vector_store or not self.text_chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_store.search(query_embedding.astype('float32'), top_k)
        
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                relevant_chunks.append(self.text_chunks[idx])
        
        return relevant_chunks
    
    def generate_response(self, user_question: str, relevant_chunks: List[str]) -> str:
        """Generate response using OpenRouter API"""
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""You are a helpful nursing assistant specialized in pediatric and neonatal care. 
        Based on the following nursing guide information, please answer the user's question accurately and concisely.

        Nursing Guide Information:
        {context}

        User Question: {user_question}

        Please provide a clear, accurate answer based on the nursing guide information provided. If the information is not sufficient to answer the question, please say so and suggest consulting additional resources."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.secrets.get('OPENROUTER_API_KEY', 'your-api-key-here')}"
        }
        
        data = {
            "model": "microsoft/wizardlm-2-8x22b",  # Using a good model from OpenRouter
            "messages": [
                {"role": "system", "content": "You are a helpful nursing assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(LM_STUDIO_URL, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Unable to get response from AI model (Status: {response.status_code})"
        except requests.RequestException as e:
            return f"Error: Failed to connect to AI service. Please try again later."

class FluidCalculator:
    @staticmethod
    def calculate_maintenance_fluid(weight_kg: float) -> Dict[str, Any]:
        """Calculate daily maintenance fluid using Holliday-Segar method"""
        if weight_kg <= 0:
            return {"error": "Weight must be positive"}
        
        daily_ml = 0
        hourly_ml = 0
        
        # First 10 kg: 100 ml/kg
        if weight_kg <= 10:
            daily_ml = weight_kg * 100
        else:
            daily_ml = 10 * 100  # First 10 kg
            remaining_weight = weight_kg - 10
            
            # Next 10 kg: 50 ml/kg
            if remaining_weight <= 10:
                daily_ml += remaining_weight * 50
            else:
                daily_ml += 10 * 50  # Next 10 kg
                # Remaining kg: 20 ml/kg
                daily_ml += (remaining_weight - 10) * 20
        
        hourly_ml = daily_ml / 24
        
        return {
            "weight_kg": weight_kg,
            "daily_ml": round(daily_ml, 1),
            "hourly_ml": round(hourly_ml, 1),
            "ml_per_hour_range": f"{round(hourly_ml * 0.8, 1)}-{round(hourly_ml * 1.2, 1)}"
        }

class QuizModule:
    def __init__(self):
        self.questions = self.generate_nursing_questions()
    
    def generate_nursing_questions(self) -> List[Dict]:
        """Generate nursing quiz questions"""
        return [
            {
                "question": "What is the normal heart rate range for neonates (0-28 days)?",
                "options": ["80-100 bpm", "100-120 bpm", "120-160 bpm", "160-200 bpm"],
                "correct": 2,
                "explanation": "Normal neonatal heart rate is 120-160 bpm at rest."
            },
            {
                "question": "What is the recommended dose of paracetamol for children?",
                "options": ["5-10 mg/kg", "10-15 mg/kg", "15-20 mg/kg", "20-25 mg/kg"],
                "correct": 1,
                "explanation": "The recommended dose is 10-15 mg/kg every 4-6 hours."
            },
            {
                "question": "At what age should solid foods typically be introduced?",
                "options": ["3-4 months", "4-5 months", "6 months", "8-9 months"],
                "correct": 2,
                "explanation": "WHO recommends introducing complementary foods at 6 months."
            },
            {
                "question": "What is the normal respiratory rate for infants (1-12 months)?",
                "options": ["12-20 breaths/min", "20-30 breaths/min", "30-40 breaths/min", "40-50 breaths/min"],
                "correct": 2,
                "explanation": "Normal respiratory rate for infants is 30-40 breaths per minute."
            },
            {
                "question": "What is the antidote for paracetamol overdose?",
                "options": ["Naloxone", "N-acetylcysteine", "Flumazenil", "Atropine"],
                "correct": 1,
                "explanation": "N-acetylcysteine (NAC) is the specific antidote for paracetamol poisoning."
            },
            {
                "question": "What is the normal blood pressure range for school-age children (6-12 years)?",
                "options": ["80/50 - 90/60 mmHg", "90/60 - 110/70 mmHg", "100/60 - 120/80 mmHg", "110/70 - 130/85 mmHg"],
                "correct": 2,
                "explanation": "Normal BP for school-age children is approximately 100/60 - 120/80 mmHg."
            },
            {
                "question": "How long should exclusive breastfeeding continue?",
                "options": ["3 months", "4 months", "6 months", "12 months"],
                "correct": 2,
                "explanation": "WHO recommends exclusive breastfeeding for the first 6 months of life."
            },
            {
                "question": "What is the normal temperature range for children?",
                "options": ["36.0-37.0°C", "36.5-37.5°C", "37.0-38.0°C", "37.5-38.5°C"],
                "correct": 1,
                "explanation": "Normal body temperature for children is 36.5-37.5°C (97.7-99.5°F)."
            },
            {
                "question": "At what weight should car seats be rear-facing until?",
                "options": ["9 kg", "13 kg", "15 kg", "18 kg"],
                "correct": 1,
                "explanation": "Children should remain rear-facing until at least 13 kg (about 15 months)."
            },
            {
                "question": "What is the recommended iron dose for iron deficiency anemia in children?",
                "options": ["1-2 mg/kg/day", "3-6 mg/kg/day", "6-10 mg/kg/day", "10-15 mg/kg/day"],
                "correct": 1,
                "explanation": "The recommended therapeutic dose is 3-6 mg/kg/day of elemental iron."
            },
            {
                "question": "When should the anterior fontanelle typically close?",
                "options": ["6-8 months", "8-12 months", "12-18 months", "18-24 months"],
                "correct": 2,
                "explanation": "The anterior fontanelle typically closes between 12-18 months of age."
            },
            {
                "question": "What is the normal urine output for children?",
                "options": ["0.5-1 ml/kg/hr", "1-2 ml/kg/hr", "2-3 ml/kg/hr", "3-4 ml/kg/hr"],
                "correct": 1,
                "explanation": "Normal urine output for children is 1-2 ml/kg/hr."
            },
            {
                "question": "At what age do children typically begin to walk independently?",
                "options": ["9-12 months", "12-15 months", "15-18 months", "18-21 months"],
                "correct": 1,
                "explanation": "Most children walk independently between 12-15 months of age."
            },
            {
                "question": "What is the recommended fluid intake for fever management?",
                "options": ["Normal intake", "1.2x normal intake", "1.5x normal intake", "2x normal intake"],
                "correct": 1,
                "explanation": "Increase fluid intake by approximately 20% (1.2x) for each degree of fever."
            },
            {
                "question": "When should a child first visit the dentist?",
                "options": ["6 months", "12 months", "18 months", "24 months"],
                "correct": 1,
                "explanation": "First dental visit should occur by 12 months or within 6 months of first tooth eruption."
            }
        ]

def main():
    st.set_page_config(
        page_title="KKH Nursing Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = NursingChatbot()
    
    if "quiz_module" not in st.session_state:
        st.session_state.quiz_module = QuizModule()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
    
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = []
    
    if "show_fluid_calc" not in st.session_state:
        st.session_state.show_fluid_calc = False
    
    # Header
    st.markdown('<h1 class="main-header">🏥 KKH Nursing Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # New Chat Button
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.quiz_active = False
            st.session_state.show_fluid_calc = False
            st.rerun()
        
        st.divider()
        
        # Quiz Controls
        st.subheader("📝 Quiz Module")
        if not st.session_state.quiz_active:
            if st.button("🚀 Start Quiz", use_container_width=True):
                st.session_state.quiz_active = True
                st.session_state.current_question = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []
                st.session_state.show_fluid_calc = False
                st.rerun()
        else:
            if st.button("🔄 Reset Quiz", use_container_width=True):
                st.session_state.quiz_active = False
                st.session_state.current_question = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []
                st.rerun()
            
            st.write(f"Question: {st.session_state.current_question + 1}/{len(st.session_state.quiz_module.questions)}")
            st.write(f"Score: {st.session_state.quiz_score}")
        
        st.divider()
        
        # Fluid Calculator Toggle
        st.subheader("🧮 Fluid Calculator")
        if st.button("⚖️ Toggle Calculator", use_container_width=True):
            st.session_state.show_fluid_calc = not st.session_state.show_fluid_calc
            st.session_state.quiz_active = False
            st.rerun()
        
        st.divider()
        
        # Chat History
        st.subheader("💬 Chat History")
        if st.session_state.chat_history:
            for i, (user_msg, _) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                st.write(f"💭 {user_msg[:50]}..." if len(user_msg) > 50 else f"💭 {user_msg}")
        else:
            st.write("No chat history yet")
    
    # Main content area
    if st.session_state.quiz_active:
        # Quiz Interface
        st.header("📝 Nursing Knowledge Quiz")
        
        questions = st.session_state.quiz_module.questions
        current_q = st.session_state.current_question
        
        if current_q < len(questions):
            question_data = questions[current_q]
            
            st.subheader(f"Question {current_q + 1} of {len(questions)}")
            st.write(question_data["question"])
            
            # Answer options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"A) {question_data['options'][0]}", key="option_a"):
                    st.session_state.quiz_answers.append(0)
                    if 0 == question_data["correct"]:
                        st.session_state.quiz_score += 1
                    st.session_state.current_question += 1
                    st.rerun()
                
                if st.button(f"C) {question_data['options'][2]}", key="option_c"):
                    st.session_state.quiz_answers.append(2)
                    if 2 == question_data["correct"]:
                        st.session_state.quiz_score += 1
                    st.session_state.current_question += 1
                    st.rerun()
            
            with col2:
                if st.button(f"B) {question_data['options'][1]}", key="option_b"):
                    st.session_state.quiz_answers.append(1)
                    if 1 == question_data["correct"]:
                        st.session_state.quiz_score += 1
                    st.session_state.current_question += 1
                    st.rerun()
                
                if st.button(f"D) {question_data['options'][3]}", key="option_d"):
                    st.session_state.quiz_answers.append(3)
                    if 3 == question_data["correct"]:
                        st.session_state.quiz_score += 1
                    st.session_state.current_question += 1
                    st.rerun()
        
        else:
            # Quiz completed
            st.header("🎉 Quiz Complete!")
            score_percentage = (st.session_state.quiz_score / len(questions)) * 100
            
            st.metric("Final Score", f"{st.session_state.quiz_score}/{len(questions)}", f"{score_percentage:.1f}%")
            
            if score_percentage >= 80:
                st.success("Excellent work! 🌟")
            elif score_percentage >= 60:
                st.info("Good job! Keep studying! 📚")
            else:
                st.warning("Consider reviewing the material. 💪")
            
            if st.button("🔄 Restart Quiz"):
                st.session_state.quiz_active = False
                st.session_state.current_question = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []
                st.rerun()
    
    elif st.session_state.show_fluid_calc:
        # Fluid Calculator Interface
        st.header("🧮 Fluid Calculator")
        st.subheader("Holliday-Segar Method")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            weight = st.number_input("Patient Weight (kg)", min_value=0.1, max_value=150.0, value=10.0, step=0.1)
            
            if st.button("Calculate", use_container_width=True):
                result = FluidCalculator.calculate_maintenance_fluid(weight)
                
                if "error" not in result:
                    st.session_state.fluid_result = result
        
        with col2:
            if hasattr(st.session_state, 'fluid_result'):
                result = st.session_state.fluid_result
                st.subheader("Maintenance Fluid Requirements")
                
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Daily Total", f"{result['daily_ml']} ml")
                with col2b:
                    st.metric("Hourly Rate", f"{result['hourly_ml']} ml/hr")
                with col2c:
                    st.metric("Range", f"{result['ml_per_hour_range']} ml/hr")
                
                st.info("""
                **Holliday-Segar Method:**
                - First 10 kg: 100 ml/kg/day
                - Next 10 kg: 50 ml/kg/day  
                - Each kg > 20 kg: 20 ml/kg/day
                """)
    
    else:
        # Chat Interface
        st.header("💬 Chat with Nursing Assistant")
        
        # Prompt suggestions
        st.subheader("💡 Quick Questions")
        prompt_suggestions = [
            "What is normal heart rate for neonates?",
            "Dose of NAC in paracetamol overdose?",
            "Normal respiratory rate for infants?",
            "When to introduce solid foods?",
            "Iron dosage for anemia in children?"
        ]
        
        cols = st.columns(len(prompt_suggestions))
        for i, suggestion in enumerate(prompt_suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"prompt_{i}", use_container_width=True):
                    # Auto-fill and send the prompt
                    user_question = suggestion
                    relevant_chunks = st.session_state.chatbot.search_relevant_chunks(user_question)
                    
                    with st.spinner("Generating response..."):
                        response = st.session_state.chatbot.generate_response(user_question, relevant_chunks)
                    
                    st.session_state.chat_history.append((user_question, response))
                    st.rerun()
        
        st.divider()
        
        # Chat history display
        for user_msg, assistant_msg in st.session_state.chat_history:
            with st.container():
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {assistant_msg}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask me anything about nursing care...")
        
        if user_input:
            # Search for relevant chunks
            relevant_chunks = st.session_state.chatbot.search_relevant_chunks(user_input)
            
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.generate_response(user_input, relevant_chunks)
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, response))
            st.rerun()

if __name__ == "__main__":
    main()
