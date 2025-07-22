import streamlit as st
import os
import json
import pickle
import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Tuple
import time

# Set page config
st.set_page_config(
    page_title="KKH Nursing Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
LM_STUDIO_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
PDF_PATH = "data/KKH Information file.pdf"
EMBEDDINGS_PATH = "embeddings.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = None
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def load_embedding_model(self):
        """Load the embedding model"""
        if self.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self.embedding_model
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def create_embeddings(self, force_reload=False):
        """Create or load embeddings for PDF content"""
        if os.path.exists(EMBEDDINGS_PATH) and not force_reload:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
                self.index = data['index']
            return
        
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text = self.extract_text_from_pdf(PDF_PATH)
        
        if not text:
            st.error("Failed to extract text from PDF")
            return
        
        # Chunk the text
        with st.spinner("Chunking document..."):
            self.chunks = self.chunk_text(text)
        
        # Load embedding model
        model = self.load_embedding_model()
        
        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = model.encode(self.chunks, show_progress_bar=True)
            self.embeddings = embeddings
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'index': self.index
            }, f)
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[str]:
        """Search for similar chunks using semantic similarity"""
        if self.index is None:
            return []
        
        model = self.load_embedding_model()
        query_embedding = model.encode([query])
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

class ChatBot:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            st.error("Please set OPENROUTER_API_KEY environment variable")
    
    def generate_response(self, query: str, context: str = "") -> str:
        """Generate response using Zephyr model via OpenRouter"""
        if not self.api_key:
            return "Error: API key not configured"
        
        system_prompt = """You are a helpful nursing assistant for KKH (KK Women's and Children's Hospital). 
        You provide accurate, evidence-based medical information to nursing staff. 
        Always be professional, concise, and focus on practical nursing care.
        If you're unsure about something, recommend consulting hospital protocols or senior staff."""
        
        user_prompt = f"""Context from nursing guidelines:
{context}

Question: {query}

Please provide a helpful response based on the context and your nursing knowledge."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(LM_STUDIO_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error communicating with AI service: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

class FluidCalculator:
    @staticmethod
    def calculate_maintenance_fluid(weight_kg: float, age_years: int = None) -> Dict:
        """Calculate maintenance fluid requirements"""
        if weight_kg <= 0:
            return {"error": "Weight must be positive"}
        
        # Holliday-Segar method
        if weight_kg <= 10:
            daily_ml = weight_kg * 100
        elif weight_kg <= 20:
            daily_ml = 1000 + (weight_kg - 10) * 50
        else:
            daily_ml = 1500 + (weight_kg - 20) * 20
        
        hourly_ml = daily_ml / 24
        
        return {
            "daily_ml": daily_ml,
            "hourly_ml": round(hourly_ml, 1),
            "weight_kg": weight_kg
        }
    
    @staticmethod
    def calculate_dehydration_fluid(weight_kg: float, dehydration_percent: float) -> Dict:
        """Calculate fluid replacement for dehydration"""
        if weight_kg <= 0 or dehydration_percent < 0:
            return {"error": "Invalid input values"}
        
        deficit_ml = weight_kg * 1000 * (dehydration_percent / 100)
        
        return {
            "deficit_ml": round(deficit_ml),
            "replacement_over_24h": round(deficit_ml / 24, 1),
            "dehydration_percent": dehydration_percent
        }

class QuizModule:
    def __init__(self):
        self.questions = [
            {
                "question": "What is the normal heart rate range for a newborn?",
                "options": ["60-100 bpm", "100-160 bpm", "120-200 bpm", "80-120 bpm"],
                "correct": 1,
                "explanation": "Normal newborn heart rate is 100-160 bpm when awake and alert."
            },
            {
                "question": "What is the correct compression-to-ventilation ratio for infant CPR?",
                "options": ["15:2", "30:2", "5:1", "3:1"],
                "correct": 1,
                "explanation": "For infant CPR, the ratio is 30 compressions to 2 ventilations."
            },
            {
                "question": "At what blood glucose level should hypoglycemia be treated in neonates?",
                "options": ["<2.6 mmol/L", "<3.0 mmol/L", "<4.0 mmol/L", "<5.0 mmol/L"],
                "correct": 0,
                "explanation": "Neonatal hypoglycemia is typically treated when blood glucose is <2.6 mmol/L."
            },
            {
                "question": "What is the recommended depth of chest compressions for an infant?",
                "options": ["1/3 of chest depth", "1/4 of chest depth", "1/2 of chest depth", "2 inches"],
                "correct": 0,
                "explanation": "Chest compressions should be at least 1/3 the depth of the chest for infants."
            },
            {
                "question": "What is the normal respiratory rate for a newborn?",
                "options": ["10-20 breaths/min", "20-40 breaths/min", "30-60 breaths/min", "40-80 breaths/min"],
                "correct": 2,
                "explanation": "Normal newborn respiratory rate is 30-60 breaths per minute."
            },
            {
                "question": "What is the first-line treatment for severe neonatal hypoglycemia?",
                "options": ["Oral glucose", "IV dextrose 10%", "Glucagon injection", "Sugar water"],
                "correct": 1,
                "explanation": "IV dextrose 10% is the first-line treatment for severe neonatal hypoglycemia."
            },
            {
                "question": "What is the normal temperature range for a newborn?",
                "options": ["36.0-37.0°C", "36.5-37.5°C", "37.0-38.0°C", "35.5-36.5°C"],
                "correct": 1,
                "explanation": "Normal newborn temperature range is 36.5-37.5°C (97.7-99.5°F)."
            },
            {
                "question": "How often should vital signs be monitored in a stable postpartum patient?",
                "options": ["Every 15 minutes", "Every 30 minutes", "Every hour", "Every 4 hours"],
                "correct": 3,
                "explanation": "Stable postpartum patients typically have vital signs monitored every 4 hours."
            },
            {
                "question": "What is the recommended position for a newborn during feeding?",
                "options": ["Supine", "Prone", "Side-lying", "Semi-upright"],
                "correct": 3,
                "explanation": "Semi-upright position helps prevent aspiration during feeding."
            },
            {
                "question": "What is the normal urine output for a newborn per day?",
                "options": ["0.5-1 ml/kg/hr", "1-2 ml/kg/hr", "2-3 ml/kg/hr", "3-4 ml/kg/hr"],
                "correct": 1,
                "explanation": "Normal newborn urine output is 1-2 ml/kg/hr."
            },
            {
                "question": "What is the primary sign of respiratory distress in newborns?",
                "options": ["Cyanosis", "Tachypnea", "Grunting", "All of the above"],
                "correct": 3,
                "explanation": "Respiratory distress can present with cyanosis, tachypnea, grunting, and retractions."
            },
            {
                "question": "What is the recommended frequency for newborn feeding?",
                "options": ["Every hour", "Every 2-3 hours", "Every 4 hours", "Every 6 hours"],
                "correct": 1,
                "explanation": "Newborns should be fed every 2-3 hours or 8-12 times per day."
            },
            {
                "question": "What is the normal blood pressure range for a term newborn?",
                "options": ["40-60/20-40 mmHg", "50-70/25-45 mmHg", "60-80/30-50 mmHg", "70-90/35-55 mmHg"],
                "correct": 1,
                "explanation": "Normal term newborn blood pressure is approximately 50-70/25-45 mmHg."
            },
            {
                "question": "What is the first step in newborn resuscitation?",
                "options": ["Chest compressions", "Bag-mask ventilation", "Dry and stimulate", "Intubation"],
                "correct": 2,
                "explanation": "The first step in newborn resuscitation is to dry, warm, and stimulate the baby."
            },
            {
                "question": "What is the recommended duration of exclusive breastfeeding?",
                "options": ["3 months", "4 months", "6 months", "12 months"],
                "correct": 2,
                "explanation": "WHO recommends exclusive breastfeeding for the first 6 months of life."
            }
        ]

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = {
            'active': False,
            'current_question': 0,
            'score': 0,
            'answers': [],
            'show_result': False
        }
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatBot()
    if 'quiz_module' not in st.session_state:
        st.session_state.quiz_module = QuizModule()

def render_sidebar():
    """Render sidebar with navigation"""
    st.sidebar.title("🏥 KKH Nursing Assistant")
    
    # Logo
    if os.path.exists("logo/photo_2025-06-16_15-57-21.jpg"):
        st.sidebar.image("logo/photo_2025-06-16_15-57-21.jpg", width=200)
    
    st.sidebar.markdown("---")
    
    # Navigation buttons
    if st.sidebar.button("💬 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.quiz_state['active'] = False
        st.rerun()
    
    if st.sidebar.button("📚 Start Quiz", use_container_width=True):
        st.session_state.quiz_state = {
            'active': True,
            'current_question': 0,
            'score': 0,
            'answers': [],
            'show_result': False
        }
        st.rerun()
    
    if st.sidebar.button("💧 Fluid Calculator", use_container_width=True):
        st.session_state.show_calculator = True
        st.session_state.quiz_state['active'] = False
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Pre-defined questions
    st.sidebar.subheader("Quick Questions")
    quick_questions = [
        "How to treat hypoglycemia in neonates?",
        "What is the CPR ratio for infants?",
        "Normal vital signs for newborns?",
        "Steps for newborn resuscitation?",
        "Breastfeeding guidelines?"
    ]
    
    for question in quick_questions:
        if st.sidebar.button(f"❓ {question}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

def render_chat_interface():
    """Render main chat interface"""
    st.title("🏥 KKH Nursing Chatbot")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about nursing care..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search for relevant context
                relevant_chunks = st.session_state.doc_processor.search_similar_chunks(prompt)
                context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
                
                # Generate response
                response = st.session_state.chatbot.generate_response(prompt, context)
                st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})

def render_quiz_interface():
    """Render quiz interface"""
    quiz_state = st.session_state.quiz_state
    quiz_module = st.session_state.quiz_module
    
    if not quiz_state['show_result']:
        current_q = quiz_state['current_question']
        question_data = quiz_module.questions[current_q]
        
        st.title("📚 Nursing Knowledge Quiz")
        st.progress((current_q + 1) / len(quiz_module.questions))
        st.subheader(f"Question {current_q + 1} of {len(quiz_module.questions)}")
        
        st.write(question_data['question'])
        
        # Answer options
        answer = st.radio(
            "Choose your answer:",
            question_data['options'],
            key=f"quiz_q_{current_q}"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Submit Answer"):
                selected_index = question_data['options'].index(answer)
                is_correct = selected_index == question_data['correct']
                
                quiz_state['answers'].append({
                    'question': current_q,
                    'selected': selected_index,
                    'correct': is_correct
                })
                
                if is_correct:
                    quiz_state['score'] += 1
                
                if current_q + 1 < len(quiz_module.questions):
                    quiz_state['current_question'] += 1
                else:
                    quiz_state['show_result'] = True
                
                st.rerun()
        
        with col2:
            if st.button("Skip Question"):
                quiz_state['answers'].append({
                    'question': current_q,
                    'selected': -1,
                    'correct': False
                })
                
                if current_q + 1 < len(quiz_module.questions):
                    quiz_state['current_question'] += 1
                else:
                    quiz_state['show_result'] = True
                
                st.rerun()
    
    else:
        # Show results
        st.title("🎯 Quiz Results")
        score = quiz_state['score']
        total = len(quiz_module.questions)
        percentage = (score / total) * 100
        
        st.metric("Your Score", f"{score}/{total} ({percentage:.1f}%)")
        
        if percentage >= 80:
            st.success("🎉 Excellent! You have strong nursing knowledge!")
        elif percentage >= 60:
            st.info("👍 Good job! Consider reviewing some topics.")
        else:
            st.warning("📚 Keep studying! Review the nursing guidelines.")
        
        # Show detailed results
        st.subheader("Detailed Results")
        for i, answer in enumerate(quiz_state['answers']):
            question_data = quiz_module.questions[answer['question']]
            
            with st.expander(f"Question {i+1}: {'✅' if answer['correct'] else '❌'}"):
                st.write(question_data['question'])
                
                if answer['selected'] >= 0:
                    st.write(f"Your answer: {question_data['options'][answer['selected']]}")
                else:
                    st.write("Your answer: Skipped")
                
                st.write(f"Correct answer: {question_data['options'][question_data['correct']]}")
                st.write(f"Explanation: {question_data['explanation']}")
        
        if st.button("Restart Quiz"):
            st.session_state.quiz_state = {
                'active': True,
                'current_question': 0,
                'score': 0,
                'answers': [],
                'show_result': False
            }
            st.rerun()

def render_fluid_calculator():
    """Render fluid calculator interface"""
    st.title("💧 Fluid Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Maintenance Fluid Calculator")
        weight = st.number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
        age = st.number_input("Age (years, optional)", min_value=0, max_value=18, value=0)
        
        if st.button("Calculate Maintenance"):
            result = FluidCalculator.calculate_maintenance_fluid(weight, age)
            if 'error' not in result:
                st.success(f"**Daily fluid requirement:** {result['daily_ml']} ml")
                st.success(f"**Hourly rate:** {result['hourly_ml']} ml/hr")
            else:
                st.error(result['error'])
    
    with col2:
        st.subheader("Dehydration Replacement Calculator")
        weight_dehy = st.number_input("Weight (kg)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, key="dehy_weight")
        dehydration = st.selectbox("Dehydration Level", [3, 5, 7, 10, 15], index=1)
        
        if st.button("Calculate Replacement"):
            result = FluidCalculator.calculate_dehydration_fluid(weight_dehy, dehydration)
            if 'error' not in result:
                st.success(f"**Fluid deficit:** {result['deficit_ml']} ml")
                st.success(f"**Replacement over 24h:** {result['replacement_over_24h']} ml/hr")
            else:
                st.error(result['error'])
    
    st.markdown("---")
    st.info("""
    **Fluid Calculation Guidelines:**
    - Maintenance fluid uses Holliday-Segar method
    - Dehydration replacement is based on percentage of body weight lost
    - Always consider patient's clinical condition and consult protocols
    """)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Initialize document processor and embeddings
    if not hasattr(st.session_state.doc_processor, 'embeddings') or st.session_state.doc_processor.embeddings is None:
        with st.spinner("Initializing knowledge base..."):
            st.session_state.doc_processor.create_embeddings()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if hasattr(st.session_state, 'show_calculator') and st.session_state.show_calculator:
        render_fluid_calculator()
        if st.button("← Back to Chat"):
            st.session_state.show_calculator = False
            st.rerun()
    elif st.session_state.quiz_state['active']:
        render_quiz_interface()
        if st.button("← Back to Chat"):
            st.session_state.quiz_state['active'] = False
            st.rerun()
    else:
        render_chat_interface()

if __name__ == "__main__":
    main()
