import streamlit as st
import requests
import json
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import random
from typing import List, Dict, Tuple
import os
import base64

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback configuration if config.py is not found
    LM_STUDIO_HOST = "192.168.75.1"
    LM_STUDIO_PORT = "1234"
    LM_STUDIO_MODEL = "openhermes-2.5-mistral-7b"
    TEMPERATURE = 0.7
    MAX_TOKENS = 500
    SEARCH_TOP_K = 3
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    CHUNK_SIZE = 500
    PDF_PATH = "data/KKH Information file.pdf"
    PAGE_TITLE = "KKH Nursing Chatbot"
    PAGE_ICON = "🏥"
    LAYOUT = "wide"
    PRIMARY_COLOR = "#2E86AB"

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .feature-card {{
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid {PRIMARY_COLOR};
        margin: 1rem 0;
    }}
    .quick-prompt-btn {{
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        margin: 0.2rem;
        cursor: pointer;
    }}
    .quiz-question {{
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }}
    .correct-answer {{
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        color: #155724;
    }}
    .incorrect-answer {{
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        color: #721c24;
    }}
    .fluid-calc-result {{
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid {PRIMARY_COLOR};
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# Configuration
LM_STUDIO_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"

class KKHChatbot:
    def __init__(self):
        self.embedding_model = None
        self.pdf_chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.quiz_questions = []
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load the embedding model"""
        try:
            model = SentenceTransformer(EMBEDDING_MODEL)
            return model
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return None
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF and split into chunks"""
        try:
            chunks = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Split text into chunks (configurable size)
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk = text[i:i + CHUNK_SIZE].strip()
                    if chunk:
                        chunks.append(chunk)
            
            return chunks
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            return []
    
    def create_embeddings(self, chunks: List[str]):
        """Create embeddings for PDF chunks and build FAISS index"""
        if not self.embedding_model:
            self.embedding_model = self.load_embedding_model()
        
        if not self.embedding_model:
            return False
        
        try:
            # Create embeddings
            embeddings = self.embedding_model.encode(chunks)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            self.pdf_chunks = chunks
            self.embeddings = embeddings
            self.faiss_index = index
            
            return True
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = SEARCH_TOP_K) -> List[str]:
        """Perform semantic search to find relevant chunks"""
        if not self.faiss_index or not self.embedding_model:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Return relevant chunks
            relevant_chunks = [self.pdf_chunks[idx] for idx in indices[0]]
            return relevant_chunks
        except Exception as e:
            st.error(f"Error in semantic search: {e}")
            return []
    
    def get_llm_response(self, prompt: str, context: str = "") -> str:
        """Get response from LM Studio LLM"""
        try:
            # Construct the full prompt
            if context:
                full_prompt = f"""You are a helpful KKH nursing assistant. Use the following context to answer the question accurately and concisely.

Context: {context}

Question: {prompt}

Answer:"""
            else:
                full_prompt = f"""You are a helpful KKH nursing assistant. Please answer the following question:

Question: {prompt}

Answer:"""
            
            # Make API call to LM Studio
            payload = {
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                return f"Error: Unable to get response from LLM (Status: {response.status_code})"
                
        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to LM Studio. Please ensure it's running at http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_quiz_questions(self) -> List[Dict]:
        """Generate quiz questions from PDF content"""
        quiz_questions = [
            {
                "question": "What is the first step in pediatric CPR?",
                "options": ["Check for responsiveness", "Open airway", "Start chest compressions", "Call for help"],
                "correct": 0,
                "explanation": "Always check for responsiveness first to determine if CPR is needed."
            },
            {
                "question": "What is the normal heart rate range for a newborn?",
                "options": ["60-80 bpm", "80-100 bpm", "100-160 bpm", "160-200 bpm"],
                "correct": 2,
                "explanation": "Normal newborn heart rate is 100-160 beats per minute."
            },
            {
                "question": "Signs of dehydration in children include:",
                "options": ["Dry mouth only", "Sunken eyes and dry mucous membranes", "Normal skin elasticity", "Increased urination"],
                "correct": 1,
                "explanation": "Sunken eyes and dry mucous membranes are key signs of dehydration."
            },
            {
                "question": "What is the maintenance fluid requirement for a 10kg child?",
                "options": ["500ml/day", "750ml/day", "1000ml/day", "1500ml/day"],
                "correct": 2,
                "explanation": "For children 10kg or less: 100ml/kg/day. So 10kg × 100ml = 1000ml/day."
            },
            {
                "question": "During a seizure, the priority action is:",
                "options": ["Restrain the child", "Put something in their mouth", "Protect from injury and time the seizure", "Give medications immediately"],
                "correct": 2,
                "explanation": "Protect the child from injury and time the seizure duration."
            },
            {
                "question": "Normal respiratory rate for a 2-year-old is:",
                "options": ["12-20 breaths/min", "20-30 breaths/min", "30-40 breaths/min", "40-50 breaths/min"],
                "correct": 1,
                "explanation": "Normal respiratory rate for toddlers (1-3 years) is 20-30 breaths per minute."
            },
            {
                "question": "The compression-to-ventilation ratio for pediatric CPR (single rescuer) is:",
                "options": ["15:2", "30:2", "5:1", "3:1"],
                "correct": 1,
                "explanation": "For single rescuer pediatric CPR, the ratio is 30:2 (compressions:ventilations)."
            },
            {
                "question": "Fluid resuscitation for pediatric shock typically uses:",
                "options": ["5ml/kg bolus", "10ml/kg bolus", "20ml/kg bolus", "50ml/kg bolus"],
                "correct": 2,
                "explanation": "Standard fluid resuscitation bolus is 20ml/kg of isotonic crystalloid."
            },
            {
                "question": "Temperature indicating fever in children is:",
                "options": ["37°C", "37.5°C", "38°C", "39°C"],
                "correct": 2,
                "explanation": "Fever is defined as temperature ≥38°C (100.4°F)."
            },
            {
                "question": "Glasgow Coma Scale assesses:",
                "options": ["Only eye opening", "Eye opening, verbal response, motor response", "Only verbal response", "Heart rate and blood pressure"],
                "correct": 1,
                "explanation": "GCS assesses three components: eye opening, verbal response, and motor response."
            },
            {
                "question": "Normal blood pressure for a 5-year-old is approximately:",
                "options": ["70/40 mmHg", "85/55 mmHg", "95/65 mmHg", "110/70 mmHg"],
                "correct": 2,
                "explanation": "Normal systolic BP for children: 90 + (2 × age in years). For 5yo: 90 + 10 = 100 systolic."
            },
            {
                "question": "Capillary refill time should be less than:",
                "options": ["1 second", "2 seconds", "3 seconds", "5 seconds"],
                "correct": 1,
                "explanation": "Normal capillary refill time is less than 2 seconds."
            },
            {
                "question": "The most common cause of cardiac arrest in children is:",
                "options": ["Heart attack", "Respiratory failure", "Trauma", "Poisoning"],
                "correct": 1,
                "explanation": "Unlike adults, pediatric cardiac arrest is most commonly due to respiratory failure."
            },
            {
                "question": "Minimum urine output for children should be:",
                "options": ["0.5ml/kg/hr", "1ml/kg/hr", "2ml/kg/hr", "3ml/kg/hr"],
                "correct": 1,
                "explanation": "Minimum adequate urine output for children is 1ml/kg/hour."
            },
            {
                "question": "When should you call for emergency assistance during seizure?",
                "options": ["Immediately when seizure starts", "After 5 minutes", "Only if child stops breathing", "Never during seizure"],
                "correct": 1,
                "explanation": "Call for emergency help if seizure lasts longer than 5 minutes or if it's the child's first seizure."
            }
        ]
        return quiz_questions

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    chatbot = KKHChatbot()
    
    # Load PDF and create embeddings
    if os.path.exists(PDF_PATH):
        with st.spinner("Loading and processing PDF content..."):
            chunks = chatbot.load_pdf(PDF_PATH)
            if chunks:
                success = chatbot.create_embeddings(chunks)
                if success:
                    st.success("✅ PDF content loaded and indexed successfully!")
                else:
                    st.error("❌ Failed to create embeddings")
            else:
                st.error("❌ Failed to load PDF content")
    else:
        st.warning("⚠️ PDF file not found. Some features may be limited.")
    
    return chatbot

def calculate_fluid_requirements(weight: float, age: int, situation: str) -> Dict:
    """Calculate fluid requirements based on weight, age, and clinical situation"""
    results = {}
    
    # Maintenance fluid calculation (Holliday-Segar method)
    if weight <= 10:
        maintenance = weight * 100  # 100ml/kg/day for first 10kg
    elif weight <= 20:
        maintenance = 1000 + (weight - 10) * 50  # 1000ml + 50ml/kg for next 10kg
    else:
        maintenance = 1500 + (weight - 20) * 20  # 1500ml + 20ml/kg for each kg > 20
    
    results["maintenance_daily"] = maintenance
    results["maintenance_hourly"] = round(maintenance / 24, 1)
    
    # Situation-specific calculations
    if situation == "Dehydration (5%)":
        deficit = weight * 1000 * 0.05  # 5% of body weight
        results["deficit"] = deficit
        results["total_24h"] = maintenance + deficit
    elif situation == "Dehydration (10%)":
        deficit = weight * 1000 * 0.10  # 10% of body weight
        results["deficit"] = deficit
        results["total_24h"] = maintenance + deficit
    elif situation == "Shock/Resuscitation":
        bolus = weight * 20  # 20ml/kg bolus
        results["bolus"] = bolus
        results["total_24h"] = maintenance
    else:  # Maintenance only
        results["total_24h"] = maintenance
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 KKH Nursing Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = []
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = chatbot.generate_quiz_questions()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🏥 Navigation")
        feature = st.selectbox(
            "Choose a feature:",
            ["🔍 Semantic Search Chat", "📋 Fluid Calculator", "❓ Knowledge Quiz", "ℹ️ About"]
        )
    
    # Main content based on selected feature
    if feature == "🔍 Semantic Search Chat":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.header("🔍 Semantic Search Chatbot")
        st.write("Ask questions about KKH nursing procedures and get AI-powered answers based on our knowledge base.")
        
        # Quick prompt buttons
        st.subheader("💬 Quick Prompts")
        col1, col2, col3, col4 = st.columns(4)
        
        quick_prompts = [
            "Signs of dehydration",
            "Paediatric CPR steps", 
            "Seizure protocol",
            "Fluid resuscitation"
        ]
        
        with col1:
            if st.button(quick_prompts[0]):
                st.session_state.user_input = quick_prompts[0]
        with col2:
            if st.button(quick_prompts[1]):
                st.session_state.user_input = quick_prompts[1]
        with col3:
            if st.button(quick_prompts[2]):
                st.session_state.user_input = quick_prompts[2]
        with col4:
            if st.button(quick_prompts[3]):
                st.session_state.user_input = quick_prompts[3]
        
        # Chat interface
        user_input = st.text_input(
            "Ask your question:", 
            value=st.session_state.get('user_input', ''),
            key='chat_input'
        )
        
        if st.button("Send") and user_input:
            with st.spinner("Searching knowledge base and generating response..."):
                # Perform semantic search
                relevant_chunks = chatbot.semantic_search(user_input)
                context = "\n".join(relevant_chunks) if relevant_chunks else ""
                
                # Get LLM response
                response = chatbot.get_llm_response(user_input, context)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": response,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                
                # Clear input
                if 'user_input' in st.session_state:
                    del st.session_state.user_input
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 chats
                with st.container():
                    st.write(f"**👤 You ({chat['timestamp']}):** {chat['user']}")
                    st.write(f"**🤖 KKH Assistant:** {chat['bot']}")
                    st.divider()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif feature == "📋 Fluid Calculator":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.header("📋 Pediatric Fluid Calculator")
        st.write("Calculate fluid requirements for pediatric patients based on weight, age, and clinical situation.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg):", min_value=0.5, max_value=100.0, value=10.0, step=0.1)
            age = st.number_input("Age (years):", min_value=0, max_value=18, value=2)
        
        with col2:
            situation = st.selectbox(
                "Clinical Situation:",
                ["Maintenance", "Dehydration (5%)", "Dehydration (10%)", "Shock/Resuscitation"]
            )
        
        if st.button("Calculate Fluid Requirements"):
            results = calculate_fluid_requirements(weight, age, situation)
            
            st.markdown('<div class="fluid-calc-result">', unsafe_allow_html=True)
            st.subheader("📊 Calculation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Maintenance (24h)", f"{results['maintenance_daily']:.0f} ml")
                st.metric("Maintenance (hourly)", f"{results['maintenance_hourly']:.1f} ml/hr")
            
            with col2:
                if 'deficit' in results:
                    st.metric("Fluid Deficit", f"{results['deficit']:.0f} ml")
                if 'bolus' in results:
                    st.metric("Resuscitation Bolus", f"{results['bolus']:.0f} ml")
                st.metric("Total 24h Requirement", f"{results['total_24h']:.0f} ml")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional guidance
            st.info("💡 **Clinical Notes:**\n"
                   "- Monitor urine output (goal: >1ml/kg/hr)\n"
                   "- Reassess hydration status regularly\n"
                   "- Consider electrolyte replacement if indicated\n"
                   "- Adjust based on ongoing losses")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif feature == "❓ Knowledge Quiz":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.header("❓ KKH Nursing Knowledge Quiz")
        
        if not st.session_state.quiz_started:
            st.write("Test your knowledge with our comprehensive nursing quiz!")
            st.write(f"📝 **{len(st.session_state.quiz_questions)} questions** covering pediatric nursing essentials")
            
            if st.button("🚀 Start Quiz"):
                st.session_state.quiz_started = True
                st.session_state.current_question = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []
                st.rerun()
        
        else:
            # Quiz in progress
            questions = st.session_state.quiz_questions
            current_q = st.session_state.current_question
            
            if current_q < len(questions):
                question_data = questions[current_q]
                
                st.markdown('<div class="quiz-question">', unsafe_allow_html=True)
                st.subheader(f"Question {current_q + 1} of {len(questions)}")
                st.write(f"**{question_data['question']}**")
                
                # Answer options
                user_answer = st.radio(
                    "Select your answer:",
                    question_data['options'],
                    key=f"q_{current_q}"
                )
                
                if st.button("Submit Answer"):
                    correct_idx = question_data['correct']
                    user_idx = question_data['options'].index(user_answer)
                    is_correct = user_idx == correct_idx
                    
                    if is_correct:
                        st.session_state.quiz_score += 1
                        st.markdown(f'<div class="correct-answer">✅ Correct! {question_data["explanation"]}</div>', unsafe_allow_html=True)
                    else:
                        correct_answer = question_data['options'][correct_idx]
                        st.markdown(f'<div class="incorrect-answer">❌ Incorrect. The correct answer is: {correct_answer}<br>{question_data["explanation"]}</div>', unsafe_allow_html=True)
                    
                    st.session_state.quiz_answers.append({
                        'question': question_data['question'],
                        'user_answer': user_answer,
                        'correct_answer': question_data['options'][correct_idx],
                        'is_correct': is_correct
                    })
                    
                    time.sleep(2)  # Brief pause to show feedback
                    st.session_state.current_question += 1
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Progress bar
                progress = (current_q + 1) / len(questions)
                st.progress(progress)
            
            else:
                # Quiz completed
                st.subheader("🎉 Quiz Completed!")
                score_percentage = (st.session_state.quiz_score / len(questions)) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{st.session_state.quiz_score}/{len(questions)}")
                with col2:
                    st.metric("Percentage", f"{score_percentage:.1f}%")
                with col3:
                    if score_percentage >= 80:
                        st.success("🌟 Excellent!")
                    elif score_percentage >= 70:
                        st.info("👍 Good Job!")
                    else:
                        st.warning("📚 Keep Studying!")
                
                # Show review
                with st.expander("📋 Review Your Answers"):
                    for i, answer in enumerate(st.session_state.quiz_answers):
                        if answer['is_correct']:
                            st.success(f"Q{i+1}: ✅ {answer['question']}")
                        else:
                            st.error(f"Q{i+1}: ❌ {answer['question']}")
                            st.write(f"Your answer: {answer['user_answer']}")
                            st.write(f"Correct answer: {answer['correct_answer']}")
                
                if st.button("🔄 Restart Quiz"):
                    st.session_state.quiz_started = False
                    st.session_state.current_question = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = []
                    # Shuffle questions for variety
                    random.shuffle(st.session_state.quiz_questions)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif feature == "ℹ️ About":
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.header("ℹ️ About KKH Nursing Chatbot")
        
        st.write("""
        ### 🏥 Welcome to the KKH Nursing Chatbot
        
        This comprehensive nursing assistant is designed to support healthcare professionals with:
        
        **🔍 Semantic Search Chat**
        - AI-powered responses based on KKH medical guidelines
        - Natural language question processing
        - Context-aware answers using embedded PDF knowledge
        
        **📋 Fluid Calculator**
        - Holliday-Segar method for maintenance fluids
        - Dehydration deficit calculations
        - Shock resuscitation protocols
        
        **❓ Knowledge Quiz**
        - 15 comprehensive nursing questions
        - Immediate feedback and explanations
        - Score tracking and review
        
        ### 🔧 Technical Details
        - **LLM**: OpenHermes 2.5 Mistral 7B (via LM Studio)
        - **Embeddings**: Multilingual E5 Large Instruct
        - **Vector Search**: FAISS indexing
        - **Framework**: Streamlit
        
        ### 📞 Support
        For technical issues or content updates, please contact the development team.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
