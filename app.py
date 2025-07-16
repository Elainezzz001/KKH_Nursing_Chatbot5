import streamlit as st
import requests
import json
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import random
from typing import List, Dict, Tuple
import time

# Page configuration
st.set_page_config(
    page_title="KKH Nursing Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #E3F2FD;
        margin-left: 20%;
    }
    
    .bot-message {
        background-color: #F5F5F5;
        margin-right: 20%;
    }
    
    .quick-prompt-btn {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        background-color: #2E86AB;
        color: white;
        border: none;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    
    .fluid-calc-container {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .quiz-question {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .correct-answer {
        background-color: #E8F5E8;
        color: #2E7D32;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    
    .incorrect-answer {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_current' not in st.session_state:
    st.session_state.quiz_current = 0
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}

# LM Studio configuration
LM_STUDIO_URL = "http://192.168.75.1:1234/v1/chat/completions"

class KKHChatbot:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.text_chunks = []
        self.load_embeddings()
    
    @st.cache_resource
    def load_embedding_model(_self):
        """Load the sentence transformer model"""
        try:
            return SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return None
    
    def load_embeddings(self):
        """Load and process the PDF content into embeddings"""
        if st.session_state.embeddings_loaded:
            return
        
        with st.spinner("Loading KKH nursing content..."):
            try:
                # Load embedding model
                self.embedding_model = self.load_embedding_model()
                if not self.embedding_model:
                    return
                
                # Load PDF content
                pdf_path = "data/KKH Information file.pdf"
                if os.path.exists(pdf_path):
                    self.text_chunks = self.extract_pdf_content(pdf_path)
                    
                    if self.text_chunks:
                        # Create embeddings
                        embeddings = self.embedding_model.encode(self.text_chunks)
                        
                        # Build FAISS index
                        dimension = embeddings.shape[1]
                        self.faiss_index = faiss.IndexFlatIP(dimension)
                        self.faiss_index.add(embeddings.astype('float32'))
                        
                        st.session_state.embeddings_loaded = True
                        st.success("KKH nursing content loaded successfully!")
                    else:
                        st.error("No content extracted from PDF")
                else:
                    st.error("KKH Information file not found")
                    
            except Exception as e:
                st.error(f"Error loading embeddings: {e}")
    
    def extract_pdf_content(self, pdf_path: str) -> List[str]:
        """Extract text content from PDF and split into chunks"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Split into chunks (approximately 500 characters each)
            chunks = []
            chunk_size = 500
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) > 50:  # Only include substantial chunks
                    chunks.append(chunk.strip())
            
            return chunks
        except Exception as e:
            st.error(f"Error extracting PDF content: {e}")
            return []
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant context chunks for a query"""
        if not self.embedding_model or not self.faiss_index:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search similar chunks
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Return relevant chunks
            relevant_chunks = []
            for idx in indices[0]:
                if idx < len(self.text_chunks):
                    relevant_chunks.append(self.text_chunks[idx])
            
            return relevant_chunks
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return []
    
    def chat_with_lm_studio(self, message: str, context: List[str] = None) -> str:
        """Send chat request to LM Studio"""
        try:
            # Prepare context-enhanced prompt
            prompt = message
            if context:
                context_text = "\n\n".join(context)
                prompt = f"""Based on the following KKH nursing information, please answer the question:

Context:
{context_text}

Question: {message}

Please provide a helpful and accurate response based on the nursing information provided."""
            
            # Prepare request payload
            payload = {
                "model": "openhermes-2.5-mistral-7b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful nursing assistant specializing in KKH (KK Women's and Children's Hospital) procedures and pediatric care. Provide accurate, professional medical guidance."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # Send request to LM Studio
            response = requests.post(
                LM_STUDIO_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Unable to connect to LM Studio (Status: {response.status_code})"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to LM Studio. Please ensure LM Studio is running on http://localhost:1234"
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return KKHChatbot()

chatbot = get_chatbot()

# Fluid Calculator Functions
def calculate_maintenance_fluid(weight_kg: float, age_years: int) -> Dict:
    """Calculate maintenance fluid using Holliday-Segar method"""
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
        "description": "Maintenance fluid for normal daily needs"
    }

def calculate_dehydration_fluid(weight_kg: float, dehydration_percent: int) -> Dict:
    """Calculate fluid replacement for dehydration"""
    maintenance = calculate_maintenance_fluid(weight_kg, 0)
    deficit_ml = weight_kg * 1000 * (dehydration_percent / 100)
    
    # Replace deficit over 24 hours plus maintenance
    total_daily = maintenance["daily_ml"] + deficit_ml
    hourly_ml = total_daily / 24
    
    return {
        "daily_ml": round(total_daily),
        "hourly_ml": round(hourly_ml, 1),
        "deficit_ml": round(deficit_ml),
        "maintenance_ml": maintenance["daily_ml"],
        "description": f"Maintenance + {dehydration_percent}% dehydration replacement"
    }

def calculate_shock_fluid(weight_kg: float) -> Dict:
    """Calculate fluid bolus for shock/resuscitation"""
    bolus_ml = weight_kg * 20  # 20ml/kg bolus
    
    return {
        "bolus_ml": bolus_ml,
        "description": "Immediate bolus for shock/resuscitation (20ml/kg)",
        "note": "May repeat up to 3 times. Reassess after each bolus."
    }

# Quiz Questions Database
QUIZ_QUESTIONS = [
    {
        "question": "What is the first-line fluid for pediatric resuscitation?",
        "options": ["5% Dextrose", "Normal Saline (0.9%)", "Half Normal Saline (0.45%)", "Lactated Ringer's"],
        "correct": 1,
        "explanation": "Normal Saline (0.9%) is the first-line fluid for pediatric resuscitation as it provides rapid volume expansion."
    },
    {
        "question": "How much fluid bolus should be given for pediatric shock?",
        "options": ["10ml/kg", "20ml/kg", "30ml/kg", "40ml/kg"],
        "correct": 1,
        "explanation": "20ml/kg is the standard fluid bolus for pediatric shock, which can be repeated up to 3 times."
    },
    {
        "question": "What are early signs of dehydration in infants?",
        "options": ["Sunken fontanelle", "Decreased skin turgor", "Dry mucous membranes", "All of the above"],
        "correct": 3,
        "explanation": "All listed signs are early indicators of dehydration in infants and should prompt immediate assessment."
    },
    {
        "question": "At what heart rate should you be concerned in a 2-year-old child?",
        "options": [">120 bpm", ">140 bpm", ">160 bpm", ">180 bpm"],
        "correct": 3,
        "explanation": "Heart rate >160 bpm in a 2-year-old child is concerning and requires immediate evaluation."
    },
    {
        "question": "What is the compression to ventilation ratio for pediatric CPR (single rescuer)?",
        "options": ["15:2", "30:2", "5:1", "10:2"],
        "correct": 1,
        "explanation": "For single rescuer pediatric CPR, the ratio is 30:2 (30 compressions to 2 ventilations)."
    },
    {
        "question": "When should you refer a febrile child to a doctor immediately?",
        "options": ["Temperature >38°C", "Temperature >39°C", "Any fever in infant <3 months", "Fever lasting >24 hours"],
        "correct": 2,
        "explanation": "Any fever in an infant less than 3 months old requires immediate medical evaluation due to immature immune system."
    },
    {
        "question": "What is the normal respiratory rate for a 1-year-old child?",
        "options": ["12-20 breaths/min", "20-30 breaths/min", "30-40 breaths/min", "40-60 breaths/min"],
        "correct": 1,
        "explanation": "Normal respiratory rate for a 1-year-old is 20-30 breaths per minute."
    },
    {
        "question": "Which medication is contraindicated in children with Reye's syndrome risk?",
        "options": ["Paracetamol", "Ibuprofen", "Aspirin", "Codeine"],
        "correct": 2,
        "explanation": "Aspirin is contraindicated in children due to the risk of Reye's syndrome, especially during viral illnesses."
    },
    {
        "question": "What is the recommended depth of chest compressions for a child?",
        "options": ["1/3 of chest diameter", "1/2 of chest diameter", "2 inches", "1 inch"],
        "correct": 0,
        "explanation": "Chest compressions should be at least 1/3 of the chest diameter for effective circulation in children."
    },
    {
        "question": "At what weight should you use adult AED pads instead of pediatric pads?",
        "options": [">15 kg", ">20 kg", ">25 kg", ">30 kg"],
        "correct": 2,
        "explanation": "Adult AED pads should be used for children weighing more than 25 kg or older than 8 years."
    },
    {
        "question": "What is the most common cause of bradycardia in children?",
        "options": ["Heart block", "Hypothermia", "Hypoxia", "Medication"],
        "correct": 2,
        "explanation": "Hypoxia is the most common cause of bradycardia in children, making airway management priority."
    },
    {
        "question": "How often should vital signs be monitored in a critically ill child?",
        "options": ["Every 30 minutes", "Every hour", "Every 15 minutes", "Continuously"],
        "correct": 3,
        "explanation": "Critically ill children require continuous monitoring of vital signs for early detection of deterioration."
    },
    {
        "question": "What is the first intervention for a choking conscious child?",
        "options": ["Back blows", "Chest thrusts", "Abdominal thrusts", "Finger sweep"],
        "correct": 0,
        "explanation": "For a conscious choking child, start with 5 back blows between the shoulder blades."
    },
    {
        "question": "What glucose level indicates hypoglycemia in children?",
        "options": ["<4.0 mmol/L", "<3.5 mmol/L", "<3.0 mmol/L", "<2.5 mmol/L"],
        "correct": 2,
        "explanation": "Blood glucose <3.0 mmol/L indicates hypoglycemia in children and requires immediate treatment."
    },
    {
        "question": "What is the maximum time for attempting intubation in a child?",
        "options": ["20 seconds", "30 seconds", "45 seconds", "60 seconds"],
        "correct": 1,
        "explanation": "Intubation attempts should not exceed 30 seconds in children to prevent hypoxia."
    }
]

# Sidebar Navigation
st.sidebar.title("🏥 KKH Nursing Chatbot")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    ["💬 Chat", "💧 Fluid Calculator", "📋 Quiz", "ℹ️ About"]
)

# Main Content based on selected page
if page == "💬 Chat":
    st.markdown('<h1 class="main-header">💬 KKH Nursing Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Quick prompt buttons
    st.markdown("### Quick Prompts")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_prompts = [
        "Signs of dehydration",
        "Paediatric CPR steps", 
        "Fluid resuscitation",
        "When to refer to doctor"
    ]
    
    for i, prompt in enumerate(quick_prompts):
        with [col1, col2, col3, col4][i]:
            if st.button(prompt, key=f"quick_{i}"):
                # Add to chat history and process
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Get relevant context
                context = chatbot.retrieve_context(prompt)
                
                # Get response from LM Studio
                response = chatbot.chat_with_lm_studio(prompt, context)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                st.rerun()
    
    # Chat input
    user_input = st.text_input("Ask me anything about KKH nursing procedures:", key="chat_input")
    
    if st.button("Send") and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get relevant context
        with st.spinner("Searching KKH knowledge base..."):
            context = chatbot.retrieve_context(user_input)
        
        # Get response from LM Studio
        with st.spinner("Generating response..."):
            response = chatbot.chat_with_lm_studio(user_input, context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Display chat history
    st.markdown("### Conversation")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

elif page == "💧 Fluid Calculator":
    st.markdown('<h1 class="main-header">💧 Pediatric Fluid Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="fluid-calc-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input("Patient Weight (kg)", min_value=0.5, max_value=100.0, value=10.0, step=0.1)
        age = st.number_input("Patient Age (years)", min_value=0, max_value=18, value=2)
    
    with col2:
        situation = st.selectbox(
            "Clinical Situation",
            ["Maintenance", "Dehydration (5%)", "Dehydration (10%)", "Shock/Resuscitation"]
        )
    
    if st.button("Calculate Fluid Requirements"):
        if situation == "Maintenance":
            result = calculate_maintenance_fluid(weight, age)
            st.success(f"**{result['description']}**")
            st.write(f"- Daily fluid: **{result['daily_ml']} ml**")
            st.write(f"- Hourly rate: **{result['hourly_ml']} ml/hr**")
            
        elif "Dehydration" in situation:
            percent = 5 if "5%" in situation else 10
            result = calculate_dehydration_fluid(weight, percent)
            st.success(f"**{result['description']}**")
            st.write(f"- Maintenance: **{result['maintenance_ml']} ml/day**")
            st.write(f"- Deficit replacement: **{result['deficit_ml']} ml**")
            st.write(f"- Total daily fluid: **{result['daily_ml']} ml**")
            st.write(f"- Hourly rate: **{result['hourly_ml']} ml/hr**")
            
        elif "Shock" in situation:
            result = calculate_shock_fluid(weight)
            st.error(f"**{result['description']}**")
            st.write(f"- Immediate bolus: **{result['bolus_ml']} ml**")
            st.write(f"- {result['note']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fluid calculation reference
    st.markdown("### 📚 Reference Information")
    st.markdown("""
    **Holliday-Segar Method:**
    - First 10 kg: 100 ml/kg/day
    - Next 10 kg: 50 ml/kg/day  
    - Each additional kg: 20 ml/kg/day
    
    **Dehydration Assessment:**
    - Mild (5%): Slightly dry mucous membranes, decreased urine output
    - Moderate (10%): Dry mucous membranes, sunken eyes, decreased skin turgor
    - Severe (15%): All above plus sunken fontanelle, altered mental status
    
    **Shock Management:**
    - 20 ml/kg normal saline bolus
    - Reassess after each bolus
    - Maximum 3 boluses (60 ml/kg total)
    """)

elif page == "📋 Quiz":
    st.markdown('<h1 class="main-header">📋 KKH Nursing Knowledge Quiz</h1>', unsafe_allow_html=True)
    
    total_questions = len(QUIZ_QUESTIONS)
    
    # Quiz controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start New Quiz"):
            st.session_state.quiz_current = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = {}
            st.rerun()
    
    with col2:
        st.write(f"Question {st.session_state.quiz_current + 1} of {total_questions}")
    
    with col3:
        st.write(f"Score: {st.session_state.quiz_score}/{len(st.session_state.quiz_answers)}")
    
    if st.session_state.quiz_current < total_questions:
        current_q = QUIZ_QUESTIONS[st.session_state.quiz_current]
        
        st.markdown(f'<div class="quiz-question">', unsafe_allow_html=True)
        st.markdown(f"**Question {st.session_state.quiz_current + 1}:** {current_q['question']}")
        
        # Show options
        user_answer = st.radio(
            "Select your answer:",
            current_q['options'],
            key=f"quiz_{st.session_state.quiz_current}"
        )
        
        if st.button("Submit Answer"):
            selected_index = current_q['options'].index(user_answer)
            is_correct = selected_index == current_q['correct']
            
            # Store answer
            st.session_state.quiz_answers[st.session_state.quiz_current] = {
                'selected': selected_index,
                'correct': is_correct,
                'explanation': current_q['explanation']
            }
            
            if is_correct:
                st.session_state.quiz_score += 1
                st.markdown('<div class="correct-answer">✅ Correct!</div>', unsafe_allow_html=True)
            else:
                correct_answer = current_q['options'][current_q['correct']]
                st.markdown(f'<div class="incorrect-answer">❌ Incorrect. The correct answer is: {correct_answer}</div>', unsafe_allow_html=True)
            
            st.write(f"**Explanation:** {current_q['explanation']}")
            
            # Move to next question
            st.session_state.quiz_current += 1
            time.sleep(2)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Quiz completed
        st.success("🎉 Quiz Completed!")
        final_score = st.session_state.quiz_score
        percentage = (final_score / total_questions) * 100
        
        st.markdown(f"### Final Score: {final_score}/{total_questions} ({percentage:.1f}%)")
        
        if percentage >= 80:
            st.balloons()
            st.success("Excellent work! You have a strong understanding of KKH nursing procedures.")
        elif percentage >= 60:
            st.warning("Good job! Consider reviewing the areas you missed.")
        else:
            st.error("You may want to review the KKH nursing materials and try again.")
        
        # Show review of answers
        if st.button("Review Answers"):
            for i, (q_idx, answer_data) in enumerate(st.session_state.quiz_answers.items()):
                question = QUIZ_QUESTIONS[q_idx]
                st.write(f"**Q{q_idx + 1}:** {question['question']}")
                
                if answer_data['correct']:
                    st.success(f"✅ Your answer: {question['options'][answer_data['selected']]}")
                else:
                    st.error(f"❌ Your answer: {question['options'][answer_data['selected']]}")
                    st.info(f"✅ Correct answer: {question['options'][question['correct']]}")
                
                st.write(f"*{answer_data['explanation']}*")
                st.markdown("---")

elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About KKH Nursing Chatbot</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏥 KKH Nursing Chatbot
    
    This application is designed to assist nursing staff at KK Women's and Children's Hospital (KKH) 
    with quick access to pediatric nursing procedures, calculations, and knowledge verification.
    
    ### 🔧 Features:
    
    #### 💬 **Intelligent Chat Assistant**
    - Powered by LM Studio with openhermes-2.5-mistral-7b model
    - Context-aware responses using KKH-specific nursing documentation
    - Quick access to common procedures and protocols
    
    #### 💧 **Pediatric Fluid Calculator**
    - Holliday-Segar method for maintenance fluids
    - Dehydration replacement calculations (5% and 10%)
    - Shock/resuscitation fluid bolus calculations
    - Age and weight-based recommendations
    
    #### 📋 **Knowledge Assessment Quiz**
    - 15 evidence-based multiple choice questions
    - Immediate feedback with explanations
    - Score tracking and performance review
    - Topics covering pediatric emergencies, procedures, and protocols
    
    ### 🤖 **Technology Stack:**
    - **Frontend:** Streamlit with custom CSS
    - **AI Model:** LM Studio (openhermes-2.5-mistral-7b)
    - **Embeddings:** intfloat/multilingual-e5-large-instruct
    - **Vector Search:** FAISS
    - **PDF Processing:** PyPDF2
    - **Deployment:** Docker + Fly.io
    
    ### 🔒 **Privacy & Security:**
    - All processing happens locally - no external API calls
    - Patient data remains within your infrastructure
    - Offline-capable design for reliable access
    
    ### ⚠️ **Important Disclaimer:**
    This tool is designed to assist healthcare professionals and should not replace clinical judgment, 
    hospital protocols, or direct physician consultation. Always follow your institution's guidelines 
    and seek appropriate medical supervision when needed.
    
    ### 📞 **Support:**
    For technical issues or content updates, please contact your IT department or system administrator.
    
    ---
    *Version 1.0 | Built for KKH Nursing Staff*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "KKH Nursing Chatbot © 2025 | For KKH Staff Use Only"
    "</div>",
    unsafe_allow_html=True
)
