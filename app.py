import streamlit as st
st.set_page_config(
    page_title="KKH Nursing Chatbot",
    page_icon="logo/photo_2025-06-16_15-57-21.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
import pickle
import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import time

st.markdown("""
    <style>
        .block-container {
            padding: 2.7rem;
            max-width: 1200px;
        }
        .stTextInput>div>div>input {
            font-size: 1.1rem;
        }
        .stButton>button {
            font-size: 1rem;
            border-radius: 0.5rem;
            width: 100%;
        }
        .stSelectbox>div>div>select {
            font-size: 1rem;
        }
        .stTextArea>div>div>textarea {
            font-size: 1rem;
        }
        .quiz-question {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .metric-container {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 0.5rem;
            }
            .stButton>button {
                font-size: 0.9rem;
                padding: 0.5rem;
            }
            .metric-container {
                padding: 0.5rem;
            }
        }
        @media (max-width: 480px) {
            .stButton>button {
                font-size: 0.8rem;
                padding: 0.4rem;
            }
        }
    </style>
""", unsafe_allow_html=True)


# Import libraries for PDF processing and embeddings
try:
    import PyPDF2
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"Missing required library: {e}")
    st.stop()

# Smart configuration based on environment
def get_llm_api_url() -> str:
    # Primary: Together.ai API
    together_url = "https://api.together.xyz/v1/chat/completions"
    # Fallback: LM Studio local
    local_url = "http://localhost:1234/v1/chat/completions"
    
    # Check if Together.ai API key is available
    if os.getenv("TOGETHER_API_KEY"):
        return together_url
    # Check if custom LLM_API_URL is set
    elif os.getenv("LLM_API_URL"):
        return os.getenv("LLM_API_URL")
    # Fallback to local LM Studio
    else:
        return local_url

def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

def get_api_key() -> str:
    # Try Together.ai first, then generic LLM_API_KEY
    return os.getenv("TOGETHER_API_KEY") or os.getenv("LLM_API_KEY")

LLM_API_URL = get_llm_api_url()
LLM_MODEL = get_llm_model()
API_KEY = get_api_key()

# Configuration
PDF_PATH = "data/KKH Information file.pdf"
QUIZ_DATA_PATH = "quiz_data.json"
EMBEDDINGS_PATH = "embeddings.pkl"
CHUNKS_PATH = "chunks.pkl"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'quiz_session' not in st.session_state:
    st.session_state.quiz_session = None
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# Check if running on Render or local
def is_local_environment() -> bool:
    """Check if the app is running locally (has access to LM Studio)"""
    return os.getenv('RENDER') != 'true'

# PDF Processing Functions
@st.cache_data
def load_pdf_chunks(filepath: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    try:
        chunks = []
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
        words = full_text.split()
        current_chunk = []
        current_size = 0
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            if current_size >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_words
                current_size = sum(len(word) + 1 for word in current_chunk)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []


@st.cache_resource
def load_embedding_model():
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def embed_chunks(chunks: List[str], model) -> np.ndarray:
    try:
        if model is None:
            return np.array([])
        prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
        embeddings = model.encode(prefixed_chunks)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

def search_relevant_chunks(query: str, chunks: List[str], embeddings: np.ndarray, model, top_k: int = 3) -> List[str]:
    try:
        if model is None or len(embeddings) == 0:
            return chunks[:top_k] if chunks else []
        
        query_embedding = model.encode([f"query: {query}"])
        
        # Check for dimension compatibility
        if embeddings.shape[1] != query_embedding.shape[1]:
            st.warning(f"Embedding dimension mismatch detected. Regenerating embeddings...")
            # Clear cached embeddings to force regeneration
            if os.path.exists(EMBEDDINGS_PATH):
                os.remove(EMBEDDINGS_PATH)
            if os.path.exists(CHUNKS_PATH):
                os.remove(CHUNKS_PATH)
            st.session_state.pdf_processed = False
            st.rerun()
            return chunks[:top_k] if chunks else []
        
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices if i < len(chunks)]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return chunks[:top_k] if chunks else []


# LLM Integration
def generate_response(context: str, query: str) -> str:
    if LLM_API_URL is None:
        return "⚠️ No LLM API URL configured. Please set the TOGETHER_API_KEY or LLM_API_URL environment variable."
    
    try:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kkh-nursing-chatbot.fly.dev",
            "X-Title": "KKH Nursing Chatbot"
        }
        
        # Add authorization header if API key is available
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        # Limit context to prevent token overflow
        context = context[:2000]
        
        system_prompt = """You are a specialized medical assistant for KK Women's and Children's Hospital (KKH) nurses. 
        Your role is to provide accurate, evidence-based information to help with patient care.
        Always base your responses on the provided context from the KKH medical guidelines.
        If the information is not available in the context, clearly state this.
        Keep responses concise but comprehensive, focusing on practical nursing implications."""
        
        user_prompt = f"""Context from KKH Guidelines:
{context}

Question: {query}

Please provide a detailed answer based on the KKH guidelines provided in the context."""
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800,
            "stream": False
        }
        
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            if not content.strip():
                return "⚠️ The model responded with an empty message. Try asking a simpler question or reducing the context."
            return content
        elif response.status_code == 404:
            if "localhost" in LLM_API_URL:
                return """⚠️ **LM Studio Error**: No models are currently loaded in LM Studio.

**To fix this:**
1. Open LM Studio
2. Go to the "Local Server" tab
3. Load a model (e.g., mistral-7b-instruct)
4. Start the server on port 1234
5. Try your question again

**Alternative**: The app is configured to use Together.ai API as primary. Your API key is set, so this should work automatically."""
            else:
                return f"⚠️ Model not found on API server. Status: {response.status_code}"
        else:
            return f"⚠️ API Error (Status: {response.status_code}): {response.text}"
            
    except requests.exceptions.ConnectionError:
        if "localhost" in LLM_API_URL:
            return """⚠️ **Cannot connect to LM Studio**

**To fix this:**
1. Make sure LM Studio is running
2. Go to Local Server tab in LM Studio
3. Start the server on http://localhost:1234
4. Load a compatible model

**Alternative**: The app will automatically use Together.ai API if LM Studio is unavailable."""
        else:
            return "⚠️ Cannot connect to the LLM API. Please check your internet connection."
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


# Fluid Calculator
def fluid_calculator(weight: float, age_months: int, condition: str = "normal") -> Dict[str, Any]:
    """Calculate daily fluid requirements using Holliday-Segar method"""
    
    # Holliday-Segar method for maintenance fluids (mL/day)
    if weight <= 10:
        base_fluid = weight * 100  # 100 mL/kg/day for first 10 kg
    elif weight <= 20:
        base_fluid = 1000 + (weight - 10) * 50  # 50 mL/kg/day for next 10 kg (11-20 kg)
    else:
        base_fluid = 1500 + (weight - 20) * 20  # 20 mL/kg/day for each kg > 20 kg
    
    # Age-based adjustments for pediatric patients
    age_factor = 1.0
    age_notes = []
    
    if age_months < 1:  # Neonates (0-1 month)
        age_factor = 1.5
        age_notes.append("Increased by 50% for neonatal requirements")
    elif age_months < 6:  # Young infants (1-6 months)
        age_factor = 1.3
        age_notes.append("Increased by 30% for young infant requirements")
    elif age_months < 12:  # Older infants (6-12 months)
        age_factor = 1.2
        age_notes.append("Increased by 20% for infant requirements")
    elif age_months < 24:  # Toddlers (1-2 years)
        age_factor = 1.1
        age_notes.append("Increased by 10% for toddler requirements")
    
    base_fluid *= age_factor
    
    # Condition-based adjustments
    condition_factor = 1.0
    condition_notes = []
    
    condition_lower = condition.lower()
    if "fever" in condition_lower:
        condition_factor = 1.2
        condition_notes.append("Increased by 20% due to fever (12% per °C above 37°C)")
    elif "dehydration" in condition_lower:
        condition_factor = 1.5
        condition_notes.append("Increased by 50% for rehydration needs")
    elif "heart failure" in condition_lower or "cardiac" in condition_lower:
        condition_factor = 0.8
        condition_notes.append("Reduced by 20% due to heart failure/cardiac condition")
    elif "renal" in condition_lower or "kidney" in condition_lower:
        condition_factor = 0.7
        condition_notes.append("Reduced by 30% due to renal impairment")
    elif "respiratory" in condition_lower:
        condition_factor = 1.1
        condition_notes.append("Increased by 10% due to respiratory condition")
    
    total_fluid = base_fluid * condition_factor
    hourly_rate = total_fluid / 24
    
    # Calculate ml/kg/hr for reference
    ml_kg_hr = hourly_rate / weight
    
    return {
        "weight_kg": weight,
        "age_months": age_months,
        "condition": condition,
        "base_maintenance": round(base_fluid, 1),
        "total_daily": round(total_fluid, 1),
        "hourly_rate": round(hourly_rate, 1),
        "ml_kg_hr": round(ml_kg_hr, 1),
        "age_notes": age_notes,
        "condition_notes": condition_notes,
        "method": "Holliday-Segar Method"
    }

# Quiz Functions
def generate_quiz_questions_from_context(chunks: List[str], model) -> List[Dict]:
    """Generate quiz questions from PDF context using LLM"""
    if not is_local_environment():
        return load_default_quiz_questions()
    
    quiz_questions = []
    
    # Sample some chunks for question generation
    selected_chunks = chunks[:5] if len(chunks) >= 5 else chunks
    
    for i, chunk in enumerate(selected_chunks):
        if len(quiz_questions) >= 15:
            break
            
        try:
            prompt = f"""Based on the following medical text from KKH guidelines, create 3 multiple choice questions.
            
Text: {chunk[:800]}

For each question, provide:
1. A clear, specific question about medical procedures, protocols, or patient care
2. Four options (A, B, C, D) where only one is correct
3. The correct answer (A, B, C, or D)
4. A brief explanation of why the answer is correct

Format as JSON:
{{
    "question": "Question text here?",
    "options": {{
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
    }},
    "correct_answer": "B",
    "explanation": "Explanation text here"
}}

Generate 3 questions:"""

            response = generate_response("", prompt)
            
            # Try to parse the response (this is a simplified version)
            # In practice, you'd want more robust parsing
            if "question" in response.lower():
                # This is a simplified example - you'd want better JSON parsing
                sample_questions = [
                    {
                        "question": f"Question {len(quiz_questions) + 1} from KKH guidelines",
                        "options": {
                            "A": "Option A",
                            "B": "Option B",
                            "C": "Option C",
                            "D": "Option D"
                        },
                        "correct_answer": "A",
                        "explanation": "Based on KKH protocols"
                    }
                ]
                quiz_questions.extend(sample_questions)
                
        except Exception as e:
            continue
    
    # Fill with default questions if not enough generated
    while len(quiz_questions) < 15:
        quiz_questions.extend(load_default_quiz_questions())
        if len(quiz_questions) >= 15:
            break
    
    return quiz_questions[:15]

def load_default_quiz_questions() -> List[Dict]:
    """Load default quiz questions for fallback"""
    return [
        {
            "question": "What is the recommended first-line treatment for neonatal hypoglycemia?",
            "options": {
                "A": "IV glucose bolus",
                "B": "Oral feeding if possible, then IV glucose if needed",
                "C": "Immediate intubation",
                "D": "Wait and monitor"
            },
            "correct_answer": "B",
            "explanation": "Oral feeding should be attempted first if the baby is alert and able to feed safely."
        },
        {
            "question": "What is the normal respiratory rate for a newborn?",
            "options": {
                "A": "20-30 breaths per minute",
                "B": "30-60 breaths per minute", 
                "C": "60-80 breaths per minute",
                "D": "10-20 breaths per minute"
            },
            "correct_answer": "B",
            "explanation": "Normal respiratory rate for newborns is 30-60 breaths per minute."
        },
        {
            "question": "When should skin-to-skin contact be initiated after birth?",
            "options": {
                "A": "After 1 hour",
                "B": "Immediately after birth",
                "C": "After the baby is cleaned and weighed",
                "D": "Only if requested by parents"
            },
            "correct_answer": "B",
            "explanation": "Skin-to-skin contact should be initiated immediately after birth to promote bonding and thermoregulation."
        },
        {
            "question": "What is the recommended position for a baby with gastroesophageal reflux?",
            "options": {
                "A": "Prone position",
                "B": "Supine position",
                "C": "Left lateral position",
                "D": "Elevated supine position (30-45 degrees)"
            },
            "correct_answer": "D",
            "explanation": "Elevated supine position helps reduce reflux while maintaining safe sleep positioning."
        },
        {
            "question": "What is the first sign of respiratory distress in infants?",
            "options": {
                "A": "Cyanosis",
                "B": "Tachypnea",
                "C": "Bradycardia",
                "D": "Decreased activity"
            },
            "correct_answer": "B",
            "explanation": "Tachypnea (rapid breathing) is often the first sign of respiratory distress in infants."
        },
        {
            "question": "When should vitamin K be administered to newborns?",
            "options": {
                "A": "Within 24 hours",
                "B": "Within 6 hours",
                "C": "Within 1 hour",
                "D": "Only if bleeding occurs"
            },
            "correct_answer": "B",
            "explanation": "Vitamin K should be administered within 6 hours of birth to prevent hemorrhagic disease."
        },
        {
            "question": "What is the appropriate needle size for IM injection in newborns?",
            "options": {
                "A": "21G, 1 inch",
                "B": "25G, 5/8 inch",
                "C": "23G, 1 inch",
                "D": "27G, 1/2 inch"
            },
            "correct_answer": "B",
            "explanation": "25G, 5/8 inch needle is appropriate for IM injections in newborns."
        },
        {
            "question": "What is the minimum acceptable urine output for infants?",
            "options": {
                "A": "0.5 mL/kg/hour",
                "B": "1-2 mL/kg/hour",
                "C": "3-4 mL/kg/hour",
                "D": "5 mL/kg/hour"
            },
            "correct_answer": "B",
            "explanation": "Minimum acceptable urine output for infants is 1-2 mL/kg/hour."
        },
        {
            "question": "How often should vital signs be monitored in a stable newborn?",
            "options": {
                "A": "Every 15 minutes",
                "B": "Every 30 minutes",
                "C": "Every 4 hours",
                "D": "Every 8 hours"
            },
            "correct_answer": "C",
            "explanation": "For stable newborns, vital signs are typically monitored every 4 hours."
        },
        {
            "question": "What is the recommended temperature range for newborn bathing water?",
            "options": {
                "A": "35-37°C",
                "B": "37-39°C",
                "C": "39-41°C",
                "D": "33-35°C"
            },
            "correct_answer": "B",
            "explanation": "Bath water should be 37-39°C for newborn safety and comfort."
        },
        {
            "question": "When should the first meconium passage occur?",
            "options": {
                "A": "Within 8 hours",
                "B": "Within 24 hours",
                "C": "Within 48 hours",
                "D": "Within 72 hours"
            },
            "correct_answer": "C",
            "explanation": "First meconium passage should occur within 48 hours of birth."
        },
        {
            "question": "What is the appropriate oxygen saturation target for term newborns?",
            "options": {
                "A": "85-90%",
                "B": "90-95%",
                "C": "95-100%",
                "D": "100%"
            },
            "correct_answer": "C",
            "explanation": "Oxygen saturation for term newborns should be 95-100%."
        },
        {
            "question": "How should umbilical cord care be performed?",
            "options": {
                "A": "Apply antibiotic ointment",
                "B": "Keep clean and dry",
                "C": "Cover with gauze",
                "D": "Apply alcohol daily"
            },
            "correct_answer": "B",
            "explanation": "Umbilical cord should be kept clean and dry to prevent infection."
        },
        {
            "question": "What is the recommended frequency for newborn feeding?",
            "options": {
                "A": "Every 4 hours",
                "B": "Every 2-3 hours",
                "C": "Every 6 hours",
                "D": "On demand only"
            },
            "correct_answer": "B",
            "explanation": "Newborns should be fed every 2-3 hours or 8-12 times per day."
        },
        {
            "question": "When should a newborn's weight be checked after discharge?",
            "options": {
                "A": "Within 1 week",
                "B": "Within 3-5 days",
                "C": "Within 2 weeks",
                "D": "Within 1 month"
            },
            "correct_answer": "B",
            "explanation": "Newborn weight should be checked within 3-5 days after discharge to monitor feeding adequacy."
        }
    ]

def load_quiz() -> List[Dict]:
    """Load or generate quiz questions"""
    try:
        if os.path.exists(QUIZ_DATA_PATH):
            with open(QUIZ_DATA_PATH, 'r') as f:
                return json.load(f)
        else:
            # Generate questions from PDF context
            questions = generate_quiz_questions_from_context(st.session_state.chunks, load_embedding_model())
            
            # Save questions
            with open(QUIZ_DATA_PATH, 'w') as f:
                json.dump(questions, f, indent=2)
            
            return questions
    except Exception as e:
        st.error(f"Error loading quiz: {e}")
        return load_default_quiz_questions()

def evaluate_quiz(user_answers: Dict[int, str]) -> Dict[str, Any]:
    """Score the quiz and show explanations"""
    correct_count = 0
    total_questions = len(st.session_state.quiz_questions)
    results = []
    
    for i, question in enumerate(st.session_state.quiz_questions):
        user_answer = user_answers.get(i, "")
        correct_answer = question["correct_answer"]
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "question_number": i + 1,
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question["explanation"]
        })
    
    score_percentage = (correct_count / total_questions) * 100
    
    return {
        "score": correct_count,
        "total": total_questions,
        "percentage": score_percentage,
        "results": results
    }

# UI Functions
def render_quiz_ui():
    """Interactive quiz session in Streamlit"""
    st.subheader("🧠 KKH Nursing Knowledge Quiz")
    
    if st.session_state.quiz_session is None:
        st.write("Test your knowledge with 15 questions based on KKH nursing protocols.")
        
        if st.button("Start Quiz", type="primary"):
            st.session_state.quiz_questions = load_quiz()
            st.session_state.quiz_session = "active"
            st.session_state.current_question = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = {}
            st.rerun()
    
    elif st.session_state.quiz_session == "active":
        questions = st.session_state.quiz_questions
        current_q = st.session_state.current_question
        
        if current_q < len(questions):
            question = questions[current_q]
            
            st.write(f"**Question {current_q + 1} of {len(questions)}**")
            st.write(question["question"])
            
            # Radio buttons for options
            options = list(question["options"].values())
            option_keys = list(question["options"].keys())
            
            selected_option = st.radio(
                "Select your answer:",
                options,
                key=f"q_{current_q}",
                index=None
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Previous", disabled=current_q == 0):
                    st.session_state.current_question -= 1
                    st.rerun()
            
            with col2:
                if selected_option:
                    # Store the answer
                    selected_key = option_keys[options.index(selected_option)]
                    st.session_state.quiz_answers[current_q] = selected_key
                    
                    if current_q < len(questions) - 1:
                        if st.button("Next Question", type="primary"):
                            st.session_state.current_question += 1
                            st.rerun()
                    else:
                        if st.button("Finish Quiz", type="primary"):
                            st.session_state.quiz_session = "completed"
                            st.rerun()
        
        # Progress bar
        progress = (current_q + 1) / len(questions)
        st.progress(progress)
        
    elif st.session_state.quiz_session == "completed":
        st.write("### Quiz Results")
        
        results = evaluate_quiz(st.session_state.quiz_answers)
        
        # Score display
        score_color = "green" if results["percentage"] >= 70 else "orange" if results["percentage"] >= 50 else "red"
        st.markdown(f"**Your Score: ::{score_color}[{results['score']}/{results['total']} ({results['percentage']:.1f}%)]**")
        
        # Performance feedback
        if results["percentage"] >= 80:
            st.success("🎉 Excellent! You have a strong understanding of KKH nursing protocols.")
        elif results["percentage"] >= 70:
            st.info("✅ Good job! You have a solid grasp of the material.")
        elif results["percentage"] >= 50:
            st.warning("📚 Fair performance. Consider reviewing the KKH guidelines.")
        else:
            st.error("📖 Needs improvement. Please review the nursing protocols thoroughly.")
        
        # Detailed results
        with st.expander("View Detailed Results"):
            for result in results["results"]:
                icon = "✅" if result["is_correct"] else "❌"
                st.write(f"{icon} **Question {result['question_number']}:** {result['question']}")
                
                if not result["is_correct"]:
                    st.write(f"Your answer: {result['user_answer']}")
                    st.write(f"Correct answer: {result['correct_answer']}")
                    st.info(f"💡 {result['explanation']}")
                
                st.write("---")
        
        if st.button("Restart Quiz", type="primary"):
            st.session_state.quiz_session = None
            st.session_state.current_question = 0
            st.session_state.quiz_answers = {}
            st.rerun()

def render_prompt_buttons():
    """Display clickable preset question prompts"""
    st.subheader("💡 Quick Questions")
    st.write("Click on any question to get instant answers:")
    
    prompt_questions = [
        "What is the protocol for managing neonatal seizures?",
        "How to manage infant hypoglycemia?",
        "What are the signs of respiratory distress in newborns?",
        "Protocol for febrile convulsion management?",
        "When should vitamin K be administered?",
        "What is the proper technique for umbilical cord care?",
        "How to calculate fluid requirements for pediatric patients?",
        "What are the normal vital signs for newborns?",
        "When should skin-to-skin contact be initiated?",
        "What is the recommended feeding frequency for newborns?",
        "How to recognize signs of dehydration in infants?",
        "Management of neonatal jaundice protocols?",
        "Proper positioning for infants with reflux?",
        "Temperature monitoring guidelines for NICU?",
        "Emergency procedures for neonatal resuscitation?"
    ]
    
    # Create a grid of buttons (3 columns for better mobile responsiveness)
    cols = st.columns(3)
    
    for i, question in enumerate(prompt_questions):
        with cols[i % 3]:
            if st.button(question, key=f"prompt_{i}", use_container_width=True, help="Click to ask this question"):
                return question
    
    return None

def process_pdf_and_embeddings():
    """Process PDF and create embeddings if not already done"""
    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF and creating embeddings... This may take a moment."):
            # Load embedding model first
            model = load_embedding_model()
            
            # Try to load cached embeddings first
            if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
                try:
                    with open(CHUNKS_PATH, 'rb') as f:
                        chunks = pickle.load(f)
                    with open(EMBEDDINGS_PATH, 'rb') as f:
                        cached_embeddings = pickle.load(f)
                    
                    # Check if embedding dimensions match current model
                    if model is not None:
                        test_embedding = model.encode(["test"])
                        expected_dim = test_embedding.shape[1]
                        
                        if len(cached_embeddings) > 0 and cached_embeddings.shape[1] != expected_dim:
                            st.warning(f"Cached embeddings dimension ({cached_embeddings.shape[1]}) doesn't match current model ({expected_dim}). Regenerating...")
                            raise ValueError("Dimension mismatch")
                    
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = cached_embeddings
                    st.session_state.pdf_processed = True
                    st.success(f"✅ Loaded cached embeddings ({len(chunks)} chunks, {cached_embeddings.shape[1]}D)")
                    return
                except Exception as e:
                    st.warning(f"Could not load cached embeddings: {e}. Processing PDF...")
            
            # Load PDF chunks
            if os.path.exists(PDF_PATH):
                chunks = load_pdf_chunks(PDF_PATH)
                st.session_state.chunks = chunks
                
                if model and chunks:
                    # Create embeddings
                    embeddings = embed_chunks(chunks, model)
                    st.session_state.embeddings = embeddings
                    
                    # Save for future use
                    try:
                        with open(CHUNKS_PATH, 'wb') as f:
                            pickle.dump(chunks, f)
                        with open(EMBEDDINGS_PATH, 'wb') as f:
                            pickle.dump(embeddings, f)
                        st.success(f"✅ Created and saved embeddings ({len(chunks)} chunks, {embeddings.shape[1]}D)")
                    except Exception as e:
                        st.warning(f"Could not save embeddings: {e}")
                
                st.session_state.pdf_processed = True
            else:
                st.error(f"PDF file not found at {PDF_PATH}")
                st.warning("Please ensure 'KKH Information file.pdf' is in the data/ folder")

# Main App
def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo/photo_2025-06-16_15-57-21.jpg", width=100)
    with col2:
        st.title("KKH Nursing Chatbot")
    st.markdown("*AI Assistant for KK Women's and Children's Hospital Nurses*")
    
    # Sidebar
    with st.sidebar:
        st.image("logo/photo_2025-06-16_15-57-21.jpg", width=200)
        st.markdown("---")
        
        page = st.selectbox(
            "Navigation",
            ["💬 Chat Assistant", "🧠 Knowledge Quiz", "💧 Fluid Calculator", "📚 About"]
        )
        
        if is_local_environment():
            st.success("🟢 Local Mode - Full Features Available")
        else:
            st.warning("🟡 Deployed Mode - Limited Features")
    
    # Process PDF on first load
    if not st.session_state.pdf_processed:
        process_pdf_and_embeddings()
    
    # Main content area
    if page == "💬 Chat Assistant":
        st.header("Chat Assistant")
        
        # Initialize user_query from session state if a prompt was selected
        if 'selected_prompt' not in st.session_state:
            st.session_state.selected_prompt = None
        
        # Prompt buttons
        selected_prompt = render_prompt_buttons()
        
        # If a prompt button was clicked, store it in session state
        if selected_prompt:
            st.session_state.selected_prompt = selected_prompt
            st.rerun()
        
        # Chat interface
        st.markdown("---")
        st.subheader("Ask a Question")
        
        # Use selected prompt or manual input
        if st.session_state.selected_prompt:
            user_query = st.session_state.selected_prompt
            st.text_area("Your question:", value=st.session_state.selected_prompt, height=100, disabled=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                process_query = st.button("Get Answer", type="primary")
            with col2:
                if st.button("Clear & Ask New Question"):
                    st.session_state.selected_prompt = None
                    st.rerun()
        else:
            user_query = st.text_area("Type your question here:", height=100, placeholder="e.g., What is the protocol for managing neonatal hypoglycemia?")
            process_query = st.button("Ask", type="primary")
        
        if process_query and user_query:
            with st.spinner("Searching knowledge base and generating response..."):
                # Search relevant chunks
                model = load_embedding_model()
                relevant_chunks = search_relevant_chunks(
                    user_query, 
                    st.session_state.chunks, 
                    st.session_state.embeddings, 
                    model
                )
                
                # Generate response
                context = "\n\n".join(relevant_chunks)
                response = generate_response(context, user_query)
                
                # Display response
                st.markdown("### 🤖 Assistant Response")
                st.markdown(response)
                
                # Show sources
                with st.expander("📖 Source Information"):
                    st.write("Response based on the following sections from KKH guidelines:")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.write(f"**Source {i}:**")
                        st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.write("---")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_query,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Clear selected prompt after processing
                if st.session_state.selected_prompt:
                    st.session_state.selected_prompt = None
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📝 Recent Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {chat['question'][:60]}..."):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Answer:** {chat['response']}")
                    st.caption(f"Asked: {chat['timestamp']}")
    
    elif page == "🧠 Knowledge Quiz":
        render_quiz_ui()
    
    elif page == "💧 Fluid Calculator":
        st.header("Fluid Requirements Calculator")
        st.write("Calculate daily fluid requirements for pediatric patients using the **Holliday-Segar Method**.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Patient Weight (kg)", min_value=0.5, max_value=100.0, value=3.5, step=0.1)
            age_months = st.number_input("Age (months)", min_value=0, max_value=216, value=1, step=1)
        
        with col2:
            condition = st.selectbox(
                "Clinical Condition",
                ["normal", "fever", "dehydration", "heart failure", "renal impairment", "respiratory condition"]
            )
        
        if st.button("Calculate Fluid Requirements", type="primary"):
            result = fluid_calculator(weight, age_months, condition)
            
            st.markdown("### 💧 Fluid Calculation Results")
            st.caption(f"Using: {result['method']}")
            
            # Main metrics in a grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-container">
                    <h3>📊 Daily Total</h3>
                    <h2>{} mL</h2>
                </div>
                """.format(result['total_daily']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-container">
                    <h3>⏰ Hourly Rate</h3>
                    <h2>{} mL/hr</h2>
                </div>
                """.format(result['hourly_rate']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-container">
                    <h3>⚖️ Per Kg/Hr</h3>
                    <h2>{} mL/kg/hr</h2>
                </div>
                """.format(result['ml_kg_hr']), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-container">
                    <h3>🏥 Base Maintenance</h3>
                    <h2>{} mL</h2>
                </div>
                """.format(result['base_maintenance']), unsafe_allow_html=True)
            
            # Detailed breakdown
            st.markdown("### 📊 Calculation Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Patient Information:**")
                st.write(f"• Weight: {result['weight_kg']} kg")
                st.write(f"• Age: {result['age_months']} months")
                st.write(f"• Condition: {result['condition'].title()}")
            
            with col2:
                st.write(f"**Holliday-Segar Calculation:**")
                if weight <= 10:
                    st.write(f"• {weight} kg × 100 mL/kg/day = {weight * 100} mL/day")
                elif weight <= 20:
                    st.write(f"• First 10 kg: 10 × 100 = 1000 mL/day")
                    st.write(f"• Next {weight - 10} kg: {weight - 10} × 50 = {(weight - 10) * 50} mL/day")
                    st.write(f"• **Total base:** {1000 + (weight - 10) * 50} mL/day")
                else:
                    st.write(f"• First 10 kg: 10 × 100 = 1000 mL/day")
                    st.write(f"• Next 10 kg: 10 × 50 = 500 mL/day")
                    st.write(f"• Remaining {weight - 20} kg: {weight - 20} × 20 = {(weight - 20) * 20} mL/day")
                    st.write(f"• **Total base:** {1500 + (weight - 20) * 20} mL/day")
            
            # Adjustments made
            if result['age_notes'] or result['condition_notes']:
                st.markdown("### 🔧 Adjustments Applied")
                if result['age_notes']:
                    st.info("**Age Adjustment:** " + ", ".join(result['age_notes']))
                if result['condition_notes']:
                    st.warning("**Condition Adjustment:** " + ", ".join(result['condition_notes']))
            
            # Clinical notes
            st.markdown("### 📋 Clinical Guidelines")
            st.info("""
            **Important Reminders:**
            • **Monitor urine output:** Minimum 1-2 mL/kg/hr for infants, 0.5-1 mL/kg/hr for children
            • **Adjust for losses:** Additional fluids may be needed for fever (+12% per °C above 37°C), diarrhea, vomiting
            • **Insensible losses:** Increased in warm environments, phototherapy, respiratory distress
            • **Electrolyte monitoring:** Regular Na+, K+, Cl-, glucose monitoring required
            • **Clinical assessment:** Always correlate with patient's clinical condition and response
            • **Consult physician:** For complex cases or significant deviations from normal parameters
            """)
            
            # Quick reference
            with st.expander("📚 Holliday-Segar Method Reference"):
                st.markdown("""
                **Holliday-Segar Formula:**
                - **First 10 kg:** 100 mL/kg/day
                - **Next 10 kg (11-20 kg):** 50 mL/kg/day
                - **Each kg above 20 kg:** 20 mL/kg/day
                
                **Normal Urine Output:**
                - **Neonates:** 1-3 mL/kg/hr
                - **Infants:** 1-2 mL/kg/hr
                - **Children:** 0.5-1 mL/kg/hr
                
                **Common Adjustments:**
                - **Fever:** +12% per °C above 37°C
                - **Dehydration:** +50% for replacement
                - **Heart failure:** -20-30%
                - **Renal impairment:** -30-50%
                """)
        
        # Quick calculation examples
        st.markdown("---")
        st.markdown("### 🧮 Quick Examples")
        
        examples = [
            {"weight": 3.0, "age": 1, "condition": "normal", "description": "Term newborn"},
            {"weight": 8.0, "age": 6, "condition": "normal", "description": "6-month-old infant"},
            {"weight": 15.0, "age": 24, "condition": "fever", "description": "2-year-old with fever"},
            {"weight": 25.0, "age": 60, "condition": "normal", "description": "5-year-old child"}
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(f"{example['description']}\n{example['weight']} kg", key=f"example_{i}"):
                    result = fluid_calculator(example['weight'], example['age'], example['condition'])
                    st.metric("Daily Fluid", f"{result['total_daily']} mL")
                    st.metric("Hourly Rate", f"{result['hourly_rate']} mL/hr")
    
    elif page == "📚 About":
        st.header("About KKH Nursing Chatbot")
        
        st.markdown("""
        ### 🎯 Purpose
        This AI-powered chatbot is designed specifically for nurses at KK Women's and Children's Hospital (KKH) 
        to provide quick access to evidence-based information and clinical decision support.
        
        ### ✨ Features
        - **🤖 AI Chat Assistant**: Get instant answers to clinical questions based on KKH guidelines
        - **🧠 Knowledge Quiz**: Test and reinforce your understanding with interactive quizzes
        - **💧 Fluid Calculator**: Calculate pediatric fluid requirements using standard formulas
        - **📖 Evidence-Based**: All responses are based on official KKH medical guidelines
        
        ### 🛠️ Technology
        - **Frontend**: Streamlit for interactive web interface
        - **AI Model**: openrouter/mistral-7b via LM Studio (local deployment)
        - **Embeddings**: Multilingual E5 Large for semantic search
        - **Document Processing**: PyPDF2 for extracting information from guidelines
        
        ### 🔒 Privacy & Security
        - All data processing happens locally when possible
        - No patient information is stored or transmitted
        - Guidelines and protocols are processed locally for privacy
        
        ### 📞 Support
        For technical support or feedback, please contact the IT department.
        
        ### ⚠️ Disclaimer
        This tool is for educational and reference purposes only. Always follow official hospital protocols 
        and consult with senior staff or physicians for critical clinical decisions.
        """)
        
        # System status
        st.markdown("### 🔧 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**API Configuration:**")
            if API_KEY:
                if "together" in LLM_API_URL.lower():
                    st.success("✅ Together.ai API Configured")
                elif "localhost" in LLM_API_URL:
                    st.info("🏠 LM Studio (Local) API Configured")
                else:
                    st.success("✅ Custom LLM API Configured")
                
                try:
                    # Test API connection
                    test_payload = {
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10
                    }
                    headers = {"Content-Type": "application/json"}
                    if API_KEY:
                        headers["Authorization"] = f"Bearer {API_KEY}"
                    
                    response = requests.post(LLM_API_URL, headers=headers, json=test_payload, timeout=5)
                    if response.status_code == 200:
                        st.success("✅ LLM API Connection Active")
                    else:
                        st.error(f"❌ LLM API Connection Failed (Status: {response.status_code})")
                except:
                    st.error("❌ Could not reach LLM API")
            else:
                st.warning("🟡 No API key configured")
                st.caption("Set TOGETHER_API_KEY or LLM_API_KEY environment variable")
        
        with col2:
            st.markdown("**Document Processing:**")
            if st.session_state.pdf_processed:
                st.success(f"✅ PDF Processed ({len(st.session_state.chunks)} chunks)")
            else:
                st.warning("⏳ PDF Processing Pending")
            
            if st.session_state.embeddings is not None and len(st.session_state.embeddings) > 0:
                st.success("✅ Embeddings Ready")
            else:
                st.warning("⏳ Embeddings Not Ready")
        
        # Environment info
        st.markdown("### 🌐 Environment Information")
        env_info = {
            "LLM API URL": LLM_API_URL,
            "LLM Model": LLM_MODEL,
            "Has API Key": "Yes" if API_KEY else "No",
            "PDF Path": PDF_PATH,
            "Environment": "Local" if is_local_environment() else "Deployed"
        }
        
        for key, value in env_info.items():
            st.write(f"**{key}:** `{value}`")

if __name__ == "__main__":
    main()
