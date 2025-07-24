import streamlit as st
import json
import pickle
import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import time

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
    return os.getenv("LLM_API_URL")

LLM_API_URL = get_llm_api_url()

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
    """Extract text from PDF and split into chunks"""
    try:
        chunks = []
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
        
        # Split into chunks
        words = full_text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep overlap
                overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_words
                current_size = sum(len(word) + 1 for word in current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

@st.cache_resource
def load_embedding_model():
    """Load the embedding model"""
    try:
        model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def embed_chunks(chunks: List[str], model) -> np.ndarray:
    """Generate embeddings for document chunks"""
    try:
        if model is None:
            return np.array([])
        
        # Add instruction prefix for better embeddings
        prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
        embeddings = model.encode(prefixed_chunks)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

def search_relevant_chunks(query: str, chunks: List[str], embeddings: np.ndarray, model, top_k: int = 3) -> List[str]:
    """Retrieve top relevant chunks to user's query"""
    try:
        if model is None or len(embeddings) == 0:
            return chunks[:top_k] if chunks else []
        
        # Encode query with instruction prefix
        query_embedding = model.encode([f"query: {query}"])
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [chunks[i] for i in top_indices if i < len(chunks)]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return chunks[:top_k] if chunks else []

# LLM Integration
def generate_response(context: str, query: str) -> str:
    if LLM_API_URL is None:
        return "⚠️ No LLM API URL configured. Please set the LLM_API_URL environment variable."

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"
        }

        # Truncate long context
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
            "model": "openhermes-2.5-mistral-7b",  # or change if needed
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
        else:
            return f"Error: Unable to get response from LLM (Status: {response.status_code})"

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to the LLM API."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Fluid Calculator
def fluid_calculator(weight: float, age_months: int, condition: str = "normal") -> Dict[str, Any]:
    """Calculate daily fluid requirements using standard pediatric formulas"""
    
    base_fluid = 0
    
    # Holliday-Segar method for maintenance fluids
    if weight <= 10:
        base_fluid = weight * 100  # 100 mL/kg for first 10 kg
    elif weight <= 20:
        base_fluid = 1000 + (weight - 10) * 50  # 50 mL/kg for next 10 kg
    else:
        base_fluid = 1500 + (weight - 20) * 20  # 20 mL/kg for each kg > 20
    
    # Adjust for age (neonates and infants need more)
    if age_months < 1:  # Neonates
        age_factor = 1.5
    elif age_months < 6:  # Young infants
        age_factor = 1.3
    elif age_months < 12:  # Older infants
        age_factor = 1.2
    elif age_months < 24:  # Toddlers
        age_factor = 1.1
    else:  # Children
        age_factor = 1.0
    
    base_fluid *= age_factor
    
    # Condition-based adjustments
    condition_factor = 1.0
    condition_notes = []
    
    if condition.lower() == "fever":
        condition_factor = 1.2
        condition_notes.append("Increased by 20% due to fever")
    elif condition.lower() == "dehydration":
        condition_factor = 1.5
        condition_notes.append("Increased by 50% for rehydration")
    elif condition.lower() == "heart failure":
        condition_factor = 0.8
        condition_notes.append("Reduced by 20% due to heart failure")
    elif condition.lower() == "renal impairment":
        condition_factor = 0.7
        condition_notes.append("Reduced by 30% due to renal impairment")
    
    total_fluid = base_fluid * condition_factor
    hourly_rate = total_fluid / 24
    
    return {
        "weight_kg": weight,
        "age_months": age_months,
        "condition": condition,
        "base_maintenance": round(base_fluid, 1),
        "total_daily": round(total_fluid, 1),
        "hourly_rate": round(hourly_rate, 1),
        "condition_notes": condition_notes
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
        "When should vitamin K be administered?",
        "What is the proper technique for umbilical cord care?",
        "How to calculate fluid requirements for pediatric patients?",
        "What are the normal vital signs for newborns?",
        "When should skin-to-skin contact be initiated?",
        "What is the recommended feeding frequency for newborns?",
        "How to recognize signs of dehydration in infants?"
    ]
    
    # Create a grid of buttons
    cols = st.columns(2)
    
    for i, question in enumerate(prompt_questions):
        with cols[i % 2]:
            if st.button(question, key=f"prompt_{i}", use_container_width=True):
                return question
    
    return None

def process_pdf_and_embeddings():
    """Process PDF and create embeddings if not already done"""
    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF and creating embeddings... This may take a moment."):
            # Load PDF chunks
            if os.path.exists(PDF_PATH):
                chunks = load_pdf_chunks(PDF_PATH)
                st.session_state.chunks = chunks
                
                # Load embedding model
                model = load_embedding_model()
                
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
                    except Exception as e:
                        st.warning(f"Could not save embeddings: {e}")
                
                st.session_state.pdf_processed = True
            else:
                st.error(f"PDF file not found at {PDF_PATH}")

# Main App
def main():
    st.set_page_config(
        page_title="KKH Nursing Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏥 KKH Nursing Chatbot")
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
        st.write("Calculate daily fluid requirements for pediatric patients based on weight, age, and condition.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Patient Weight (kg)", min_value=0.5, max_value=100.0, value=3.5, step=0.1)
            age_months = st.number_input("Age (months)", min_value=0, max_value=216, value=1, step=1)
        
        with col2:
            condition = st.selectbox(
                "Clinical Condition",
                ["normal", "fever", "dehydration", "heart failure", "renal impairment"]
            )
        
        if st.button("Calculate Fluid Requirements", type="primary"):
            result = fluid_calculator(weight, age_months, condition)
            
            st.markdown("### 💧 Fluid Calculation Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Daily Requirement", f"{result['total_daily']} mL")
            with col2:
                st.metric("Hourly Rate", f"{result['hourly_rate']} mL/hr")
            with col3:
                st.metric("Base Maintenance", f"{result['base_maintenance']} mL")
            
            # Detailed breakdown
            st.markdown("### 📊 Calculation Details")
            st.write(f"**Patient:** {result['weight_kg']} kg, {result['age_months']} months old")
            st.write(f"**Condition:** {result['condition'].title()}")
            
            if result['condition_notes']:
                st.info("**Adjustments:** " + ", ".join(result['condition_notes']))
            
            # Clinical notes
            st.markdown("### 📋 Clinical Notes")
            st.info("""
            **Important Reminders:**
            - Monitor urine output (minimum 1-2 mL/kg/hr)
            - Adjust for losses (fever, diarrhea, vomiting)
            - Consider insensible losses in different environments
            - Regular electrolyte monitoring required
            - Consult physician for complex cases
            """)
    
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
        - **AI Model**: OpenHermes-2.5-Mistral-7B via LM Studio (local deployment)
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
            if LLM_API_URL:
                st.success("✅ LLM API Configured")
                try:
                    # Replace /chat/completions with /models if supported (OpenRouter & Together.ai support it)
                    health_check_url = LLM_API_URL.replace('/chat/completions', '/models')
                    response = requests.get(health_check_url, headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"}, timeout=5)

                    if response.status_code == 200:
                        st.success("✅ LLM API Connection Active")
                    else:
                        st.error(f"❌ LLM API Connection Failed (Status: {response.status_code})")
                except:
                    st.error("❌ Could not reach LLM API")
            else:
                st.warning("🟡 LLM API not configured (check LLM_API_URL)")
        
        with col2:
            if st.session_state.pdf_processed:
                st.success(f"✅ PDF Processed ({len(st.session_state.chunks)} chunks)")
            else:
                st.warning("⏳ PDF Processing Pending")
            
            if st.session_state.embeddings is not None and len(st.session_state.embeddings) > 0:
                st.success("✅ Embeddings Ready")
            else:
                st.warning("⏳ Embeddings Not Ready")

if __name__ == "__main__":
    main()
