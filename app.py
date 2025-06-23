import streamlit as st
import os
# Importar y aplicar el parche para LocallyConnected2D antes de cargar DeepFace
import keras_bridge
keras_bridge.install_patch()
from resume_parser import ResumeParser

# Try to import optional modules
try:
    from facial_emotion import FacialEmotionAnalyzer
    FACIAL_EMOTION_AVAILABLE = True
except Exception as e:
    st.warning(f"Analsis facial no disponible: {str(e)}")
    FACIAL_EMOTION_AVAILABLE = False
    FacialEmotionAnalyzer = None

from voice_analysis import VoiceAnalyzer
# Importar SpeechToText directamente del módulo
from speech_to_text import SpeechToText
from content_matcher import ContentMatcher
from interview_bot import InterviewBot
import tempfile
import time
import cv2
import numpy as np
import threading
import queue
import sounddevice as sd
import wave
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image

# Palabras clave por puesto para matching
POSITION_KEYWORDS = {
    "Desarrollador Frontend": [
        "html", "css", "javascript", "react", "angular", "vue", "typescript", "bootstrap", 
        "sass", "webpack", "responsive", "ui", "ux", "jquery", "tailwind", "figma"
    ],
    "Desarrollador Backend": [
        "python", "java", "node.js", "php", "ruby", "go", "api", "rest", "database", 
        "sql", "mongodb", "postgresql", "docker", "microservices", "spring", "django"
    ],
    "Desarrollador Full Stack": [
        "html", "css", "javascript", "python", "java", "react", "node.js", "database", 
        "api", "git", "docker", "aws", "mongodb", "postgresql", "full stack"
    ],
    "Data Scientist": [
        "python", "r", "machine learning", "pandas", "numpy", "scikit-learn", "tensorflow", 
        "pytorch", "jupyter", "statistics", "data analysis", "sql", "tableau", "power bi",
        "data science", "statistical analysis", "data visualization", "predictive modeling"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "azure", "jenkins", "ansible", "terraform", 
        "linux", "bash", "git", "ci/cd", "monitoring", "infrastructure", "cloud"
    ],
    "Product Manager": [
        "product management", "agile", "scrum", "roadmap", "stakeholders", "analytics", 
        "user research", "jira", "confluence", "strategy", "market research", "kpi"
    ],
    "UI/UX Designer": [
        "figma", "sketch", "adobe", "photoshop", "illustrator", "wireframes", "prototyping", 
        "user research", "usability", "design thinking", "responsive design", "typography"
    ],
    "QA Engineer": [
        "testing", "automation", "selenium", "junit", "pytest", "quality assurance", 
        "bug tracking", "test cases", "regression", "performance testing", "api testing"
    ],
    "Software Architect": [
        "architecture", "design patterns", "microservices", "scalability", "performance", 
        "system design", "cloud", "distributed systems", "technical leadership", "apis"
    ],
    "Mobile Developer": [
        "android", "ios", "swift", "kotlin", "react native", "flutter", "mobile", 
        "app store", "google play", "xamarin", "ionic", "mobile ui"
    ],
    "Machine Learning Engineer": [
        "machine learning", "deep learning", "tensorflow", "pytorch", "python", "scikit-learn", 
        "neural networks", "data science", "ai", "model deployment", "mlops"
    ],
    "Cybersecurity Specialist": [
        "cybersecurity", "penetration testing", "vulnerability assessment", "firewall", 
        "encryption", "compliance", "risk assessment", "incident response", "forensics",
        "security audit", "malware analysis", "network security", "ethical hacking"
    ],
    "Database Administrator": [
        "database", "sql", "mysql", "postgresql", "oracle", "mongodb", "backup", 
        "performance tuning", "indexing", "replication", "data modeling"
    ],
    "Cloud Engineer": [
        "aws", "azure", "gcp", "cloud", "serverless", "lambda", "ec2", "s3", 
        "kubernetes", "docker", "terraform", "cloudformation", "devops"
    ]
}

# Initialize components with lazy loading
@st.cache_resource
def get_resume_parser():
    return ResumeParser()

@st.cache_resource  
def get_facial_analyzer():
    try:
        return FacialEmotionAnalyzer()
    except Exception as e:
        print(f"Error al inicializar el analizador facial: {str(e)}")
        return None

@st.cache_resource
def get_voice_analyzer():
    return VoiceAnalyzer()

@st.cache_resource
def get_speech_to_text():
    return SpeechToText()

@st.cache_resource
def get_content_matcher():
    return ContentMatcher()

@st.cache_resource
def get_interview_bot():
    return InterviewBot()

# Lazy initialization
resume_parser = get_resume_parser()
facial_analyzer = get_facial_analyzer()
FACIAL_EMOTION_AVAILABLE = facial_analyzer is not None
voice_analyzer = get_voice_analyzer()
speech_to_text = get_speech_to_text()
content_matcher = get_content_matcher()
interview_bot = get_interview_bot()

# Global variables for state management
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'ready_to_record' not in st.session_state:
    st.session_state.ready_to_record = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'skills' not in st.session_state:
    st.session_state.skills = []
# Nuevas variables para información del candidato
if 'candidate_name' not in st.session_state:
    st.session_state.candidate_name = ""
if 'target_position' not in st.session_state:
    st.session_state.target_position = ""
if 'profile_completed' not in st.session_state:
    st.session_state.profile_completed = False

def analyze_cv_skills(skills):
    """Analizar las habilidades del CV y categorizar."""
    cv_skills_lower = [skill.lower() for skill in skills]
    
    # Categorizar habilidades
    categories = {
        'Lenguajes de Programación': [],
        'Bases de Datos': [],
        'Frameworks/Librerías': [],
        'Herramientas de Análisis': [],
        'Cloud/DevOps': [],
        'Soft Skills': [],
        'Otras': []
    }
    
    # Mapeo de habilidades a categorías
    programming_langs = ['python', 'java', 'javascript', 'c++', 'c#', 'r', 'sql', 'php', 'ruby', 'go']
    databases = ['mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'sql']
    frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'bootstrap', 'jquery']
    analysis_tools = ['excel', 'tableau', 'power bi', 'sas', 'spss', 'jupyter', 'pandas', 'numpy']
    cloud_devops = ['aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'git', 'linux']
    soft_skills = ['liderazgo', 'comunicacion', 'trabajo en equipo', 'analisis', 'organizacion']
    
    for skill in cv_skills_lower:
        categorized = False
        
        if any(lang in skill for lang in programming_langs):
            categories['Lenguajes de Programación'].append(skill)
            categorized = True
        if any(db in skill for db in databases):
            categories['Bases de Datos'].append(skill)
            categorized = True
        if any(fw in skill for fw in frameworks):
            categories['Frameworks/Librerías'].append(skill)
            categorized = True
        if any(tool in skill for tool in analysis_tools):
            categories['Herramientas de Análisis'].append(skill)
            categorized = True
        if any(cloud in skill for cloud in cloud_devops):
            categories['Cloud/DevOps'].append(skill)
            categorized = True
        if any(soft in skill for soft in soft_skills):
            categories['Soft Skills'].append(skill)
            categorized = True
        
        if not categorized:
            categories['Otras'].append(skill)
    
    return categories

def suggest_best_positions(skills):
    """Sugerir los mejores puestos basados en las habilidades del CV."""
    position_scores = {}
    
    for position, keywords in POSITION_KEYWORDS.items():
        match_percentage, matches, _ = calculate_position_match(skills, position)
        position_scores[position] = {
            'percentage': match_percentage,
            'matches': len(matches),
            'total_keywords': len(keywords)
        }
    
    # Ordenar por porcentaje de match
    sorted_positions = sorted(position_scores.items(), key=lambda x: x[1]['percentage'], reverse=True)
    return sorted_positions[:5]  # Top 5 posiciones

def calculate_position_match(skills, position):
    """Calcular el porcentaje de matching entre habilidades del CV y el puesto."""
    if position not in POSITION_KEYWORDS:
        return 0, [], []
    
    position_keywords = [kw.lower() for kw in POSITION_KEYWORDS[position]]
    cv_skills = [skill.lower().strip() for skill in skills]
    
    print(f"🔍 DEBUG: Calculando match para {position}")
    print(f"🔍 DEBUG: CV Skills: {cv_skills}")
    print(f"🔍 DEBUG: Position Keywords: {position_keywords}")
    
    # Encontrar coincidencias con algoritmo más estricto
    matches = []
    for keyword in position_keywords:
        keyword_clean = keyword.strip()
        for skill in cv_skills:
            skill_clean = skill.strip()
            
            # Coincidencia exacta (más estricta)
            if (keyword_clean == skill_clean or 
                (len(keyword_clean) > 3 and keyword_clean in skill_clean) or
                (len(skill_clean) > 3 and skill_clean in keyword_clean)):
                matches.append(keyword_clean)
                print(f"✅ MATCH FOUND: '{keyword_clean}' matches with '{skill_clean}'")
                break
    
    # Calcular porcentaje
    match_percentage = (len(set(matches)) / len(position_keywords)) * 100 if position_keywords else 0
    
    print(f"🔍 DEBUG: Final matches: {list(set(matches))}")
    print(f"🔍 DEBUG: Match percentage: {match_percentage:.1f}%")
    
    return match_percentage, list(set(matches)), position_keywords

def create_cv_analysis_chart(cv_categories):
    """Crear gráfica de análisis de habilidades del CV por categorías."""
    # Filtrar categorías con habilidades
    filtered_categories = {k: v for k, v in cv_categories.items() if v}
    
    if not filtered_categories:
        return None
    
    categories = list(filtered_categories.keys())
    counts = [len(skills) for skills in filtered_categories.values()]
    
    fig = px.bar(
        x=counts,
        y=categories,
        orientation='h',
        title='Distribución de Habilidades por Categoría',
        labels={'x': 'Cantidad de Habilidades', 'y': 'Categorías'},
        color=counts,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_position_match_chart(match_percentage, matches, total_keywords):
    """Crear gráfica de matching con el puesto."""
    fig = go.Figure()
    
    # Gráfica de gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = match_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Compatibilidad con el Puesto"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_skills_comparison_chart(cv_skills, interview_skills):
    """Crear gráfica comparativa CV vs Entrevista."""
    # Contar habilidades mencionadas
    cv_count = len(cv_skills) if cv_skills else 0
    interview_count = len(interview_skills) if interview_skills else 0
    
    fig = go.Figure(data=[
        go.Bar(name='CV', x=['Habilidades Técnicas'], y=[cv_count], marker_color='lightblue'),
        go.Bar(name='Entrevista', x=['Habilidades Técnicas'], y=[interview_count], marker_color='lightcoral')
    ])
    
    fig.update_layout(
        title='Comparación: Habilidades en CV vs Mencionadas en Entrevista',
        xaxis_title='Fuente',
        yaxis_title='Cantidad de Habilidades',
        barmode='group',
        height=400
    )
    
    return fig

def create_emotion_table(emotion_analysis):
    """Crear tabla bonita de emociones detectadas."""
    # Extraer emociones del análisis
    if emotion_analysis and 'average_emotions' in emotion_analysis:
        emotions = emotion_analysis['average_emotions']
    elif emotion_analysis and 'emotions' in emotion_analysis:
        emotions = emotion_analysis['emotions']
    else:
        # Emociones por defecto si no hay análisis
        emotions = {
            'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 
            'sad': 0, 'surprise': 0, 'neutral': 0
        }
    
    # Crear DataFrame
    emotion_data = []
    emotion_icons = {
        'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
        'sad': '😢', 'surprise': '😮', 'neutral': '😐'
    }
    
    emotion_names = {
        'angry': 'Enojado', 'disgust': 'Disgusto', 'fear': 'Miedo', 'happy': 'Feliz',
        'sad': 'Triste', 'surprise': 'Sorpresa', 'neutral': 'Neutral'
    }
    
    for emotion, value in emotions.items():
        emotion_data.append({
            'Emoción': f"{emotion_icons.get(emotion, '😐')} {emotion_names.get(emotion, emotion.title())}",
            'Porcentaje': f"{value:.1f}%",
            'Valor': value
        })
    
    df = pd.DataFrame(emotion_data)
    df = df.sort_values('Valor', ascending=False)
    
    return df

def save_uploaded_file(uploaded_file_bytes, filename):
    """Save uploaded file to temporary directory."""
    try:
        # Crear un nombre único para el archivo temporal
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        file_extension = os.path.splitext(filename)[1]
        temp_filename = f"uploaded_cv_{unique_id}{file_extension}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Escribir el archivo
        with open(temp_path, 'wb') as tmp_file:
            tmp_file.write(uploaded_file_bytes)
        
        # Verificar que el archivo se creó correctamente
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"✅ Archivo guardado exitosamente: {temp_path}")
            return temp_path
        else:
            print(f"❌ Error: Archivo no se guardó correctamente: {temp_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error guardando archivo: {str(e)}")
        return None



def process_frame(frame, emotion_queue):
    """Process a single frame for emotion analysis with optimization."""
    try:
        if facial_analyzer:
            print(f"🔍 DEBUG: Processing frame with facial_analyzer available")
            # Optimización: reducir resolución para análisis más rápido
            height, width = frame.shape[:2]
            if width > 640:  # Solo redimensionar si es necesario
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
                print(f"🔍 DEBUG: Frame resized from {width}x{height} to {new_width}x{new_height}")
            
            analysis = facial_analyzer.analyze_frame(frame)
            print(f"🔍 DEBUG: Frame analysis result: {analysis}")
            # Ensure the analysis has the required keys
            if 'dominant_emotion' not in analysis:
                analysis['dominant_emotion'] = 'neutral'
                print(f"⚠️ WARNING: No dominant_emotion found, defaulting to neutral")
            if 'emotions' not in analysis:
                analysis['emotions'] = {}
                print(f"⚠️ WARNING: No emotions found, defaulting to empty dict")
            emotion_queue.put(analysis)
        else:
            print(f"❌ ERROR: facial_analyzer not available")
            emotion_queue.put({'dominant_emotion': 'unavailable', 'emotions': {}})
    except Exception as e:
        print(f"❌ ERROR processing frame: {str(e)}")
        import traceback
        print(f"❌ ERROR traceback: {traceback.format_exc()}")
        # Put a default emotion result even when there's an error
        emotion_queue.put({'dominant_emotion': 'error', 'emotions': {}})

def record_audio(duration, sample_rate=44100):
    """Record audio for a specified duration."""
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording

def save_audio(recording, sample_rate=44100):
    """Save recorded audio to a temporary WAV file."""
    try:
        # Create a temporary file with explicit close to avoid Windows file locking
        import uuid
        temp_filename = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Ensure recording is not empty and has valid data
        if len(recording) == 0:
            print("Warning: Empty recording")
            return None
            
        # Normalize audio data
        audio_data = np.array(recording).flatten()
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        # Save with explicit file handling
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        # Verify file was created
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 44:  # More than just header
            return temp_path
        else:
            print(f"Error: Audio file not created properly: {temp_path}")
            return None
            
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return None

def analyze_response(audio_path, video_frames, question, skills):
    """Analyze the user's response comprehensively with optimized parallel processing."""
    import concurrent.futures
    
    # Initialize default results
    transcription = {'text': '', 'segments': [], 'language': 'es'}
    voice_analysis = {}
    emotion_analysis = {'dominant_emotion': 'unavailable', 'emotions': {}}
    
    def analyze_audio():
        """Audio analysis in separate thread"""
        nonlocal transcription, voice_analysis
        if audio_path and os.path.exists(audio_path):
            try:
                # Transcribe speech
                transcription = speech_to_text.transcribe(audio_path)
                
                # Analyze voice characteristics
                voice_features = voice_analyzer.extract_features(audio_path)
                voice_analysis = voice_analyzer.analyze_voice_characteristics(voice_features)
            except Exception as e:
                print(f"Error in audio analysis: {str(e)}")
                transcription = {'text': 'Error processing audio', 'segments': [], 'language': 'en'}
                voice_analysis = {'error': 'Audio analysis failed'}
        else:
            print(f"Audio file not available: {audio_path}")
            transcription = {'text': 'Audio recording failed', 'segments': [], 'language': 'en'}
            voice_analysis = {'error': 'No audio file'}
    
    def analyze_emotions():
        """Emotion analysis in separate thread"""
        nonlocal emotion_analysis
        if facial_analyzer and video_frames:
            try:
                print(f"🔍 DEBUG: Analyzing {len(video_frames)} video frames for final emotion summary")
                # Optimización: usar solo cada 3er frame para análisis final
                sample_frames = video_frames[::3] if len(video_frames) > 30 else video_frames
                emotion_analysis = facial_analyzer.analyze_frames(sample_frames)
                print(f"🔍 DEBUG: Final emotion analysis result: {emotion_analysis}")
                # Ensure the analysis has the required keys
                if not isinstance(emotion_analysis, dict):
                    emotion_analysis = {'dominant_emotion': 'error', 'emotions': {}}
                    print(f"⚠️ WARNING: emotion_analysis is not a dict, got: {type(emotion_analysis)}")
                if 'dominant_emotion' not in emotion_analysis:
                    emotion_analysis['dominant_emotion'] = 'neutral'
                    print(f"⚠️ WARNING: No dominant_emotion in final analysis")
                if 'emotions' not in emotion_analysis:
                    emotion_analysis['emotions'] = {}
                    print(f"⚠️ WARNING: No emotions in final analysis")
            except Exception as e:
                print(f"❌ ERROR in emotion analysis: {str(e)}")
                import traceback
                print(f"❌ ERROR traceback: {traceback.format_exc()}")
                emotion_analysis = {'dominant_emotion': 'error', 'emotions': {}}
        else:
            print(f"❌ ERROR: facial_analyzer={facial_analyzer is not None}, video_frames={len(video_frames) if video_frames else 0}")
            emotion_analysis = {'dominant_emotion': 'unavailable', 'emotions': {}}
    
    # Ejecutar análisis de audio y emociones en paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = executor.submit(analyze_audio)
        emotion_future = executor.submit(analyze_emotions)
        
        # Esperar a que ambos terminen
        concurrent.futures.wait([audio_future, emotion_future])
    
    # Análisis de contenido y evaluación (rápidos, no necesitan paralelización)
    content_analysis = content_matcher.analyze_content_match(skills, transcription['text'])
    answer_evaluation = interview_bot.evaluate_answer(question, transcription['text'], skills)
    
    return {
        'transcription': transcription['text'],
        'voice_analysis': voice_analysis,
        'emotion_analysis': emotion_analysis,
        'content_analysis': content_analysis,
        'answer_evaluation': answer_evaluation,
        'timestamp': datetime.now().isoformat()
    }

def main():
    # HEADER CON LOGO Y BRANDING
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
            <div style="background: white; border-radius: 50%; padding: 10px; margin-right: 15px; 
                       box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 24px; font-weight: bold;">👥</span>
                </div>
            </div>
            <div style="text-align: left;">
                <h2 style="color: white; margin: 0; font-size: 28px;">Gesteamworks.com</h2>
                <p style="color: #e0f7ff; margin: 0; font-size: 14px; font-style: italic;">
                    Impulsando mejores resultados
                </p>
            </div>
        </div>
        <h3 style="color: white; margin: 10px 0;">🎯 Sistema de Análisis de Entrevistas</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # PANTALLA DE BIENVENIDA - Solo mostrar si no se ha completado el perfil
    if not st.session_state.profile_completed:
        
        # Formulario de información del candidato
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 25px; border-radius: 10px; 
                           border: 1px solid #e9ecef;">
                    <h4 style="text-align: center; color: #495057; margin-bottom: 20px;">
                        📝 Información del Candidato
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Campos del formulario
                candidate_name = st.text_input(
                    "👤 Nombre completo:",
                    placeholder="Ej: Juan Pérez García",
                    help="Ingresa tu nombre completo como aparece en tu CV"
                )
                
                # Lista de puestos comunes
                position_options = [
                    "Selecciona un puesto...",
                    "Desarrollador Frontend",
                    "Desarrollador Backend", 
                    "Desarrollador Full Stack",
                    "Data Scientist",
                    "DevOps Engineer",
                    "Product Manager",
                    "UI/UX Designer",
                    "QA Engineer",
                    "Software Architect",
                    "Mobile Developer",
                    "Machine Learning Engineer",
                    "Cybersecurity Specialist",
                    "Database Administrator",
                    "Cloud Engineer",
                    "Otro (especificar abajo)"
                ]
                
                target_position = st.selectbox(
                    "💼 Puesto al que te postulas:",
                    options=position_options,
                    help="Selecciona el puesto que mejor describa la posición a la que aplicas"
                )
                
                # Campo adicional si selecciona "Otro"
                custom_position = ""
                if target_position == "Otro (especificar abajo)":
                    custom_position = st.text_input(
                        "✏️ Especifica el puesto:",
                        placeholder="Ej: Senior Data Engineer"
                    )
                
                # Botón para continuar
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("🚀 Comenzar Entrevista", use_container_width=True, type="primary"):
                        if candidate_name.strip() and target_position != "Selecciona un puesto...":
                            # Guardar información
                            st.session_state.candidate_name = candidate_name.strip()
                            if target_position == "Otro (especificar abajo)":
                                st.session_state.target_position = custom_position.strip() if custom_position.strip() else "Puesto no especificado"
                            else:
                                st.session_state.target_position = target_position
                            st.session_state.profile_completed = True
                            st.success(f"¡Perfecto, {candidate_name}! Ahora puedes subir tu CV para comenzar.")
                            st.rerun()
                        else:
                            st.error("⚠️ Por favor completa todos los campos antes de continuar.")
        
        # Información adicional mientras completa el formulario
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <h4>🎤 Análisis de Voz</h4>
                <p>Evaluamos tu tono, claridad y fluidez verbal</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <h4>😊 Análisis Facial</h4>
                <p>Detectamos emociones y expresiones durante la entrevista</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <h4>📊 Análisis de Contenido</h4>
                <p>Comparamos tus respuestas con tu CV y experiencia</p>
            </div>
            """, unsafe_allow_html=True)
        
        # INFORMACIÓN DE LA EMPRESA
        st.markdown("---")
        
        # Usar columnas de Streamlit en lugar de CSS Grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; height: 200px;">
                <h4 style="color: white; margin-bottom: 15px;">🎯 Misión</h4>
                <p style="color: #f0f0f0; font-size: 13px; line-height: 1.4;">
                    Ser la mejor opción de plataforma en línea para fomentar, impulsar y mejorar 
                    la operación y gestión de nuestros clientes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; height: 200px;">
                <h4 style="color: white; margin-bottom: 15px;">🔮 Visión</h4>
                <p style="color: #f0f0f0; font-size: 13px; line-height: 1.4;">
                    Ser referente en el mercado para la generación de productividad para nuestras 
                    audiencias, el mercado y nuestros clientes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; height: 200px;">
                <h4 style="color: white; margin-bottom: 15px;">🚀 Objetivo</h4>
                <p style="color: #f0f0f0; font-size: 13px; line-height: 1.4;">
                    Desarrollar y expandir soluciones de software innovadoras que optimicen la 
                    eficiencia operativa y la gestión de recursos.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Descripción ampliada
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
            <p style="color: #f0f0f0; font-size: 14px; line-height: 1.6; margin: 0;">
                <strong>Permitiendo a empresas, organizaciones, entidades gubernamentales, 
                profesionales independientes y emprendedores alcanzar resultados superiores 
                y sostenibles, mientras fortalece su crecimiento en la industria tecnológica.</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return  # No mostrar el resto de la interfaz hasta completar el perfil
    
    # INTERFAZ PRINCIPAL - Solo mostrar después de completar el perfil
    # Header con información del candidato
    st.markdown(f"""
    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #2d5a2d;">
            👤 Candidato: <span style="color: #1a4a1a;">{st.session_state.candidate_name}</span> | 
            💼 Puesto: <span style="color: #1a4a1a;">{st.session_state.target_position}</span>
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for resume upload
    st.sidebar.header("📄 Subir CV")
    
    # Botón para cambiar información del candidato
    if st.sidebar.button("✏️ Cambiar información del candidato"):
        st.session_state.profile_completed = False
        st.session_state.candidate_name = ""
        st.session_state.target_position = ""
        st.rerun()
    
    resume_file = st.sidebar.file_uploader("Subir CV (PDF)", type=['pdf'])
    
    # Mostrar estado del análisis facial
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Estado del Sistema")
    if FACIAL_EMOTION_AVAILABLE:
        st.sidebar.success("✅ Análisis facial disponible")
    else:
        st.sidebar.error("❌ Análisis facial no disponible")
        st.sidebar.info("Intente reiniciar la aplicación o verificar la cámara")
    
    # Main interview interface
    st.header("Entrevista en vivo")
    
    if resume_file:
        # Process resume
        with st.spinner("Analizando CV..."):
            try:
                # Guardar archivo subido
                resume_path = save_uploaded_file(resume_file.getvalue(), resume_file.name)
                
                if resume_path is None:
                    st.error("❌ Error al guardar el archivo CV. Intente nuevamente.")
                    return
                
                print(f"🔍 DEBUG: Intentando analizar CV en: {resume_path}")
                
                # Verificar que el archivo existe antes de procesarlo
                if not os.path.exists(resume_path):
                    st.error(f"❌ El archivo CV no se encontró en: {resume_path}")
                    return
                
                # Analizar el CV
                resume_data = resume_parser.analyze_resume(resume_path)
                
                if resume_data and resume_data.get('skills'):
                    st.session_state.skills = resume_data['skills']
                    st.success("✅ Análisis de CV completado!")
                    
                    # Mostrar habilidades extraídas de manera más clara
                    st.sidebar.markdown("**🎯 Habilidades extraídas del CV:**")
                    for i, skill in enumerate(resume_data['skills'], 1):
                        st.sidebar.markdown(f"{i}. {skill}")
                    
                    # Debug: mostrar información adicional
                    st.sidebar.markdown("---")
                    st.sidebar.markdown(f"**📊 Total de habilidades:** {len(resume_data['skills'])}")
                    
                    # Debug en consola
                    print(f"🔍 DEBUG: Habilidades extraídas del CV:")
                    for skill in resume_data['skills']:
                        print(f"  - '{skill}'")
                    
                    # ANÁLISIS COMPLETO DEL CV
                    st.markdown("---")
                    st.header("📊 Análisis Completo del CV")
                    
                    # Analizar y categorizar habilidades del CV
                    cv_categories = analyze_cv_skills(st.session_state.skills)
                    
                    # Sugerir mejores puestos basados en el CV
                    suggested_positions = suggest_best_positions(st.session_state.skills)
                    
                    # Crear tabs para mejor organización
                    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Compatibilidad", "📋 Análisis de Habilidades", "💼 Puestos Sugeridos", "📈 Recomendaciones"])
                    
                    with tab1:
                        st.subheader(f"Compatibilidad con: {st.session_state.target_position}")
                        
                        # Calcular matching con el puesto seleccionado
                        match_percentage, matches, total_keywords = calculate_position_match(
                            st.session_state.skills, 
                            st.session_state.target_position
                        )
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Gráfica de gauge para compatibilidad
                            if st.session_state.target_position in POSITION_KEYWORDS:
                                fig_gauge = create_position_match_chart(match_percentage, matches, total_keywords)
                                st.plotly_chart(fig_gauge, use_container_width=True)
                            else:
                                st.info(f"📋 Puesto '{st.session_state.target_position}' no está en nuestra base de datos de palabras clave.")
                        
                        with col2:
                            # Información detallada del matching
                            st.markdown("""
                            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4a90e2;">
                                <h5 style="color: #2c5aa0; margin-bottom: 15px;">🎯 Detalles del Matching</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.session_state.target_position in POSITION_KEYWORDS:
                                st.metric("Compatibilidad", f"{match_percentage:.1f}%")
                                st.metric("Habilidades Coincidentes", f"{len(matches)}/{len(total_keywords)}")
                                
                                if matches:
                                    st.markdown("**✅ Habilidades que coinciden:**")
                                    for match in matches:
                                        st.markdown(f"• {match}")
                                
                                missing_skills = [kw for kw in total_keywords if kw not in matches]
                                if missing_skills:
                                    with st.expander("📝 Habilidades sugeridas para mejorar"):
                                        for skill in missing_skills[:10]:  # Mostrar solo las primeras 10
                                            st.markdown(f"• {skill}")
                            else:
                                st.info("Para mostrar el análisis de compatibilidad, selecciona un puesto de la lista predefinida.")
                    
                    with tab2:
                        st.subheader("Análisis de Habilidades por Categoría")
                        
                        # Mostrar gráfica de categorías
                        fig_categories = create_cv_analysis_chart(cv_categories)
                        if fig_categories:
                            st.plotly_chart(fig_categories, use_container_width=True)
                        
                        # Mostrar detalles por categoría
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for category, skills in cv_categories.items():
                                if skills and category in ['Lenguajes de Programación', 'Bases de Datos', 'Frameworks/Librerías']:
                                    st.markdown(f"**{category}:**")
                                    for skill in skills:
                                        st.markdown(f"• {skill.title()}")
                                    st.markdown("")
                        
                        with col2:
                            for category, skills in cv_categories.items():
                                if skills and category in ['Herramientas de Análisis', 'Cloud/DevOps', 'Soft Skills', 'Otras']:
                                    st.markdown(f"**{category}:**")
                                    for skill in skills:
                                        st.markdown(f"• {skill.title()}")
                                    st.markdown("")
                    
                    with tab3:
                        st.subheader("Puestos Más Compatibles con tu Perfil")
                        
                        # Mostrar top 5 puestos sugeridos
                        for i, (position, score_data) in enumerate(suggested_positions):
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    # Emoji para el ranking
                                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                                    st.markdown(f"### {emoji} {position}")
                                
                                with col2:
                                    st.metric("Compatibilidad", f"{score_data['percentage']:.1f}%")
                                
                                with col3:
                                    st.metric("Coincidencias", f"{score_data['matches']}/{score_data['total_keywords']}")
                                
                                # Barra de progreso visual
                                progress_color = "🟢" if score_data['percentage'] >= 70 else "🟡" if score_data['percentage'] >= 50 else "🔴"
                                st.markdown(f"{progress_color} **{score_data['percentage']:.1f}% de compatibilidad**")
                                
                                st.markdown("---")
                    
                    with tab4:
                        st.subheader("Recomendaciones Personalizadas")
                        
                        # Análisis del perfil
                        total_skills = len(st.session_state.skills)
                        programming_skills = len(cv_categories.get('Lenguajes de Programación', []))
                        analysis_skills = len(cv_categories.get('Herramientas de Análisis', []))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Resumen del Perfil")
                            st.metric("Total de Habilidades", total_skills)
                            st.metric("Lenguajes de Programación", programming_skills)
                            st.metric("Herramientas de Análisis", analysis_skills)
                        
                        with col2:
                            st.markdown("#### 💡 Recomendaciones")
                            
                            # Recomendaciones basadas en el perfil
                            if programming_skills >= 3 and analysis_skills >= 2:
                                st.success("✅ **Perfil Técnico Sólido**: Tienes una buena base en programación y análisis.")
                            elif programming_skills >= 2:
                                st.info("🔵 **Enfoque en Desarrollo**: Considera fortalecer habilidades de frameworks específicos.")
                            elif analysis_skills >= 2:
                                st.info("🔵 **Enfoque en Análisis**: Considera aprender más lenguajes de programación.")
                            else:
                                st.warning("🟡 **Desarrollo de Habilidades**: Considera fortalecer tanto programación como análisis.")
                            
                            # Sugerencias específicas basadas en el mejor match
                            if suggested_positions:
                                best_match = suggested_positions[0]
                                if best_match[1]['percentage'] < 50:
                                    st.markdown("**💡 Sugerencias para mejorar:**")
                                    st.markdown("• Considera especializarte en el área con mayor compatibilidad")
                                    st.markdown("• Agrega certificaciones relevantes al puesto deseado")
                                    st.markdown("• Practica proyectos que demuestren las habilidades requeridas")
                    
                    # Limpiar archivo temporal
                    try:
                        os.unlink(resume_path)
                        print(f"🗑️ Archivo temporal eliminado: {resume_path}")
                    except Exception as e:
                        print(f"⚠️ No se pudo eliminar archivo temporal: {str(e)}")
                    
                    # Generar múltiples preguntas inmediatamente después de procesar el CV
                    try:
                        if 'questions' not in st.session_state or not st.session_state.questions:
                            st.session_state.questions = interview_bot.generate_questions(
                                st.session_state.skills, 
                                num_questions=8  # Generar 8 preguntas en lugar de 1
                            )
                            st.session_state.current_question_index = 0
                            st.session_state.current_question = st.session_state.questions[0]
                            print(f"✅ Generadas {len(st.session_state.questions)} preguntas")
                    except Exception as e:
                        st.error(f"❌ Error al generar preguntas: {str(e)}")
                        print(f"❌ Error generando preguntas: {str(e)}")
                
                elif resume_data and not resume_data.get('skills'):
                    st.warning("⚠️ CV procesado pero no se encontraron habilidades técnicas. Intente con un CV que contenga más información técnica.")
                    # Agregar algunas habilidades por defecto para continuar
                    st.session_state.skills = ["programacion", "desarrollo", "tecnologia", "sistemas"]
                    st.info("ℹ️ Se han agregado habilidades genéricas para continuar con la entrevista.")
                    
                    # Limpiar archivo temporal
                    try:
                        os.unlink(resume_path)
                    except:
                        pass
                
                else:
                    st.error("❌ Error al analizar CV. Posibles causas:")
                    st.error("• El archivo no es un PDF válido")
                    st.error("• El PDF está protegido o encriptado")
                    st.error("• El archivo está corrupto")
                    st.error("• No se pudo extraer texto del PDF")
                    
                    # Limpiar archivo temporal si existe
                    try:
                        if resume_path and os.path.exists(resume_path):
                            os.unlink(resume_path)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"❌ Error inesperado al procesar CV: {str(e)}")
                print(f"❌ Error completo: {str(e)}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                
                # Limpiar archivo temporal si existe
                try:
                    if 'resume_path' in locals() and resume_path and os.path.exists(resume_path):
                        os.unlink(resume_path)
                except:
                    pass
    
    # Display current question - SIEMPRE mostrar si existe
    if 'skills' in st.session_state and st.session_state.skills:
        # Si no hay preguntas pero hay skills, generar múltiples preguntas
        if 'questions' not in st.session_state or not st.session_state.questions:
            try:
                st.session_state.questions = interview_bot.generate_questions(
                    st.session_state.skills, 
                    num_questions=8  # Generar 8 preguntas
                )
                st.session_state.current_question_index = 0
                st.session_state.current_question = st.session_state.questions[0]
            except Exception as e:
                st.error(f"Error al generar preguntas: {str(e)}")
        
        # Mostrar la pregunta actual y el progreso
        if st.session_state.current_question:
            st.write(f"### Pregunta {st.session_state.current_question_index + 1} de {len(st.session_state.questions)}")
            st.write(st.session_state.current_question['question'])
            
            # Mostrar todas las preguntas generadas en un expander
            with st.expander("Ver todas las preguntas generadas"):
                for i, q in enumerate(st.session_state.questions):
                    status = "✅ Completada" if i < st.session_state.current_question_index else "🔄 Actual" if i == st.session_state.current_question_index else "⏳ Pendiente"
                    st.write(f"{i+1}. {q['question']} - {status}")
            
            # Botones de navegación
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Pregunta anterior") and st.session_state.current_question_index > 0:
                    st.session_state.current_question_index -= 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.rerun()
            
            with col3:
                if st.button("Pregunta siguiente ➡️") and st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    st.session_state.current_question_index += 1
                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                    st.rerun()
            
            # ALTERNATIVAS PARA DESPLIEGUE EN SERVIDOR
            st.markdown("---")
            st.markdown("### 🎥 Opciones de Grabación")
            
            # Detectar si estamos en desarrollo local o producción
            recording_mode = st.radio(
                "Selecciona el método de grabación:",
                [
                    "🎥 Grabación en tiempo real (desarrollo local)",
                    "📷 Capturar imagen con cámara web",
                    "📁 Subir video pregrabado",
                    "🖼️ Subir imagen para análisis facial"
                ],
                help="Selecciona la opción que mejor funcione en tu entorno"
            )
            
            if recording_mode == "🎥 Grabación en tiempo real (desarrollo local)":
                # FUNCIONALIDAD MEJORADA - Con control manual de inicio
                st.info("💡 Esta opción funciona mejor en desarrollo local con cámara conectada")
                
                # Botón para preparar la grabación
                if not st.session_state.ready_to_record:
                    if st.button("🎬 Preparar grabación en tiempo real", use_container_width=True):
                        st.session_state.ready_to_record = True
                        st.rerun()
                
                # Mostrar botón de inicio de grabación solo cuando esté preparado
                if st.session_state.ready_to_record and not st.session_state.is_recording:
                    st.success("✅ Sistema preparado para grabar")
                    st.markdown("""
                    <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; 
                               border-left: 4px solid #ffc107; margin: 10px 0;">
                        <h5 style="color: #856404; margin: 0;">⚠️ ¡Listo para grabar!</h5>
                        <p style="margin: 5px 0 0 0;">Cuando presiones "Iniciar Grabación", tendrás 30 segundos para responder la pregunta.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("🔴 Iniciar Grabación", use_container_width=True, type="primary"):
                            st.session_state.is_recording = True
                            st.rerun()
                    with col2:
                        if st.button("🔄 Cancelar", use_container_width=True):
                            st.session_state.ready_to_record = False
                            st.rerun()
                
                # Solo ejecutar la grabación cuando esté activa
                if st.session_state.is_recording:
                    # Mostrar indicador de grabación activa
                    recording_status = st.empty()
                    recording_status.error("🔴 GRABANDO... Por favor responde a la pregunta")
                
                    # Create placeholders for live feedback
                    emotion_placeholder = st.empty()
                    voice_placeholder = st.empty()
                
                    # Initialize video capture
                    try:
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            st.error("❌ No se pudo acceder a la cámara. Verifique que esté conectada y no esté siendo usada por otra aplicación.")
                            st.session_state.is_recording = False
                            return
                    except Exception as e:
                        st.error(f"❌ Error al inicializar la cámara: {str(e)}")
                        st.session_state.is_recording = False
                        return
                        
                    video_frames = []
                    emotion_queue = queue.Queue()
                    
                    # INICIAR GRABACIÓN DE AUDIO SIMULTÁNEAMENTE
                    print("🎙️ Iniciando grabación de audio...")
                    audio_data = sd.rec(int(30 * 44100), samplerate=44100, channels=1)
                    
                    # Record for 30 seconds
                    start_time = time.time()
                    frame_count = 0
                    while time.time() - start_time < 30 and st.session_state.is_recording:
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                            # Optimización: solo guardar cada 5to frame para reducir memoria
                            if frame_count % 5 == 0:
                                video_frames.append(frame)
                            
                            # Procesar cada 10 frames para reducir carga (optimización)
                            if frame_count % 10 == 0:
                                print(f"🔍 DEBUG: Processing frame {frame_count}")
                                # Process frame directly instead of in thread for debugging
                                process_frame(frame, emotion_queue)
                            
                            # Actualizar tiempo restante
                            elapsed_time = time.time() - start_time
                            remaining_time = 30 - elapsed_time
                            recording_status.error(f"🔴 GRABANDO... Tiempo restante: {remaining_time:.1f} segundos")
                            
                            # Display live emotion analysis
                            if not emotion_queue.empty():
                                emotion = emotion_queue.get()
                                # Safely access dominant_emotion with a fallback
                                dominant_emotion = emotion.get('dominant_emotion', 'unknown')
                                emotion_placeholder.write(f"Emocion Actual: {dominant_emotion}")
                                print(f"🔍 DEBUG: Emoción detectada en tiempo real: {dominant_emotion}")
                            
                            # Small delay to prevent overwhelming
                            time.sleep(0.1)
                    
                    print(f"🔍 DEBUG: Grabación completada. Total frames: {len(video_frames)}")
                    recording_status.success("✅ Grabación completada - Esperando audio...")
                    
                    cap.release()
                    
                    # ESPERAR A QUE TERMINE LA GRABACIÓN DE AUDIO
                    print("🎙️ Esperando finalización de grabación de audio...")
                    sd.wait()  # Esperar a que termine la grabación de audio
                    print("🎙️ Grabación de audio completada!")
                    
                    # Guardar el audio grabado
                    audio_path = save_audio(audio_data)
                    
                    # Analyze response
                    with st.spinner("Analizando tu respuesta..."):
                        analysis = analyze_response(
                            audio_path,
                            video_frames,
                            st.session_state.current_question,
                            st.session_state.skills
                        )
                        
                        # Agregar información del candidato y audio al análisis
                        analysis['audio_path'] = audio_path
                        analysis['candidate_name'] = st.session_state.candidate_name
                        analysis['target_position'] = st.session_state.target_position
                        analysis['question_number'] = st.session_state.current_question_index + 1
                        analysis['question_text'] = st.session_state.current_question['question']
                        
                        st.session_state.analysis_results.append(analysis)
                        
                        # Display analysis results
                        st.write("### Resultado del analisis")
                        
                        # MOSTRAR AUDIO PRIMERO
                        st.write("#### 🎵 Audio grabado")
                        if audio_path and os.path.exists(audio_path):
                            st.audio(audio_path, format='audio/wav')
                            st.success("▶️ Reproducir el audio completo de tu respuesta")
                        else:
                            st.warning("⚠️ No se pudo guardar el audio.")

                        # CUADRO BONITO DE SIMILITUDES
                        st.write("#### 📊 Análisis de Similitudes")
                        
                        # Crear un contenedor con estilo para las similitudes
                        with st.container():
                            # Crear columnas para mejor presentación
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("""
                                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                                    <h5 style="color: #1f77b4; margin-bottom: 15px;">🎯 Coincidencias Encontradas</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Mostrar las similitudes de manera estructurada
                                if 'matched_skills' in analysis['content_analysis'] and analysis['content_analysis']['matched_skills']:
                                    for skill in analysis['content_analysis']['matched_skills']:
                                        st.markdown(f"✅ **{skill}**")
                                else:
                                    st.markdown("ℹ️ *No se encontraron coincidencias directas*")
                            
                            with col2:
                                st.markdown("""
                                <div style="background-color: #f0f8f0; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                                    <h5 style="color: #28a745; margin-bottom: 15px;">📈 Puntuación de Similitud</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Mostrar puntuación si está disponible
                                if 'similarity_score' in analysis['content_analysis']:
                                    score = analysis['content_analysis']['similarity_score']
                                    if isinstance(score, (int, float)):
                                        score_percent = f"{score * 100:.1f}%" if score <= 1 else f"{score:.1f}%"
                                        st.markdown(f"**Similitud General:** {score_percent}")
                                        
                                        # Barra de progreso visual
                                        progress_value = score if score <= 1 else score/100
                                        st.progress(progress_value)
                                        
                                        # Interpretación del puntaje
                                        if progress_value >= 0.8:
                                            st.success("🌟 Excelente coincidencia")
                                        elif progress_value >= 0.6:
                                            st.info("👍 Buena coincidencia")
                                        elif progress_value >= 0.4:
                                            st.warning("⚠️ Coincidencia moderada")
                                        else:
                                            st.error("❌ Coincidencia baja")
                                else:
                                    st.markdown("*Analizando similitud...*")
                        
                        # NUEVAS VISUALIZACIONES BONITAS
                        st.markdown("---")
                        
                        # Gráfica comparativa CV vs Entrevista
                        st.write("#### 📊 Comparación de Habilidades: CV vs Entrevista")
                        interview_skills = analysis['content_analysis'].get('matched_skills', [])
                        fig_comparison = create_skills_comparison_chart(st.session_state.skills, interview_skills)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Tabla de emociones detectadas
                        st.write("#### 😊 Análisis de Emociones Detectadas")
                        emotion_df = create_emotion_table(analysis['emotion_analysis'])
                        
                        # Mostrar tabla bonita
                        st.markdown("""
                        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
                            <h5 style="color: #333; margin-bottom: 15px;">🎭 Emociones durante la respuesta:</h5>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Crear gráfica de barras para emociones
                        if not emotion_df.empty:
                            fig_emotions = px.bar(
                                emotion_df, 
                                x='Valor', 
                                y='Emoción', 
                                orientation='h',
                                title='Distribución de Emociones (%)',
                                color='Valor',
                                color_continuous_scale='Viridis'
                            )
                            fig_emotions.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig_emotions, use_container_width=True)
                            
                            # Tabla detallada
                            st.dataframe(
                                emotion_df[['Emoción', 'Porcentaje']], 
                                use_container_width=True, 
                                hide_index=True
                            )
                        else:
                            st.info("No se detectaron emociones en esta respuesta.")
                        
                        # Detalles adicionales en un expander
                        with st.expander("🔍 Ver detalles del análisis de contenido"):
                            st.json(analysis['content_analysis'])

                        st.write("#### 📝 Transcripción")
                        print(f"🟡 Texto transcrito: '{analysis['transcription']}'")
                        if not analysis['transcription'].strip():
                            st.warning("⚠️ No se obtuvo ninguna transcripción del audio.")
                        else:
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                                <p style="margin: 0; font-style: italic;">"{analysis['transcription']}"</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Análisis de voz en formato bonito
                        st.write("#### 🎤 Análisis de Voz")
                        if analysis['voice_analysis']:
                            voice_cols = st.columns(3)
                            voice_data = analysis['voice_analysis']
                            
                            with voice_cols[0]:
                                if 'pitch_mean' in voice_data:
                                    st.metric("Tono Promedio", f"{voice_data['pitch_mean']:.1f} Hz")
                            with voice_cols[1]:
                                if 'energy_mean' in voice_data:
                                    st.metric("Energía Promedio", f"{voice_data['energy_mean']:.3f}")
                            with voice_cols[2]:
                                if 'speaking_rate' in voice_data:
                                    st.metric("Velocidad de Habla", f"{voice_data['speaking_rate']:.1f} palabras/min")
                            
                            with st.expander("Ver análisis completo de voz"):
                                st.json(voice_data)
                        else:
                            st.info("No se pudo realizar el análisis de voz.")
                        
                        st.write("#### ⭐ Evaluación de la Respuesta")
                        if analysis['answer_evaluation']:
                            eval_data = analysis['answer_evaluation']
                            if isinstance(eval_data, dict):
                                eval_cols = st.columns(2)
                                
                                with eval_cols[0]:
                                    if 'relevance_score' in eval_data:
                                        relevance = eval_data['relevance_score']
                                        st.metric("Relevancia", f"{relevance:.1f}/10")
                                        if relevance >= 8:
                                            st.success("🌟 Muy relevante")
                                        elif relevance >= 6:
                                            st.info("👍 Relevante")
                                        else:
                                            st.warning("⚠️ Poco relevante")
                                
                                with eval_cols[1]:
                                    if 'technical_depth' in eval_data:
                                        depth = eval_data['technical_depth']
                                        st.metric("Profundidad Técnica", f"{depth:.1f}/10")
                                        if depth >= 8:
                                            st.success("🔬 Muy técnico")
                                        elif depth >= 6:
                                            st.info("🔧 Técnico")
                                        else:
                                            st.warning("📚 Básico")
                                
                                with st.expander("Ver evaluación completa"):
                                    st.json(eval_data)
                            else:
                                st.write(eval_data)
                        else:
                            st.info("No se pudo realizar la evaluación de la respuesta.")
                    
                    # Clean up - ESPERAR UN MOMENTO ANTES DE ELIMINAR
                    # No eliminar inmediatamente para que el audio se pueda reproducir
                    # El archivo se eliminará automáticamente al cerrar la sesión
                    st.session_state.is_recording = False
                    st.session_state.ready_to_record = False  # Reset para la siguiente pregunta
                    
                    # Avanzar a la siguiente pregunta
                    if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                        st.session_state.current_question_index += 1
                        st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                        st.success(f"Avanzando a la pregunta {st.session_state.current_question_index + 1}")
                        st.rerun()  # Forzar actualización para mostrar la nueva pregunta
                    else:
                        st.success("🎉 ¡Has completado todas las preguntas de la entrevista!")
                        st.balloons()
                        # Reiniciar si se desea continuar
                        if st.button("Generar nuevas preguntas"):
                            st.session_state.questions = interview_bot.generate_questions(
                                st.session_state.skills,
                                num_questions=8
                            )
                            st.session_state.current_question_index = 0
                            st.session_state.current_question = st.session_state.questions[0]
                            st.session_state.ready_to_record = False  # Reset también para nuevas preguntas
            
            elif recording_mode == "📷 Capturar imagen con cámara web":
                st.info("💡 Esta opción usa st.camera_input() - funciona en cualquier navegador")
                
                # Usar st.camera_input para capturar imagen
                camera_image = st.camera_input("📸 Captura una imagen para análisis facial")
                
                if camera_image:
                    # También grabar audio
                    if st.button("🎙️ Grabar respuesta de audio (30 segundos)", use_container_width=True):
                        with st.spinner("🎙️ Grabando audio..."):
                            audio_data = sd.rec(int(30 * 44100), samplerate=44100, channels=1)
                            
                            # Mostrar countdown
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i in range(30):
                                progress_bar.progress((i + 1) / 30)
                                status_text.text(f"⏰ Grabando... {30 - i} segundos restantes")
                                time.sleep(1)
                            
                            sd.wait()
                            status_text.text("✅ Grabación completada!")
                            
                            # Guardar audio
                            audio_path = save_audio(audio_data)
                            
                            # Procesar imagen para análisis facial
                            from PIL import Image
                            import numpy as np
                            
                            image = Image.open(camera_image)
                            # Convertir a formato OpenCV
                            image_array = np.array(image)
                            video_frames = [image_array]  # Una sola imagen como frame
                            
                            # Analizar respuesta
                            with st.spinner("Analizando respuesta..."):
                                analysis = analyze_response(
                                    audio_path,
                                    video_frames,
                                    st.session_state.current_question,
                                    st.session_state.skills
                                )
                                
                                # Agregar información del candidato
                                analysis['audio_path'] = audio_path
                                analysis['candidate_name'] = st.session_state.candidate_name
                                analysis['target_position'] = st.session_state.target_position
                                analysis['question_number'] = st.session_state.current_question_index + 1
                                analysis['question_text'] = st.session_state.current_question['question']
                                
                                st.session_state.analysis_results.append(analysis)
                                
                                # Mostrar resultados
                                st.success("✅ Análisis completado!")
                                st.audio(audio_path, format='audio/wav')
                                
                                # Avanzar pregunta
                                if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                                    st.session_state.current_question_index += 1
                                    st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                                    st.rerun()
            
            elif recording_mode == "📁 Subir video pregrabado":
                st.info("💡 Sube un video pregrabado para análisis completo")
                
                uploaded_video = st.file_uploader(
                    "Selecciona tu video de respuesta",
                    type=['mp4', 'avi', 'mov', 'mkv'],
                    help="Formatos soportados: MP4, AVI, MOV, MKV"
                )
                
                if uploaded_video:
                    # Guardar video temporalmente
                    import uuid
                    temp_video_path = f"temp_video_{uuid.uuid4().hex[:8]}.mp4"
                    
                    with open(temp_video_path, 'wb') as f:
                        f.write(uploaded_video.getvalue())
                    
                    if st.button("🔍 Analizar video subido", use_container_width=True):
                        with st.spinner("📹 Procesando video..."):
                            # Extraer frames del video
                            cap = cv2.VideoCapture(temp_video_path)
                            video_frames = []
                            frame_count = 0
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Tomar cada 30vo frame para no sobrecargar
                                if frame_count % 30 == 0:
                                    video_frames.append(frame)
                                frame_count += 1
                            
                            cap.release()
                            
                            # Para el audio, usar un placeholder (en un caso real extraerías el audio del video)
                            audio_path = None  # Podrías extraer audio del video aquí
                            
                            # Analizar
                            analysis = analyze_response(
                                audio_path,
                                video_frames,
                                st.session_state.current_question,
                                st.session_state.skills
                            )
                            
                            # Agregar información
                            analysis['candidate_name'] = st.session_state.candidate_name
                            analysis['target_position'] = st.session_state.target_position
                            analysis['question_number'] = st.session_state.current_question_index + 1
                            analysis['question_text'] = st.session_state.current_question['question']
                            
                            st.session_state.analysis_results.append(analysis)
                            
                            st.success("✅ Video analizado!")
                            st.video(uploaded_video)
                            
                            # Limpiar archivo temporal
                            try:
                                os.unlink(temp_video_path)
                            except:
                                pass
                            
                            # Avanzar pregunta
                            if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                                st.session_state.current_question_index += 1
                                st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                                st.rerun()
            
            elif recording_mode == "🖼️ Subir imagen para análisis facial":
                st.info("💡 Sube una imagen y graba audio por separado")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    uploaded_image = st.file_uploader(
                        "📸 Sube tu imagen",
                        type=['jpg', 'jpeg', 'png'],
                        help="Formatos: JPG, JPEG, PNG"
                    )
                    
                    if uploaded_image:
                        st.image(uploaded_image, caption="Imagen para análisis facial", use_column_width=True)
                
                with col2:
                    if uploaded_image:
                        if st.button("🎙️ Grabar audio de respuesta", use_container_width=True):
                            with st.spinner("🎙️ Grabando audio por 30 segundos..."):
                                audio_data = sd.rec(int(30 * 44100), samplerate=44100, channels=1)
                                
                                # Countdown
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i in range(30):
                                    progress_bar.progress((i + 1) / 30)
                                    status_text.text(f"⏰ {30 - i} segundos restantes")
                                    time.sleep(1)
                                
                                sd.wait()
                                status_text.text("✅ Audio grabado!")
                                
                                # Procesar imagen y audio
                                audio_path = save_audio(audio_data)
                                
                                # Convertir imagen
                                from PIL import Image
                                image = Image.open(uploaded_image)
                                image_array = np.array(image)
                                video_frames = [image_array]
                                
                                # Analizar
                                with st.spinner("🔍 Analizando imagen y audio..."):
                                    analysis = analyze_response(
                                        audio_path,
                                        video_frames,
                                        st.session_state.current_question,
                                        st.session_state.skills
                                    )
                                    
                                    # Agregar información
                                    analysis['audio_path'] = audio_path
                                    analysis['candidate_name'] = st.session_state.candidate_name
                                    analysis['target_position'] = st.session_state.target_position
                                    analysis['question_number'] = st.session_state.current_question_index + 1
                                    analysis['question_text'] = st.session_state.current_question['question']
                                    
                                    st.session_state.analysis_results.append(analysis)
                                    
                                    st.success("✅ Análisis completado!")
                                    st.audio(audio_path, format='audio/wav')
                                    
                                    # Avanzar pregunta
                                    if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                                        st.session_state.current_question_index += 1
                                        st.session_state.current_question = st.session_state.questions[st.session_state.current_question_index]
                                        st.rerun()
    
    # Display interview history - ORGANIZADO Y BONITO
    if st.session_state.analysis_results:
        st.markdown("---")
        st.header(f"📋 Historial de Entrevista - {st.session_state.candidate_name}")
        st.markdown(f"**Puesto:** {st.session_state.target_position} | **Total de respuestas:** {len(st.session_state.analysis_results)}")
        
        for i, result in enumerate(st.session_state.analysis_results):
            # Header bonito para cada respuesta
            question_num = result.get('question_number', i+1)
            with st.expander(f"🎯 Pregunta {question_num} - Respuesta {i+1}", expanded=False):
                
                # Pregunta en un contenedor destacado
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; 
                           border-left: 4px solid #4a90e2; margin-bottom: 15px;">
                    <h5 style="color: #2c5aa0; margin: 0;">❓ Pregunta:</h5>
                    <p style="margin: 5px 0 0 0; font-size: 16px;">{result.get('question_text', 'Pregunta no disponible')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Audio en contenedor bonito
                if 'audio_path' in result and result['audio_path'] and os.path.exists(result['audio_path']):
                    st.markdown("""
                    <div style="background-color: #fff5f5; padding: 15px; border-radius: 8px; 
                               border-left: 4px solid #e74c3c; margin-bottom: 15px;">
                        <h5 style="color: #c0392b; margin: 0 0 10px 0;">🎵 Audio de la Respuesta:</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(result['audio_path'], format='audio/wav')
                
                # Similitudes en contenedor elegante
                st.markdown("""
                <div style="background-color: #f8fff8; padding: 15px; border-radius: 8px; 
                           border-left: 4px solid #27ae60; margin-bottom: 15px;">
                    <h5 style="color: #1e8449; margin: 0 0 15px 0;">📊 Análisis de Similitudes:</h5>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**🎯 Coincidencias:**")
                    if 'matched_skills' in result['content_analysis'] and result['content_analysis']['matched_skills']:
                        for skill in result['content_analysis']['matched_skills']:
                            st.markdown(f"✅ **{skill}**")
                    else:
                        st.markdown("ℹ️ *Sin coincidencias directas*")
                
                with col2:
                    st.markdown("**📈 Puntuación:**")
                    if 'similarity_score' in result['content_analysis']:
                        score = result['content_analysis']['similarity_score']
                        if isinstance(score, (int, float)):
                            score_percent = f"{score * 100:.1f}%" if score <= 1 else f"{score:.1f}%"
                            st.markdown(f"**Similitud:** {score_percent}")
                            progress_value = score if score <= 1 else score/100
                            st.progress(progress_value)
                            
                            # Interpretación del puntaje
                            if progress_value >= 0.8:
                                st.success("🌟 Excelente coincidencia")
                            elif progress_value >= 0.6:
                                st.info("👍 Buena coincidencia")
                            elif progress_value >= 0.4:
                                st.warning("⚠️ Coincidencia moderada")
                            else:
                                st.error("❌ Coincidencia baja")
                
                # VISUALIZACIONES MEJORADAS EN EL HISTORIAL
                
                # Gráfica de emociones para esta respuesta
                st.markdown("**😊 Emociones Detectadas:**")
                emotion_df_hist = create_emotion_table(result['emotion_analysis'])
                if not emotion_df_hist.empty:
                    # Mini gráfica de emociones
                    fig_emotions_mini = px.pie(
                        emotion_df_hist, 
                        values='Valor', 
                        names='Emoción',
                        title=f'Emociones - Respuesta {question_num}'
                    )
                    fig_emotions_mini.update_layout(height=300, showlegend=True)
                    st.plotly_chart(fig_emotions_mini, use_container_width=True)
                else:
                    st.info("No se detectaron emociones en esta respuesta.")
                
                # Transcripción y evaluación en contenedores separados
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("""
                    <div style="background-color: #fefefe; padding: 15px; border-radius: 8px; 
                               border: 1px solid #ddd; margin-bottom: 10px;">
                        <h5 style="color: #555; margin: 0 0 10px 0;">📝 Transcripción:</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    if result['transcription']:
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #007bff;">
                            <p style="margin: 0; font-style: italic;">"{result['transcription']}"</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No hay transcripción disponible.")
                
                with col2:
                    st.markdown("""
                    <div style="background-color: #fefefe; padding: 15px; border-radius: 8px; 
                               border: 1px solid #ddd; margin-bottom: 10px;">
                        <h5 style="color: #555; margin: 0 0 10px 0;">⭐ Evaluación:</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar evaluación de forma bonita
                    if result['answer_evaluation']:
                        eval_data = result['answer_evaluation']
                        if isinstance(eval_data, dict):
                            if 'relevance_score' in eval_data:
                                relevance = eval_data['relevance_score']
                                st.metric("Relevancia", f"{relevance:.1f}/10")
                            if 'technical_depth' in eval_data:
                                depth = eval_data['technical_depth']
                                st.metric("Profundidad Técnica", f"{depth:.1f}/10")
                        else:
                            st.write(eval_data)
                    else:
                        st.info("No hay evaluación disponible.")
                
                # Información adicional del análisis
                st.markdown("---")
                st.markdown("**🔍 Análisis Detallado:**")
                
                # Usar tabs en lugar de expander anidado
                tab1, tab2, tab3 = st.tabs(["🎤 Voz", "😊 Emociones", "📊 Contenido"])
                
                with tab1:
                    if result['voice_analysis']:
                        voice_data = result['voice_analysis']
                        voice_cols_hist = st.columns(3)
                        
                        with voice_cols_hist[0]:
                            if 'pitch_mean' in voice_data:
                                st.metric("Tono", f"{voice_data['pitch_mean']:.1f} Hz")
                        with voice_cols_hist[1]:
                            if 'energy_mean' in voice_data:
                                st.metric("Energía", f"{voice_data['energy_mean']:.3f}")
                        with voice_cols_hist[2]:
                            if 'speaking_rate' in voice_data:
                                st.metric("Velocidad", f"{voice_data['speaking_rate']:.1f} p/min")
                        
                        st.json(voice_data)
                    else:
                        st.info("No hay análisis de voz disponible.")
                
                with tab2:
                    if result['emotion_analysis']:
                        st.dataframe(emotion_df_hist[['Emoción', 'Porcentaje']], use_container_width=True, hide_index=True)
                        st.json(result['emotion_analysis'])
                    else:
                        st.info("No hay análisis de emociones disponible.")
                
                with tab3:
                    if result['content_analysis']:
                        # Mostrar habilidades mencionadas
                        if 'matched_skills' in result['content_analysis']:
                            matched = result['content_analysis']['matched_skills']
                            if matched:
                                st.write("**Habilidades mencionadas:**")
                                for skill in matched:
                                    st.markdown(f"• {skill}")
                        st.json(result['content_analysis'])
                    else:
                        st.info("No hay análisis de contenido disponible.")

if __name__ == "__main__":
    main() 