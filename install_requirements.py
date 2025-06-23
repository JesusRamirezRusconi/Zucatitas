#!/usr/bin/env python3
"""
Script para instalar dependencias en orden correcto para evitar conflictos
Compatible con Python 3.9
"""

import subprocess
import sys
import os

def run_pip_install(packages, upgrade=False):
    """Instala paquetes con pip"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    
    print(f"Ejecutando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Instalado exitosamente: {', '.join(packages)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error instalando {', '.join(packages)}: {e}")
        return False

def main():
    print("=== Instalación de dependencias para Python 3.9 ===")
    print("Verificando versión de Python...")
    
    # Verificar versión de Python
    if sys.version_info < (3, 9) or sys.version_info >= (3, 10):
        print(f"ADVERTENCIA: Este script está optimizado para Python 3.9")
        print(f"Versión actual: {sys.version}")
        response = input("¿Continuar de todos modos? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Actualizar pip, setuptools y wheel primero
    print("\n1. Actualizando herramientas básicas...")
    if not run_pip_install(["pip>=23.0", "setuptools>=68.0", "wheel>=0.40"], upgrade=True):
        print("Error actualizando herramientas básicas")
        return
    
    # Instalar dependencias base
    print("\n2. Instalando dependencias base...")
    base_packages = [
        "numpy==1.24.3",
        "six==1.16.0",
        "packaging==23.1",
        "typing-extensions==4.7.1",
        "charset-normalizer==3.2.0",
        "idna==3.4",
        "urllib3==2.0.4",
        "certifi==2023.7.22",
        "requests==2.31.0"
    ]
    if not run_pip_install(base_packages):
        print("Error instalando dependencias base")
        return
    
    # Instalar dependencias científicas
    print("\n3. Instalando librerías científicas...")
    scientific_packages = [
        "scipy==1.10.1",
        "pandas==1.5.3",
        "scikit-learn==1.2.2",
        "joblib==1.2.0",
        "sympy==1.12",
        "mpmath==1.3.0"
    ]
    if not run_pip_install(scientific_packages):
        print("Error instalando librerías científicas")
        return
    
    # Instalar frameworks de ML
    print("\n4. Instalando frameworks de Machine Learning...")
    ml_packages = [
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "torchvision==0.16.0"
    ]
    if not run_pip_install(ml_packages):
        print("Error instalando PyTorch")
        return
    
    # TensorFlow por separado
    print("\n5. Instalando TensorFlow...")
    tf_packages = [
        "tensorflow==2.13.1",
        "keras==2.13.1"
    ]
    if not run_pip_install(tf_packages):
        print("Error instalando TensorFlow")
        return
    
    # Visión por computadora
    print("\n6. Instalando librerías de visión por computadora...")
    cv_packages = [
        "opencv-python==4.8.0.76",
        "pillow==9.5.0",
        "imageio==2.31.1"
    ]
    if not run_pip_install(cv_packages):
        print("Error instalando librerías de CV")
        return
    
    # Audio
    print("\n7. Instalando librerías de audio...")
    audio_packages = [
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "sounddevice==0.4.6",
        "audioread==3.0.0",
        "soxr==0.3.7"
    ]
    if not run_pip_install(audio_packages):
        print("Error instalando librerías de audio")
        return
    
    # NLP
    print("\n8. Instalando librerías de NLP...")
    nlp_packages = [
        "spacy==3.7.2",
        "spacy-legacy==3.0.12",
        "spacy-loggers==1.0.5",
        "transformers==4.30.2",
        "tokenizers==0.13.3",
        "sentencepiece==0.1.99"
    ]
    if not run_pip_install(nlp_packages):
        print("Error instalando librerías de NLP")
        return
    
    # Reconocimiento facial
    print("\n9. Instalando librerías de reconocimiento facial...")
    face_packages = [
        "deepface==0.0.79",
        "mtcnn==0.1.1",
        "retina-face==0.0.17"
    ]
    if not run_pip_install(face_packages):
        print("Error instalando librerías de reconocimiento facial")
        return
    
    # Speech y Whisper
    print("\n10. Instalando librerías de speech...")
    speech_packages = [
        "openai-whisper==20231117",
        "SpeechRecognition==3.10.0"
    ]
    if not run_pip_install(speech_packages):
        print("Error instalando librerías de speech")
        return
    
    # Web frameworks
    print("\n11. Instalando frameworks web...")
    web_packages = [
        "flask==2.3.3",
        "flask-login==0.6.3",
        "flask-migrate==4.0.5",
        "flask-sqlalchemy==3.0.5",
        "jinja2==3.1.2",
        "markupsafe==2.1.3",
        "werkzeug==2.3.7",
        "itsdangerous==2.1.2"
    ]
    if not run_pip_install(web_packages):
        print("Error instalando frameworks web")
        return
    
    # Streamlit
    print("\n12. Instalando Streamlit...")
    streamlit_packages = [
        "streamlit==1.28.0"
    ]
    if not run_pip_install(streamlit_packages):
        print("Error instalando Streamlit")
        return
    
    # FastAPI
    print("\n13. Instalando FastAPI...")
    fastapi_packages = [
        "fastapi==0.103.1",
        "uvicorn==0.23.2",
        "gunicorn==21.2.0"
    ]
    if not run_pip_install(fastapi_packages):
        print("Error instalando FastAPI")
        return
    
    # LangChain
    print("\n14. Instalando LangChain...")
    langchain_packages = [
        "langchain==0.0.350",
        "langchain-community==0.0.20",
        "langchain-core==0.1.23",
        "langsmith==0.0.87"
    ]
    if not run_pip_install(langchain_packages):
        print("Error instalando LangChain")
        return
    
    # Utilidades restantes
    print("\n15. Instalando utilidades restantes...")
    util_packages = [
        "pdfminer.six==20221105",
        "fuzzywuzzy==0.18.0",
        "python-levenshtein==0.21.1",
        "rapidFuzz==3.2.0",
        "tqdm==4.66.1",
        "click==8.1.7",
        "rich==13.5.2",
        "colorama==0.4.6",
        "typer==0.9.0",
        "python-dateutil==2.8.2",
        "pytz==2023.3",
        "tzlocal==5.0.1",
        "watchdog==3.0.0",
        "filelock==3.12.3",
        "platformdirs==3.10.0",
        "sqlalchemy==1.4.46",
        "alembic==1.11.3",
        "pydantic==1.10.12",
        "validators==0.21.2",
        "threadpoolctl==3.2.0",
        "greenlet==2.0.2",
        "blinker==1.6.2",
        "tenacity==8.2.3",
        "zipp==3.16.2",
        "importlib-metadata==6.8.0",
        "PyYAML==6.0.1",
        "toml==0.10.2",
        "jsonschema==4.19.0",
        "jsonpatch==1.33",
        "jsonpointer==2.4",
        "aiohttp==3.8.5",
        "aiosignal==1.3.1",
        "anyio==3.7.1",
        "httplib2==0.22.0",
        "cryptography==41.0.4"
    ]
    if not run_pip_install(util_packages):
        print("Error instalando utilidades")
        return
    
    print("\n🎉 ¡Instalación completada exitosamente!")
    print("\nPara verificar la instalación, ejecute:")
    print("python -c \"import torch, tensorflow, cv2, librosa, streamlit; print('✓ Todas las librerías principales están disponibles')\"")

if __name__ == "__main__":
    main()

