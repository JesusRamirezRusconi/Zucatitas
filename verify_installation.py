#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias estén instaladas correctamente
Compatible con Python 3.9
"""

import sys
import importlib
from packaging import version

def check_package(package_name, min_version=None, display_name=None):
    """
    Verifica si un paquete está instalado y opcionalmente su versión
    """
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        
        if min_version and hasattr(module, '__version__'):
            installed_version = module.__version__
            if version.parse(installed_version) >= version.parse(min_version):
                print(f"✓ {display_name}: {installed_version} (OK)")
                return True
            else:
                print(f"✗ {display_name}: {installed_version} (Se requiere >= {min_version})")
                return False
        else:
            ver_str = getattr(module, '__version__', 'Desconocida')
            print(f"✓ {display_name}: {ver_str}")
            return True
            
    except ImportError as e:
        print(f"✗ {display_name}: No instalado ({e})")
        return False
    except Exception as e:
        print(f"⚠ {display_name}: Error al verificar ({e})")
        return False

def main():
    print("=== Verificación de instalación ===")
    print(f"Python: {sys.version}")
    print()
    
    # Lista de paquetes críticos a verificar
    packages_to_check = [
        # Paquetes base
        ('numpy', '1.20.0', 'NumPy'),
        ('scipy', '1.8.0', 'SciPy'),
        ('pandas', '1.3.0', 'Pandas'),
        ('sklearn', '1.0.0', 'Scikit-learn'),
        
        # Frameworks de ML
        ('torch', '2.0.0', 'PyTorch'),
        ('tensorflow', '2.10.0', 'TensorFlow'),
        
        # Visión por computadora
        ('cv2', None, 'OpenCV'),
        ('PIL', None, 'Pillow'),
        
        # Audio
        ('librosa', '0.9.0', 'Librosa'),
        ('soundfile', None, 'SoundFile'),
        ('sounddevice', None, 'SoundDevice'),
        
        # NLP
        ('spacy', '3.4.0', 'spaCy'),
        ('transformers', '4.20.0', 'Transformers'),
        
        # Reconocimiento facial
        ('deepface', None, 'DeepFace'),
        ('mtcnn', None, 'MTCNN'),
        
        # Speech
        ('whisper', None, 'OpenAI Whisper'),
        ('speech_recognition', None, 'SpeechRecognition'),
        
        # Web frameworks
        ('streamlit', '1.20.0', 'Streamlit'),
        ('flask', '2.0.0', 'Flask'),
        ('fastapi', '0.70.0', 'FastAPI'),
        
        # LangChain
        ('langchain', None, 'LangChain'),
        
        # Utilidades
        ('requests', '2.25.0', 'Requests'),
        ('tqdm', '4.60.0', 'tqdm'),
        ('click', '8.0.0', 'Click'),
        
        # Procesamiento de texto
        ('fuzzywuzzy', None, 'FuzzyWuzzy'),
        ('pdfminer', None, 'PDFMiner'),
    ]
    
    success_count = 0
    total_count = len(packages_to_check)
    
    print("Verificando paquetes principales:")
    print("-" * 50)
    
    for package_info in packages_to_check:
        if len(package_info) == 3:
            package, min_ver, display = package_info
        else:
            package, min_ver = package_info
            display = package
            
        if check_package(package, min_ver, display):
            success_count += 1
    
    print()
    print("-" * 50)
    print(f"Resumen: {success_count}/{total_count} paquetes verificados exitosamente")
    
    if success_count == total_count:
        print("🎉 ¡Todas las dependencias están instaladas correctamente!")
        return True
    elif success_count >= total_count * 0.8:
        print("⚠ La mayoría de las dependencias están instaladas. Algunos paquetes opcionales pueden faltar.")
        return True
    else:
        print("✗ Faltan dependencias críticas. Por favor, ejecute el script de instalación.")
        return False

def test_basic_functionality():
    """Prueba funcionalidad básica de los paquetes principales"""
    print("\n=== Pruebas de funcionalidad básica ===")
    
    tests = []
    
    # Test NumPy
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
        tests.append(("✓ NumPy: Operaciones básicas", True))
    except Exception as e:
        tests.append(("✗ NumPy: Error en operaciones básicas", False))
    
    # Test Pandas
    try:
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert len(df) == 2
        tests.append(("✓ Pandas: Creación de DataFrame", True))
    except Exception as e:
        tests.append(("✗ Pandas: Error en creación de DataFrame", False))
    
    # Test OpenCV
    try:
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tests.append(("✓ OpenCV: Conversión de color", True))
    except Exception as e:
        tests.append(("✗ OpenCV: Error en conversión de color", False))
    
    # Test PyTorch
    try:
        import torch
        x = torch.tensor([1, 2, 3])
        y = x + 1
        assert y.sum().item() == 9
        tests.append(("✓ PyTorch: Operaciones con tensores", True))
    except Exception as e:
        tests.append(("✗ PyTorch: Error en operaciones con tensores", False))
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        x = tf.constant([1, 2, 3])
        y = tf.add(x, 1)
        tests.append(("✓ TensorFlow: Operaciones básicas", True))
    except Exception as e:
        tests.append(("✗ TensorFlow: Error en operaciones básicas", False))
    
    # Test Streamlit (solo importación)
    try:
        import streamlit as st
        tests.append(("✓ Streamlit: Importación exitosa", True))
    except Exception as e:
        tests.append(("✗ Streamlit: Error en importación", False))
    
    print()
    for test_result, success in tests:
        print(test_result)
    
    successful_tests = sum(1 for _, success in tests if success)
    total_tests = len(tests)
    
    print(f"\nPruebas exitosas: {successful_tests}/{total_tests}")
    
    return successful_tests >= total_tests * 0.8

if __name__ == "__main__":
    installation_ok = main()
    
    if installation_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 ¡Sistema listo para usar!")
            sys.exit(0)
        else:
            print("\n⚠ Algunos componentes pueden no funcionar correctamente.")
            sys.exit(1)
    else:
        print("\n✗ Por favor, instale las dependencias faltantes antes de continuar.")
        sys.exit(1)

