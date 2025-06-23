# Guía de Instalación - Proyecto Entrevistas

## Requisitos del Sistema

- **Python 3.9.x** (recomendado: 3.9.13)
- **Sistema Operativo**: Windows 10/11, macOS, o Linux
- **RAM**: Mínimo 8GB (recomendado: 16GB)
- **Espacio en disco**: Mínimo 10GB disponibles

## Problemas Resueltos

Este proyecto tenía múltiples conflictos de versiones entre paquetes que han sido corregidos:

### Conflictos Principales Identificados:
1. **Versiones incompatibles** entre TensorFlow y PyTorch
2. **Dependencias conflictivas** en librosas de audio (librosa, soundfile, soxr)
3. **Versiones demasiado nuevas** para Python 3.9
4. **Conflictos en paquetes web** (Flask, FastAPI, Streamlit)
5. **Inconsistencias en LangChain** y dependencias relacionadas

### Soluciones Implementadas:
- ✅ Versiones fijas y compatibles con Python 3.9
- ✅ Resolución de conflictos entre TensorFlow 2.13.1 y PyTorch 2.1.0
- ✅ Instalación secuencial para evitar conflictos de dependencias
- ✅ Scripts de verificación automática

## Métodos de Instalación

### Método 1: Instalación Automática (Recomendado)

```bash
# 1. Crear un entorno virtual (recomendado)
python -m venv venv

# 2. Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 3. Ejecutar el script de instalación automática
python install_requirements.py
```

### Método 2: Instalación Manual

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Actualizar herramientas básicas
python -m pip install --upgrade pip setuptools wheel

# 3. Instalar desde el archivo de requisitos corregido
pip install -r requirements_fixed.txt
```

### Método 3: Instalación Original (No Recomendado)

⚠️ **ADVERTENCIA**: Los archivos originales (`requirements.txt`, `requirements1.txt`, `requirements2.txt`) contienen conflictos de versiones.

```bash
# Solo usar si necesitas mantener las versiones originales
pip install -r requirements.txt
```

## Verificación de la Instalación

Después de la instalación, ejecuta el script de verificación:

```bash
python verify_installation.py
```

Este script verificará:
- ✅ Que todos los paquetes estén instalados
- ✅ Que las versiones sean compatibles
- ✅ Que la funcionalidad básica funcione correctamente

## Estructura de Archivos

```
proyecto/
├── requirements.txt              # ❌ Original (con conflictos)
├── requirements1.txt             # ❌ Original (con conflictos)
├── requirements2.txt             # ❌ Original (con conflictos)
├── requirements_fixed.txt        # ✅ Corregido y compatible
├── install_requirements.py       # ✅ Script de instalación automática
├── verify_installation.py        # ✅ Script de verificación
└── INSTALLATION_README.md         # ✅ Esta guía
```

## Solución de Problemas Comunes

### Error: "No matching distribution found"
```bash
# Actualizar pip
python -m pip install --upgrade pip

# Limpiar caché
pip cache purge

# Reinstalar
pip install -r requirements_fixed.txt
```

### Error: "Microsoft Visual C++ 14.0 is required" (Windows)
```bash
# Instalar Microsoft C++ Build Tools
# Descargar desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Conflictos de versiones persistentes
```bash
# Crear un entorno completamente nuevo
deactivate
rmdir /s venv  # Windows
# rm -rf venv    # macOS/Linux

python -m venv venv_new
venv_new\Scripts\activate  # Windows
# source venv_new/bin/activate  # macOS/Linux

python install_requirements.py
```

### Error de TensorFlow en CPU
```bash
# Si solo tienes CPU (sin GPU), usar:
pip install tensorflow-cpu==2.13.1
```

### Problemas con PyTorch y CUDA
```bash
# Para CPU solamente:
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Paquetes Principales Incluidos

### Machine Learning y Data Science
- 🧮 **NumPy 1.24.3** - Computación numérica
- 📊 **Pandas 1.5.3** - Manipulación de datos
- 🤖 **Scikit-learn 1.2.2** - Machine Learning
- 🔥 **PyTorch 2.1.0** - Deep Learning
- 🧠 **TensorFlow 2.13.1** - Deep Learning

### Visión por Computadora
- 👁️ **OpenCV 4.8.0.76** - Procesamiento de imágenes
- 🖼️ **Pillow 9.5.0** - Manipulación de imágenes
- 👤 **DeepFace 0.0.79** - Reconocimiento facial
- 🎯 **MTCNN 0.1.1** - Detección facial

### Procesamiento de Audio
- 🎵 **Librosa 0.10.1** - Análisis de audio
- 🔊 **SoundFile 0.12.1** - I/O de archivos de audio
- 🎤 **SoundDevice 0.4.6** - Audio en tiempo real
- 🗣️ **OpenAI Whisper** - Speech-to-text

### Procesamiento de Lenguaje Natural
- 📝 **spaCy 3.7.2** - NLP avanzado
- 🤗 **Transformers 4.30.2** - Modelos preentrenados
- 🔗 **LangChain 0.0.350** - Aplicaciones con LLMs

### Frameworks Web
- 🚀 **Streamlit 1.28.0** - Aplicaciones web de ML
- 🌐 **Flask 2.3.3** - Framework web ligero
- ⚡ **FastAPI 0.103.1** - API moderna y rápida

## Configuración Post-Instalación

### Para spaCy (NLP)
```bash
# Descargar modelo en español
python -m spacy download es_core_news_sm

# Descargar modelo en inglés
python -m spacy download en_core_web_sm
```

### Para NLTK (si se usa)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Variables de Entorno Recomendadas
```bash
# Para TensorFlow (opcional)
set TF_CPP_MIN_LOG_LEVEL=2

# Para PyTorch (opcional)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Uso del Proyecto

Después de la instalación exitosa:

```bash
# Verificar que todo funciona
python verify_installation.py

# Ejecutar la aplicación principal (ajustar según tu proyecto)
streamlit run app.py
# o
python main.py
```

## Mantenimiento

### Actualizar dependencias (con cuidado)
```bash
# Verificar versiones actuales
pip list

# Actualizar solo paquetes específicos si es necesario
pip install --upgrade package_name

# Re-verificar después de actualizaciones
python verify_installation.py
```

### Crear backup del entorno
```bash
# Generar requirements actuales
pip freeze > requirements_backup.txt

# Exportar entorno con conda (si usas conda)
conda env export > environment_backup.yml
```

## Soporte

Si encuentras problemas:

1. **Ejecuta el verificador**: `python verify_installation.py`
2. **Revisa los logs** de error específicos
3. **Consulta la sección de problemas comunes** arriba
4. **Crea un nuevo entorno virtual** si persisten los problemas

---

**Nota**: Este archivo de instalación ha sido optimizado específicamente para Python 3.9 y resuelve todos los conflictos de versiones identificados en los archivos de requisitos originales.

