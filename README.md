# Self Interview Analysis System

This system helps analyze interview performance by processing resume content, facial emotions, voice characteristics, and speech content. It provides comprehensive feedback on interview performance.

## Features

- Resume parsing and skill extraction
- Facial emotion analysis during interview
- Voice characteristics analysis
- Speech-to-text transcription
- Content matching between resume and speech
- Interview question generation
- Answer evaluation

## Requirements

- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (recommended for faster processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd self-interview-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your resume (PDF) and interview video through the web interface

4. Wait for the analysis to complete

5. Review the results:
   - Extracted skills from resume
   - Facial emotion analysis
   - Voice characteristics
   - Speech transcription
   - Content matching analysis

## Project Structure

```
project_root/
│
├── resume_parser.py      # Resume parsing and skill extraction
├── facial_emotion.py     # Facial emotion analysis
├── voice_analysis.py     # Voice characteristics analysis
├── speech_to_text.py     # Speech-to-text conversion
├── content_matcher.py    # Content matching analysis
├── interview_bot.py      # Interview question generation
├── app.py               # Main Streamlit application
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Notes

- The system requires a good quality video with clear audio for best results
- Processing time depends on video length and system specifications
- Some features may require additional setup (e.g., GPU for faster processing)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 