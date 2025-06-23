import speech_recognition as sr
import os
from typing import Dict, Optional
import tempfile
import numpy as np
import time

class SpeechToText:
    def __init__(self, model_size: str = "base"):
        """
        Initialize the speech-to-text converter.
        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large")
        """
        # Use SpeechRecognition library instead of Whisper for better Windows compatibility
        print("Initializing SpeechRecognition for Windows compatibility")
        self.recognizer = sr.Recognizer()
        self.model_available = True

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio file to text.
        Args:
            audio_path: Path to the audio file
        Returns:
            Dictionary containing transcription results
        """
        if not self.model_available:
            print("Speech-to-text model not available")
            return {
                'text': '',
                'segments': [],
                'language': 'es'
            }
            
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return {
                'text': '',
                'segments': [],
                'language': 'es'
            }

        try:
            # Transcribe audio using SpeechRecognition
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                
                # Try to recognize using Google's service for better results
                try:
                    text = self.recognizer.recognize_google(audio_data, language="es-ES")
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                    text = ""
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    text = ""
            
            # Create a simple segment (since SpeechRecognition doesn't provide segments)
            segments_list = []
            if text:
                segments_list = [{
                    'start': 0.0,
                    'end': 30.0,  # Assuming a 30-second clip
                    'text': text
                }]
            
            return {
                'text': text.strip(),
                'segments': segments_list,
                'language': 'es'
            }
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return {
                'text': '',
                'segments': [],
                'language': 'es'
            }

    def get_word_timestamps(self, segments: list) -> list:
        """
        Extract word-level timestamps from segments.
        Args:
            segments: List of segments from transcription
        Returns:
            List of words with their timestamps
        """
        words = []
        for segment in segments:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            
            # Simple word splitting (can be improved with better tokenization)
            word_list = text.split()
            duration = end - start
            word_duration = duration / len(word_list)
            
            for i, word in enumerate(word_list):
                word_start = start + (i * word_duration)
                word_end = word_start + word_duration
                words.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end
                })
        
        return words 

# Asegurarse de que la clase esté disponible para importación
if __name__ != "__main__":
    # Esto garantiza que la clase esté disponible cuando se importa el módulo
    __all__ = ['SpeechToText'] 