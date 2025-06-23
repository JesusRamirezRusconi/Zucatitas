import librosa
import numpy as np
from typing import Dict, List, Tuple
import os

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate
        self.n_mfcc = 13  # Number of MFCC features

    def extract_features(self, audio_path: str) -> Dict:
        """
        Extract audio features from the audio file.
        Returns a dictionary containing various audio features.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = {
                'mfcc': self._extract_mfcc(y, sr),
                'pitch': self._extract_pitch(y, sr),
                'energy': self._extract_energy(y),
                'tempo': self._extract_tempo(y, sr),
                'duration': librosa.get_duration(y=y, sr=sr)
            }
            
            return features
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return {}

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features."""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        return np.mean(mfccs, axis=1)

    def _extract_pitch(self, y: np.ndarray, sr: int) -> float:
        """Extract pitch (fundamental frequency)."""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)])
        return float(pitch_mean)

    def _extract_energy(self, y: np.ndarray) -> float:
        """Extract energy (RMS)."""
        return float(np.mean(librosa.feature.rms(y=y)))

    def _extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """Extract tempo (BPM)."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Use the updated API for librosa >= 0.10.0
            tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
            return float(tempo[0]) if len(tempo) > 0 else 120.0
        except Exception:
            # Fallback to default tempo if extraction fails
            return 120.0

    def analyze_voice_characteristics(self, features: Dict) -> Dict:
        """
        Analyze voice characteristics based on extracted features.
        Returns a dictionary with voice analysis results.
        """
        if not features:
            return {}

        analysis = {
            'voice_energy': 'high' if features['energy'] > 0.1 else 'low',
            'speaking_tempo': 'fast' if features['tempo'] > 120 else 'slow',
            'pitch_level': 'high' if features['pitch'] > 200 else 'low',
            'duration_seconds': features['duration']
        }

        return analysis 