# Importar el puente para LocallyConnected2D
import keras_bridge
keras_bridge.install_patch()

from deepface import DeepFace
import cv2
import numpy as np
from typing import List, Dict
import os

class FacialEmotionAnalyzer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze emotions in a single frame.
        Args:
            frame: BGR image frame from webcam
        Returns:
            Dictionary with emotion analysis results
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"🔍 DEBUG: Frame shape: {frame_rgb.shape}")
            
            # Analyze the frame
            analysis = DeepFace.analyze(
                frame_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            print(f"🔍 DEBUG: DeepFace analysis result: {analysis}")
            print(f"🔍 DEBUG: Dominant emotion: {analysis.get('dominant_emotion', 'not found')}")
            print(f"🔍 DEBUG: All emotions: {analysis.get('emotion', {})}")
            
            return {
                'dominant_emotion': analysis['dominant_emotion'],
                'emotions': analysis['emotion']
            }
            
        except Exception as e:
            print(f"❌ ERROR analyzing frame: {str(e)}")
            print(f"❌ ERROR type: {type(e)}")
            import traceback
            print(f"❌ ERROR traceback: {traceback.format_exc()}")
            return {
                'dominant_emotion': 'neutral',
                'emotions': {emotion: 0.0 for emotion in self.emotions}
            }

    def analyze_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze emotions across multiple frames.
        Args:
            frames: List of BGR image frames
        Returns:
            Dictionary with emotion analysis summary
        """
        results = []
        for frame in frames:
            result = self.analyze_frame(frame)
            results.append(result)
        
        return self.get_emotion_summary(results)

    def get_emotion_summary(self, results: List[Dict]) -> Dict:
        """
        Summarize emotion analysis results.
        Args:
            results: List of emotion analysis results
        Returns:
            Dictionary with emotion statistics
        """
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        
        for result in results:
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            emotion_counts[dominant_emotion] += 1
        
        total_frames = len(results)
        if total_frames == 0:
            return {}
            
        emotion_percentages = {
            emotion: (count / total_frames) * 100
            for emotion, count in emotion_counts.items()
        }
        
        # Calculate average emotion values
        avg_emotions = {emotion: 0.0 for emotion in self.emotions}
        for result in results:
            if 'emotions' in result:
                for emotion, value in result['emotions'].items():
                    avg_emotions[emotion] += value / total_frames
        
        return {
            'emotion_percentages': emotion_percentages,
            'average_emotions': avg_emotions,
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0],
            'total_frames_analyzed': total_frames
        } 