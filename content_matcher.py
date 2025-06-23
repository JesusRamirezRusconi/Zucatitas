from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple
import re

class ContentMatcher:
    def __init__(self, threshold: int = 70):
        """
        Initialize the content matcher.
        Args:
            threshold: Minimum similarity score (0-100) to consider a match
        """
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def match_skills(self, resume_skills: List[str], transcript: str) -> Dict[str, int]:
        """
        Match skills from resume against transcript.
        Returns a dictionary of skills and their match scores.
        """
        transcript = self.preprocess_text(transcript)
        results = {}
        
        for skill in resume_skills:
            skill = self.preprocess_text(skill)
            # Use partial ratio to catch partial matches
            score = fuzz.partial_ratio(skill, transcript)
            results[skill] = score
        
        return results

    def find_key_phrases(self, text: str, phrases: List[str]) -> List[Tuple[str, int]]:
        """
        Find key phrases in text and their positions.
        Returns list of (phrase, position) tuples.
        """
        text = self.preprocess_text(text)
        matches = []
        
        for phrase in phrases:
            phrase = self.preprocess_text(phrase)
            # Find all occurrences of the phrase
            start = 0
            while True:
                start = text.find(phrase, start)
                if start == -1:
                    break
                matches.append((phrase, start))
                start += len(phrase)
        
        return sorted(matches, key=lambda x: x[1])

    def analyze_content_match(self, resume_skills: List[str], transcript: str) -> Dict:
        """
        Analyze how well the transcript matches the resume content.
        Returns a comprehensive analysis of the match.
        """
        # Match skills
        skill_matches = self.match_skills(resume_skills, transcript)
        
        # Find matched skills above threshold
        matched_skills = {
            skill: score for skill, score in skill_matches.items()
            if score >= self.threshold
        }
        
        # Calculate match statistics
        total_skills = len(resume_skills)
        matched_count = len(matched_skills)
        match_percentage = (matched_count / total_skills * 100) if total_skills > 0 else 0
        
        return {
            'matched_skills': matched_skills,
            'unmatched_skills': {
                skill: score for skill, score in skill_matches.items()
                if score < self.threshold
            },
            'match_statistics': {
                'total_skills': total_skills,
                'matched_skills': matched_count,
                'match_percentage': match_percentage
            }
        } 