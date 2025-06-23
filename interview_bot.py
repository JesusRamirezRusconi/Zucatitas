from typing import List, Dict
import random

class InterviewBot:
    def __init__(self):
        # Common interview questions by category
        self.question_templates = {
            'behavioral': [
                "Cuéntame sobre una ocasión en la que {scenario}.",
                "Describe una situación donde tuviste que {scenario}.",
                "Dame un ejemplo de cuando {scenario}.",
                "Comparte un caso específico en el que {scenario}."
            ],
            'technical': [
                "¿Cómo abordarías {scenario}?",
                "Explica tu experiencia con {scenario}.",
                "¿Cuál es tu comprensión sobre {scenario}?",
                "¿Cómo manejas {scenario}?"
            ],
            'situational': [
                "¿Qué harías si {scenario}?",
                "¿Cómo manejarías una situación donde {scenario}?",
                "Si te enfrentas a {scenario}, ¿cuál sería tu enfoque?",
                "Imagina que estás en una situación donde {scenario}. ¿Cómo responderías?"
            ]
        }
        
        # Common scenarios based on skills
        self.skill_scenarios = {
            'leadership': [
                "lideraste un equipo durante un proyecto desafiante",
                "tuviste que tomar una decisión difícil que afectó a tu equipo",
                "motivaste a miembros del equipo en un momento complicado",
                "resolviste un conflicto entre miembros del equipo"
            ],
            'problem_solving': [
                "enfrentaste un problema técnico complejo",
                "tuviste que depurar un error crítico bajo presión",
                "optimizaste un sistema de bajo rendimiento",
                "encontraste una solución creativa a un problema desafiante"
            ],
            'communication': [
                "explicaste un concepto técnico complejo a personas no técnicas",
                "presentaste tu trabajo a la alta dirección",
                "escribiste documentación técnica",
                "colaboraste con diferentes equipos"
            ],
            'technical': [
                "trabajaste con una nueva tecnología o framework",
                "tuviste que aprender un nuevo lenguaje de programación rápidamente",
                "diseñaste un sistema escalable",
                "implementaste una funcionalidad compleja"
            ]
        }

    def generate_questions(self, skills: List[str], num_questions: int = 5) -> List[Dict]:
        """
        Generate interview questions based on skills.
        Args:
            skills: List of skills from resume
            num_questions: Number of questions to generate
        Returns:
            List of questions with their categories and context
        """
        questions = []
        used_scenarios = set()
        
        # Map skills to question categories
        skill_categories = self._categorize_skills(skills)
        
        # Asegurarse de que siempre tengamos al menos una categoría
        if not skill_categories:
            skill_categories = ['technical']
        
        # Intentar generar preguntas basadas en habilidades
        attempts = 0
        max_attempts = 20  # Limitar intentos para evitar bucles infinitos
        
        while len(questions) < num_questions and attempts < max_attempts:
            attempts += 1
            
            # Select a random category
            category = random.choice(list(self.question_templates.keys()))
            
            # Get relevant scenarios for the skills
            relevant_scenarios = []
            for skill_category in skill_categories:
                if skill_category in self.skill_scenarios:
                    relevant_scenarios.extend(self.skill_scenarios[skill_category])
            
            # Si no hay escenarios relevantes, usar escenarios técnicos por defecto
            if not relevant_scenarios:
                relevant_scenarios = self.skill_scenarios['technical']
                
            # Select a random scenario
            scenario = random.choice(relevant_scenarios)
            if scenario in used_scenarios and len(used_scenarios) < len(relevant_scenarios):
                continue
                
            used_scenarios.add(scenario)
            
            # Generate question
            template = random.choice(self.question_templates[category])
            question = template.format(scenario=scenario)
            
            questions.append({
                'question': question,
                'category': category,
                'context': scenario
            })
        
        # Si después de todos los intentos no se generaron preguntas, crear una pregunta genérica
        if not questions:
            category = 'technical'
            scenario = "trabajaste con tecnologías de programación"
            template = self.question_templates[category][0]
            question = template.format(scenario=scenario)
            
            questions.append({
                'question': question,
                'category': category,
                'context': scenario
            })
        
        return questions

    def _categorize_skills(self, skills: List[str]) -> List[str]:
        """Categorize skills into general categories."""
        categories = set()
        
        # Simple categorization based on keywords
        for skill in skills:
            skill_lower = str(skill).lower()
            if any(word in skill_lower for word in ['lead', 'manage', 'team', 'fundidor']):
                categories.add('leadership')
            if any(word in skill_lower for word in ['solve', 'debug', 'optimize', 'aws', 'automation']):
                categories.add('problem_solving')
            if any(word in skill_lower for word in ['communicate', 'present', 'write', 'javascript', 'python', 'html']):
                categories.add('communication')
            if any(word in skill_lower for word in ['program', 'code', 'develop', 'design', 'python', 'javascript', 'aws', 'academy']):
                categories.add('technical')
        
        # Si no se encontró ninguna categoría, asignar 'technical' por defecto
        if not categories:
                categories.add('technical')
        
        return list(categories)

    def evaluate_answer(self, question: Dict, answer: str, skills: List[str]) -> Dict:
        """
        Evaluate an answer based on the question and expected skills.
        Args:
            question: Question dictionary
            answer: Candidate's answer
            skills: List of relevant skills
        Returns:
            Dictionary with evaluation results
        """
        # Simple evaluation based on keyword matching
        answer = answer.lower()
        relevant_skills = [skill.lower() for skill in skills]
        
        # Count how many relevant skills were mentioned
        mentioned_skills = [
            skill for skill in relevant_skills
            if skill in answer
        ]
        
        # Calculate basic metrics
        skill_coverage = len(mentioned_skills) / len(relevant_skills) if relevant_skills else 0
        
        return {
            'mentioned_skills': mentioned_skills,
            'skill_coverage': skill_coverage,
            'answer_length': len(answer.split()),
            'evaluation': {
                'completeness': 'high' if skill_coverage > 0.7 else 'medium' if skill_coverage > 0.3 else 'low',
                'relevance': 'high' if len(mentioned_skills) > 0 else 'low'
            }
        } 


        