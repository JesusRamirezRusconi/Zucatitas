from pdfminer.high_level import extract_text
import spacy
import os
from typing import Dict, List, Optional

class ResumeParser:
    def __init__(self):
        """Initialize the resume parser with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            os.system("python -m spacy download es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm")

    def extract_resume_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF resume.
        Args:
            pdf_path: Path to the PDF file
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(pdf_path):
                print(f"❌ Error: El archivo no existe: {pdf_path}")
                return None
            
            # Verificar que el archivo no está vacío
            if os.path.getsize(pdf_path) == 0:
                print(f"❌ Error: El archivo está vacío: {pdf_path}")
                return None
            
            print(f"🔍 DEBUG: Extrayendo texto de: {pdf_path}")
            text = extract_text(pdf_path)
            
            if text and len(text.strip()) > 0:
                print(f"✅ Texto extraído exitosamente ({len(text)} caracteres)")
                return text
            else:
                print(f"⚠️ Warning: No se pudo extraer texto del PDF o el texto está vacío")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {str(e)}")
            import traceback
            print(f"❌ Traceback completo: {traceback.format_exc()}")
            return None

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from resume text.
        Args:
            text: Resume text
        Returns:
            List of extracted skills
        """
        # Common technical skills to look for (perfil de TI completo)
        technical_skills = [
            # Lenguajes de programación
            "python", "java", "javascript", "c++", "c#", "ruby", "php", "go", "rust",
            "typescript", "scala", "r", "matlab", "perl", "lua", "dart", "elixir",
            
            # Frontend
            "html", "css", "react", "angular", "vue", "vue.js", "svelte", "jquery",
            "bootstrap", "tailwind", "sass", "less", "webpack", "vite", "parcel",
            
            # Backend
            "node.js", "django", "flask", "spring", "express", "fastapi", "laravel",
            "rails", "asp.net", "struts", "hibernate", "spring boot", "nest.js",
            
            # Bases de datos
            "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "redis",
            "elasticsearch", "cassandra", "dynamodb", "sqlite", "mariadb",
            
            # Cloud y DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ansible",
            "terraform", "vagrant", "helm", "istio", "openshift", "heroku",
            "digitalocean", "linode", "cloudflare", "serverless", "lambda",
            
            # Metodologías y herramientas
            "git", "github", "gitlab", "bitbucket", "agile", "scrum", "kanban",
            "jira", "confluence", "slack", "teams", "notion", "trello",
            
            # Data Science y AI
            "machine learning", "ai", "artificial intelligence", "data science", 
            "big data", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
            "jupyter", "anaconda", "spark", "hadoop", "kafka", "airflow",
            "tableau", "power bi", "excel", "statistics", "deep learning",
            
            # APIs y arquitectura
            "rest api", "graphql", "microservices", "api", "json", "xml",
            "soap", "grpc", "websockets", "oauth", "jwt", "api gateway",
            
            # Sistemas y redes
            "linux", "unix", "bash", "shell scripting", "networking", "tcp/ip",
            "dns", "load balancing", "nginx", "apache", "iis", "firewall",
            
            # Seguridad
            "security", "cybersecurity", "ssl", "tls", "encryption", "penetration testing",
            "vulnerability assessment", "owasp", "authentication", "authorization",
            
            # Desarrollo móvil
            "mobile development", "ios", "android", "swift", "kotlin", "flutter",
            "react native", "xamarin", "ionic", "cordova", "objective-c",
            
            # Testing
            "testing", "unit testing", "integration testing", "selenium", "jest",
            "pytest", "junit", "cypress", "postman", "test automation",
            
            # Blockchain y emerging tech
            "blockchain", "ethereum", "bitcoin", "smart contracts", "solidity",
            "web3", "nft", "defi", "cryptocurrency", "iot", "edge computing",
            
            # Soft skills técnicos
            "problem solving", "debugging", "code review", "documentation",
            "technical writing", "mentoring", "team leadership", "project management",
            
            # Herramientas específicas
            "visual studio", "vscode", "intellij", "eclipse", "xcode", "android studio",
            "figma", "sketch", "photoshop", "illustrator", "blender", "unity",
            
            # Frameworks adicionales
            "spring framework", "hibernate", "struts", "express.js", "koa",
            "fastify", "gin", "fiber", "echo", "chi", "gorilla", "beego"
        ]
        
        # Convert text to lowercase for better matching
        text_lower = text.lower()
        
        # Extract named entities
        doc = self.nlp(text)
        entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 2]
        
        # Find skills from predefined list (búsqueda mejorada)
        found_skills = []
        for skill in technical_skills:
            # Buscar tanto la habilidad exacta como variaciones
            skill_variations = [
                skill,
                skill.replace(" ", ""),  # sin espacios
                skill.replace(".", ""),  # sin puntos
                skill.replace("-", ""),  # sin guiones
                skill.replace("_", "")   # sin guiones bajos
            ]
            
            for variation in skill_variations:
                if variation in text_lower:
                    found_skills.append(skill)
                    break  # Solo agregar una vez por habilidad
        
        # Add any additional relevant skills found in entities
        relevant_entities = []
        for entity in entities:
            # Filtrar entidades que parezcan tecnológicas
            if (len(entity) > 2 and 
                (any(tech_word in entity for tech_word in ['dev', 'tech', 'soft', 'data', 'web', 'app', 'sys']) or
                 entity.endswith(('js', 'py', 'sql', 'api', 'db', 'os', 'ui', 'ux')) or
                 any(char.isdigit() for char in entity) and len(entity) < 10)):  # versiones como "python3", "node16"
                relevant_entities.append(entity)
        
        # Combinar habilidades encontradas y entidades relevantes
        all_skills = found_skills + relevant_entities
        
        # Eliminar duplicados y asegurar que hay al menos algunas habilidades
        unique_skills = list(set(all_skills))
        
        # Si no se encontraron habilidades, intentar extraer palabras técnicas comunes
        if len(unique_skills) < 2:
            print("🔍 DEBUG: Pocas habilidades encontradas, buscando términos técnicos...")
            technical_terms = []
            words = text_lower.split()
            for word in words:
                word_clean = word.strip('.,!?;:"()[]{}')
                if (len(word_clean) > 3 and 
                    (word_clean.endswith(('ing', 'ment', 'tion', 'ness', 'able', 'ful')) or
                     word_clean in ['desarrollo', 'programacion', 'sistemas', 'redes', 'base', 'datos', 'servidor', 'cliente'])):
                    technical_terms.append(word_clean)
            
            unique_skills.extend(technical_terms[:5])  # Máximo 5 términos adicionales
            unique_skills = list(set(unique_skills))
        
        print(f"🔍 DEBUG: Habilidades extraídas: {unique_skills}")
        return unique_skills[:20]  # Limitar a 20 habilidades máximo

    def analyze_resume(self, pdf_path: str) -> Optional[Dict]:
        """
        Complete resume analysis.
        Args:
            pdf_path: Path to the PDF file
        Returns:
            Dictionary containing analysis results or None if analysis fails
        """
        try:
            print(f"🔍 DEBUG: Iniciando análisis de CV: {pdf_path}")
            
            # Extraer texto del PDF
            text = self.extract_resume_text(pdf_path)
            
            if not text:
                print(f"❌ No se pudo extraer texto del CV")
                return None
            
            print(f"✅ Texto extraído, analizando habilidades...")
            
            # Extraer habilidades
            skills = self.extract_skills(text)
            
            if not skills:
                print(f"⚠️ No se encontraron habilidades en el CV")
                # Retornar resultado con habilidades vacías en lugar de None
                return {
                    "text": text,
                    "skills": []
                }
            
            print(f"✅ Análisis completado. Habilidades encontradas: {len(skills)}")
            
            return {
                "text": text,
                "skills": skills
            }
            
        except Exception as e:
            print(f"❌ Error en analyze_resume: {str(e)}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return None