import io
import os 
import uuid
import logging
import json
import hashlib
import re
import camelot

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Classe pour le traitement et l'analyse des fichiers PDF"""

    def __init__(self, model_name="mistral:latest", context_size=4096, log_level=logging.INFO):
        """
        Initialise le processeur PDF avec un modèle d'IA.
        
        Args:
            model_name: Nom du modèle à utiliser (Ollama) 
            context_size: Taille du contexte pour le modèle
            log_level: Niveau de détail des logs  
        """
        self.model_name = model_name
        self.context_size = context_size
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Configurer le logger
        self.logger = logging.getLogger(f"{__name__}.PDFProcessor")
        self.logger.setLevel(log_level)

    def process_pdf(self, pdf_path, context=None):
        """
        Traite un fichier PDF et génère une analyse.
        
        Args:  
            pdf_path (str): Chemin vers le fichier PDF
            context (str): Contexte utilisateur ou historique pour l'analyse
            
        Returns:
            dict: Résultats de l'analyse avec métadonnées
        """
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Créer un ID unique pour ce PDF
            pdf_id = str(uuid.uuid4())

            # Extraction du texte et des métadonnées
            pdf_text, metadata = self.extract_pdf_content(pdf_bytes)

            if not pdf_text:  
                return {
                    "success": False,
                    "error": "Impossible d'extraire le texte du PDF",
                    "pdf_id": pdf_id
                }

            # Extraction des tableaux avec Camelot
            tables = self.extract_tables(pdf_path)

            # Génération de l'analyse
            analysis = self.generate_analysis(pdf_text, metadata, context)

            # Construire le résultat
            result = {  
                "success": True,
                "pdf_id": pdf_id, 
                "metadata": metadata,
                "analysis": analysis,
                "tables": tables  # Ajouter les tableaux extraits
            }

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du PDF: {e}")
            return {
                "success": False,
                "error": str(e), 
                "pdf_id": str(uuid.uuid4())
            }

    def extract_pdf_content(self, pdf_bytes):
        """
        Extrait le texte et les métadonnées d'un fichier PDF.
        
        Args:
            pdf_bytes (bytes): Contenu du PDF en bytes
            
        Returns:  
            tuple: (texte extrait, métadonnées)
        """
        try:
            import fitz  # PyMuPDF
            
            # Ouvrir le document
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Extraire les métadonnées
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""), 
                "page_count": len(doc)
            }

            # Extraire le texte  
            text = ""
            word_count = 0
            char_count = 0

            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n" + page_text

                # Compter les mots et caractères
                words = re.findall(r'\w+', page_text)  
                word_count += len(words)
                char_count += len(page_text)

            # Ajouter des statistiques aux métadonnées
            metadata["word_count"] = word_count
            metadata["character_count"] = char_count

            doc.close()
            return text, metadata

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction du contenu PDF: {e}") 
            raise

    def extract_tables(self, pdf_path):
        """
        Extrait les tableaux du PDF avec Camelot.
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF 
            
        Returns:
            list: Liste des tableaux extraits sous forme de dictionnaires
        """
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            # Convertir les tableaux en listes de dictionnaires
            table_data = []
            for table in tables:
                table_data.append(table.df.to_dict(orient='records'))
            
            return table_data
        
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'extraction des tableaux: {e}")
            return []

    def generate_analysis(self, pdf_text, metadata, context=None):
        """
        Génère une analyse du contenu du PDF avec un modèle d'IA.
        
        Args:
            pdf_text (str): Texte extrait du PDF 
            metadata (dict): Métadonnées du PDF
            context (str): Contexte utilisateur ou historique pour l'analyse
            
        Returns:
            dict: Analyse structurée du PDF  
        """
        try:
            # Créer une analyse basique des métadonnées
            summary = f"Document de {metadata['page_count']} pages contenant environ {metadata['word_count']} mots."
            
            if metadata.get('title'):
                summary += f" Titre: {metadata['title']}."
            
            if metadata.get('author'):
                summary += f" Auteur: {metadata['author']}."
            
            # Extraire quelques insights basiques
            insights = []
            
            # Longueur du document
            if metadata['page_count'] > 50:
                insights.append("Document relativement long (plus de 50 pages).")
            elif metadata['page_count'] < 5:
                insights.append("Document court (moins de 5 pages).")
            
            # Densité de texte
            avg_words_per_page = metadata['word_count'] / metadata['page_count'] if metadata['page_count'] > 0 else 0
            if avg_words_per_page > 500:
                insights.append(f"Document dense en texte (moyenne de {int(avg_words_per_page)} mots par page).")
            elif avg_words_per_page < 100:
                insights.append(f"Document peu dense en texte (moyenne de {int(avg_words_per_page)} mots par page).")
            
            # Adapter l'analyse en fonction du contexte
            if context:
                if context.lower() == 'summary':
                    # Résumé simpliste basé sur les premières lignes
                    first_paragraphs = pdf_text.split('\n\n')[:3]
                    content_summary = ' '.join(first_paragraphs)
                    insights.append(f"Aperçu du contenu: {content_summary[:300]}...")
                
                elif context.lower() == 'keywords':
                    # Extraction simplifiée de mots-clés (à améliorer avec NLP)
                    words = re.findall(r'\b[A-Za-z]{4,}\b', pdf_text.lower())
                    word_counts = {}
                    for word in words:
                        if word not in ['page', 'this', 'that', 'with', 'from', 'have', 'there']:
                            word_counts[word] = word_counts.get(word, 0) + 1
                    
                    # Trier par fréquence
                    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    insights.append(f"Mots-clés potentiels: {', '.join([k for k, v in keywords])}")
            
            # Structure de l'analyse
            analysis = {
                "summary": summary,
                "insights": insights,
                "context_type": context or "general"
            }
            
            return analysis
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de l'analyse: {e}")
            return {
                "summary": "Analyse non disponible en raison d'une erreur.",
                "insights": ["Une erreur s'est produite lors de l'analyse."],
                "context_type": "error"
            }