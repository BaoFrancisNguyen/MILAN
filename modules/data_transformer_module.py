import os
import sys
import pandas as pd
import numpy as np
import re
import logging
import requests
import json
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

# Configuration avancée du logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configuration complète du système de logging
    
    Args:
        log_level: Niveau de logging
        log_file: Fichier de log (optionnel)
    """
    # Configuration des handlers
    handlers = [logging.StreamHandler(sys.stdout)]  # Toujours log vers la console
    
    # Ajouter un handler de fichier si un chemin est fourni
    if log_file:
        try:
            # Utiliser un chemin absolu si possible
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            # Ajouter le handler de fichier
            file_handler = logging.FileHandler(log_file, mode='a')
            handlers.append(file_handler)
        except Exception as e:
            print(f"Impossible de créer le fichier de log : {e}")
    
    # Configuration du logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

# Initialiser le logger avec un nom de fichier par défaut
log_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data_transformer.log'
)
logger = setup_logging(log_file=log_file)

class DataTransformer:
    """
    Classe avancée pour la transformation et l'analyse de données avec capacités d'IA
    """
    
    def __init__(self, 
                 model_name="mistral:latest", 
                 context_size=4096, 
                 log_level=logging.INFO,
                 ollama_url=None):
        """
        Initialisation du transformateur de données
        
        Args:
            model_name: Nom du modèle IA
            context_size: Taille maximale du contexte
            log_level: Niveau de logging
            ollama_url: URL personnalisée pour l'API Ollama
        """
        self.model_name = model_name
        self.context_size = context_size
        
        # Configuration du logger
        self.logger = setup_logging(log_level, log_file)
        
        # Configuration de l'URL Ollama
        self.ollama_url = ollama_url or os.environ.get(
            "OLLAMA_API_URL", 
            "http://localhost:11434/api/generate"
        )
        
        # Vérifier la disponibilité du modèle
        self._validate_model_availability()
    
    def _validate_model_availability(self):
        """
        Vérifie la disponibilité et la configuration du modèle IA
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m.get("name") for m in models]
                
                if self.model_name in available_models:
                    self.logger.info(f"Modèle IA disponible : {self.model_name}")
                else:
                    self.logger.warning(
                        f"Modèle {self.model_name} non trouvé. "
                        f"Modèles disponibles : {available_models}"
                    )
            else:
                self.logger.error("Impossible de se connecter à l'API Ollama")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erreur de connexion à Ollama : {e}")
    
    def generate_with_ai(self, 
                          prompt: str, 
                          max_tokens: int = 800, 
                          temperature: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        Génération de réponse avec le modèle IA
        
        Args:
            prompt: Texte de prompt
            max_tokens: Nombre maximum de tokens
            temperature: Paramètre de température
        
        Returns:
            Réponse générée ou None
        """
        try:
            # Préparation du prompt avec instructions en français
            language_prefix = (
                "INSTRUCTION OBLIGATOIRE: RÉPONDS EXCLUSIVEMENT EN FRANÇAIS. "
                "Sois précis, concis et informatif. "
            )
            
            full_prompt = language_prefix + prompt
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {"choices": [{"text": result.get("response", "")}]}
            else:
                self.logger.error(f"Erreur Ollama : {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.error(f"Erreur de génération IA : {e}")
            return None
    
    def generate_dataset_analysis(self, 
                                   df: pd.DataFrame, 
                                   context: Optional[str] = None) -> str:
        """
        Génère une analyse complète du dataset
        
        Args:
            df: DataFrame à analyser
            context: Contexte ou question spécifique
        
        Returns:
            Analyse textuelle du dataset
        """
        try:
            # Collecte des statistiques de base
            stats = {
                'dimensions': f"{df.shape[0]} lignes, {df.shape[1]} colonnes",
                'missing_values': df.isna().sum().sum(),
                'missing_percentage': df.isna().sum().sum() / df.size * 100,
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns)
            }
            
            # Préparation du prompt
            prompt = (
                f"Analyse détaillée d'un dataset :\n"
                f"Dimensions : {stats['dimensions']}\n"
                f"Valeurs manquantes : {stats['missing_values']} ({stats['missing_percentage']:.2f}%)\n"
                f"Colonnes numériques : {', '.join(stats['numeric_columns'])}\n"
                f"Colonnes catégorielles : {', '.join(stats['categorical_columns'])}\n"
            )
            
            # Ajouter le contexte si présent
            if context:
                prompt += f"\nDemande spécifique : {context}\n"
            
            # Générer l'analyse
            response = self.generate_with_ai(prompt)
            
            return response['choices'][0]['text'] if response else "Analyse impossible"
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse : {e}")
            return f"Erreur d'analyse : {e}"
    
    def transform(self, 
                  df: pd.DataFrame, 
                  transformations=None, 
                  context: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transformation de base du DataFrame
        
        Args:
            df: DataFrame à transformer
            transformations: Liste de transformations
            context: Contexte utilisateur
        
        Returns:
            DataFrame et métadonnées
        """
        # Initialiser les métadonnées
        metadata = {
            "original_shape": df.shape,
            "missing_values": {
                "before": df.isna().sum().sum()
            },
            "transformations": transformations or []
        }
        
        # Générer une analyse si un contexte est fourni
        if context:
            analysis = self.generate_dataset_analysis(df, context)
            metadata["analysis"] = analysis
        
        return df, metadata

def test_data_transformer():
    """
    Fonction de test pour le DataTransformer
    """
    try:
        # Créer un DataFrame de test
        test_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salaire': [30000, 45000, 50000, 60000, 75000],
            'ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice']
        })
        
        # Initialiser le transformateur
        transformer = DataTransformer()
        
        # Tester l'analyse
        analyse = transformer.generate_dataset_analysis(test_df)
        print("Analyse du dataset :")
        print(analyse)
        
        # Tester la transformation
        _, metadata = transformer.transform(test_df, context="Analyse les caractéristiques principales")
        print("\nMétadonnées de transformation :")
        print(json.dumps(metadata, indent=2))
        
    except Exception as e:
        logger.error(f"Erreur lors des tests : {e}")
        print(f"Erreur lors des tests : {e}")

if __name__ == "__main__":
    test_data_transformer()