import logging
import os
import json

class TransformationManager:
    """
    Gère les transformations de manière persistante
    """
    def __init__(self, storage_path='transformations'):
        """
        Initialise le gestionnaire de transformations
        
        :param storage_path: Répertoire pour stocker les fichiers de transformation
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def _get_file_path(self, file_key):
        """
        Génère un chemin de fichier unique pour une clé donnée
        
        :param file_key: Identifiant unique pour le fichier
        :return: Chemin complet vers le fichier de transformation
        """
        return os.path.join(self.storage_path, f"{file_key}_transformations.json")
    
    def save_transformations(self, file_key, transformations):
        """
        Sauvegarde les transformations pour un fichier spécifique
        
        :param file_key: Identifiant unique pour le fichier
        :param transformations: Dictionnaire des transformations à sauvegarder
        """
        file_path = self._get_file_path(file_key)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(transformations, f, indent=4)
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des transformations : {e}")
    
    def get_transformations(self, file_key):
        """
        Récupère les transformations pour un fichier spécifique
        
        :param file_key: Identifiant unique pour le fichier
        :return: Dictionnaire des transformations ou dictionnaire vide
        """
        file_path = self._get_file_path(file_key)
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des transformations : {e}")
        
        return {}
    
    def clear_transformations(self, file_key):
        """
        Efface les transformations pour un fichier spécifique
        
        :param file_key: Identifiant unique pour le fichier
        """
        file_path = self._get_file_path(file_key)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Erreur lors de la suppression des transformations : {e}")