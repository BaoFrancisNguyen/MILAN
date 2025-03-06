import os
import torch
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchaudio
import soundfile as sf
import logging

class LocalAudioProcessor:
    def __init__(self, 
                 whisper_model_size='medium', 
                 translation_model_path=None,
                 device=None):
        """
        Initialisation du module de traitement audio local
        
        Args:
            whisper_model_size (str): Taille du modèle Whisper
            translation_model_path (str): Chemin ou nom du modèle de traduction
            device (str, optional): Appareil de calcul (cpu/cuda)
        """
        # Déterminer l'appareil
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialiser Whisper
        self.whisper_model = WhisperModel(
            whisper_model_size, 
            device=self.device, 
            compute_type='int8'  # Optimisation mémoire
        )
        
        # Initialiser le modèle de traduction (optionnel)
        self.translation_model = None
        self.translation_tokenizer = None
        if translation_model_path:
            self._load_translation_model(translation_model_path)

    def _load_translation_model(self, model_path):
        """
        Préparer le modèle de traduction, avec support spécifique pour Ollama
        
        Args:
            model_path (str): Chemin ou identificateur du modèle
        """
        # Vérifier si Ollama est en cours d'utilisation
        if model_path == 'mistral:latest' or model_path.lower() == 'ollama':
            try:
                import ollama
                # Vérifier la disponibilité du modèle Mistral
                try:
                    models = ollama.list()
                    mistral_available = any(model['name'] == 'mistral:latest' for model in models['models'])
                    
                    if not mistral_available:
                        print("Téléchargement du modèle Mistral via Ollama...")
                        ollama.pull('mistral')
                    
                    print("✅ Ollama avec Mistral est prêt pour la traduction")
                    return
                
                except Exception as e:
                    print(f"❌ Erreur de vérification du modèle Ollama : {e}")
                    self.translation_model = None
                    self.translation_tokenizer = None
                    return
            
            except ImportError:
                print("❌ Ollama n'est pas installé. Installez-le avec 'pip install ollama'")
                self.translation_model = None
                self.translation_tokenizer = None
                return

        # Pour les modèles Hugging Face classiques
        try:
            print(f"Chargement du modèle de traduction : {model_path}")
            
            # Charger le tokenizer
            self.translation_tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                use_fast=True,
                trust_remote_code=True
            )
            
            # Charger le modèle
            self.translation_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print("✅ Modèle de traduction chargé avec succès")
        
        except Exception as e:
            print(f"❌ Erreur de chargement du modèle de traduction : {e}")
            self.translation_model = None
            self.translation_tokenizer = None

    def _preprocess_audio(self, audio_path):
        """
        Prétraiter le fichier audio
        
        Args:
            audio_path (str): Chemin du fichier audio
        
        Returns:
            str: Chemin du fichier audio prétraité
        """
        # Charger l'audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convertir en mono si stéréo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resampling si nécessaire (Whisper attend 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Sauvegarder le fichier prétraité
            preprocessed_path = audio_path.replace('.', '_preprocessed.')
            sf.write(preprocessed_path, waveform.squeeze().numpy(), 16000)
            
            return preprocessed_path
        
        except Exception as e:
            print(f"Erreur lors du prétraitement audio : {e}")
            raise

    def transcribe(self, audio_path, language=None):
        """
        Transcrire un fichier audio avec des options avancées
        
        Args:
            audio_path (str): Chemin du fichier audio à transcrire
            language (str, optional): Langue attendue de l'audio
        
        Returns:
            dict: Résultats de transcription avec texte, langue et segments
        """
        # Configuration du logger
        logger = logging.getLogger(__name__)

        # Prétraiter l'audio
        processed_audio_path = self._preprocess_audio(audio_path)
        
        try:
            # Options de transcription avancées
            options = {
                'beam_size': 5,
                'condition_on_previous_text': False,
                'compression_ratio_threshold': 2.4,
                'log_prob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'language': language,
                'task': 'transcribe',
            }

            # Obtenir la durée de l'audio
            import soundfile as sf
            audio_data, sample_rate = sf.read(processed_audio_path)
            audio_duration = len(audio_data) / sample_rate
            
            # Log de la durée de l'audio
            logger.info(f"Processing audio with duration {int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}.{int((audio_duration % 1) * 1000):03d}")

            # Transcription
            segments, info = self.whisper_model.transcribe(
                processed_audio_path, 
                **options
            )

            # Normaliser la probabilité de langue
            language_prob_percent = (info.language_probability * 100) if info.language_probability <= 1 else info.language_probability
            logger.info(f"Detected language '{info.language}' with probability {language_prob_percent:.2f}%")

            # Compiler le texte avec des informations détaillées
            transcription_segments = []
            full_text = []
            
            for segment in segments:
                segment_info = {
                    'text': segment.text,
                    'start': segment.start,
                    'end': segment.end,
                    # Suppression de la vérification de probability
                }
                transcription_segments.append(segment_info)
                full_text.append(segment.text)

            # Nettoyer le fichier prétraité
            try:
                os.remove(processed_audio_path)
            except Exception as cleanup_error:
                print(f"Erreur lors du nettoyage du fichier : {cleanup_error}")

            # Informations de transcription détaillées
            return {
                'text': ' '.join(full_text),
                'language': info.language,
                'language_probability': f"{language_prob_percent:.2f}%",  # Formaté en pourcentage
                'segments': transcription_segments,
                'duration': f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}.{int((audio_duration % 1) * 1000):03d}",
                'all_language_probs': info.all_language_probs
            }
        
        except Exception as transcription_error:
            # Nettoyage en cas d'erreur
            if os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                except:
                    pass
            
            # Journalisation de l'erreur
            logger.error(f"Erreur de transcription : {transcription_error}")
            raise

    def translate(self, text, source_lang='fr', target_lang='en'):
        """
        Traduire un texte en utilisant Ollama et le modèle Mistral
        
        Args:
            text (str): Texte à traduire
            source_lang (str, optional): Langue source. Defaults to 'fr'.
            target_lang (str, optional): Langue cible. Defaults to 'en'.
        
        Returns:
            str: Texte traduit ou None en cas d'erreur
        """
        # Vérifier si le texte est vide ou trop court
        if not text or len(text.strip()) < 2:
            return text

        try:
            import ollama
            
            # Construire un prompt de traduction clair et précis
            prompt = f"""Tu es un traducteur professionnel hautement qualifié.
Translate the text below from {source_lang} to {target_lang}. 
Provide only the translation without any additional commentary or explanation.

Original text ({source_lang}):
{text}

Translation ({target_lang}):"""
            
            # Appeler Ollama avec le modèle Mistral
            response = ollama.chat(
                model='mistral:latest', 
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.3,  # Réduit la variabilité
                    'num_predict': 300,  # Limite la longueur de la traduction
                }
            )
            
            # Extraire et nettoyer la traduction
            translation = response['message']['content'].strip()
            
            return translation
        
        except ImportError:
            print("Erreur : Ollama n'est pas installé. Installez-le avec 'pip install ollama'")
            return None
        
        except Exception as e:
            print(f"Erreur de traduction via Ollama : {e}")
            return None

    def process_audio(self, audio_path, target_language=None):
        # Transcrire
        transcription_result = self.transcribe(audio_path)
        
        # Traduire si demandé
        translated_text = None
        if target_language:
            try:
                translated_text = self.translate(
                    transcription_result['text'],
                    source_lang=transcription_result['language'],
                    target_lang=target_language
                )
            except Exception as e:
                print(f"Erreur de traduction : {e}")
        
        return {
            'original_text': transcription_result['text'],
            'original_language': transcription_result['language'],
            'translated_text': translated_text,
            'target_language': target_language
        }

# Exemple d'utilisation
def main():
    # Chemins à configurer
    WHISPER_MODEL = 'medium'  # ou le chemin local de votre modèle Whisper
    TRANSLATION_MODEL = 'mistral:latest'  # Utilisation d'Ollama
    
    # Chemin du fichier audio à traiter
    AUDIO_PATH = '/chemin/vers/votre/fichier/audio.mp3'

    # Initialiser le processeur
    processor = LocalAudioProcessor(
        whisper_model_size=WHISPER_MODEL,
        translation_model_path=TRANSLATION_MODEL
    )

    # Traiter l'audio
    result = processor.process_audio(
        AUDIO_PATH, 
        target_language='en'  # Traduire en anglais
    )

    # Afficher les résultats
    print("Transcription originale:", result['original_text'])
    print("Langue détectée:", result['original_language'])
    print("Traduction:", result['translated_text'])

if __name__ == '__main__':
    main()