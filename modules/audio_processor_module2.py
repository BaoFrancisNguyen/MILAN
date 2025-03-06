import os
import torch
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchaudio
import soundfile as sf

class LocalAudioProcessor:
    def __init__(self, 
                 whisper_model_size='medium', 
                 translation_model_path=None,
                 device=None):
        """
        Initialisation du module de traitement audio local
        
        :param whisper_model_size: Taille du modèle Whisper
        :param translation_model_path: Chemin du modèle de traduction
        :param device: Appareil de calcul (cpu/cuda)
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
        Charger le modèle de traduction
        
        :param model_path: Chemin du modèle
        """
        try:
            self.translation_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.translation_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map='auto'
            )
        except Exception as e:
            print(f"Erreur de chargement du modèle de traduction : {e}")
            self.translation_model = None
            self.translation_tokenizer = None

    def _preprocess_audio(self, audio_path):
        """
        Prétraitement avancé pour réduire le bruit et améliorer la qualité audio
        
        Args:
            audio_path (str): Chemin du fichier audio à prétraiter
        
        Returns:
            str: Chemin du fichier audio prétraité
        """
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Charger l'audio avec librosa (plus robuste que torchaudio)
            audio, sample_rate = librosa.load(audio_path, sr=None)
            
            # Convertir en mono si stéréo
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            # Rééchantillonnage à 16kHz (requis par Whisper)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Réduction de bruit simple
            # Appliquer un préemphase qui renforce les hautes fréquences
            audio = librosa.effects.preemphasis(audio)
            
            # Normalisation du volume
            audio = librosa.util.normalize(audio)
            
            # Réduction du bruit de fond (optionnel)
            try:
                # Utiliser la transformation spectrale pour réduire le bruit
                spec = librosa.stft(audio)
                spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
                
                # Seuillage simple pour réduire le bruit
                noise_threshold = np.percentile(spec_db, 10)
                spec_cleaned = np.where(spec_db > noise_threshold, spec, 0)
                
                # Reconstruction du signal
                audio_cleaned = librosa.istft(spec_cleaned)
            except Exception:
                # En cas d'échec de la réduction de bruit, utiliser l'audio original
                audio_cleaned = audio
            
            # Sauvegarder le fichier prétraité
            preprocessed_path = audio_path.replace('.', '_preprocessed.')
            sf.write(preprocessed_path, audio_cleaned, sample_rate)
            
            return preprocessed_path
        
        except ImportError:
            print("Bibliothèques de traitement audio manquantes. Installez librosa et soundfile.")
            # Fallback à la méthode originale si les bibliothèques sont manquantes
            try:
                import torchaudio
                import torch
                
                # Charger l'audio
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Convertir en mono si stéréo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resampling si nécessaire
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Sauvegarder le fichier prétraité
                preprocessed_path = audio_path.replace('.', '_preprocessed.')
                sf.write(preprocessed_path, waveform.squeeze().numpy(), 16000)
                
                return preprocessed_path
            
            except Exception as e:
                print(f"Erreur de prétraitement audio : {e}")
                # Utiliser le fichier original si tous les prétraitements échouent
                return audio_path

        except Exception as e:
            print(f"Erreur lors du prétraitement audio : {e}")
            # En dernier recours, retourner le fichier original
            return audio_path

    def transcribe(self, audio_path, language=None):
        """
        Transcrire un fichier audio avec des options avancées pour différents types de qualité audio
        
        Args:
            audio_path (str): Chemin du fichier audio à transcrire
            language (str, optional): Langue attendue de l'audio
        
        Returns:
            dict: Résultats de transcription avec texte, langue et segments
        """
        # Prétraiter l'audio
        processed_audio_path = self._preprocess_audio(audio_path)
        
        try:
            # Options de transcription avancées
            options = {
                # Options de précision et de recherche
                'beam_size': 5,  # Augmente la précision de la transcription
                'best_of': 3,    # Sélectionne la meilleure hypothèse parmi plusieurs
                
                # Options pour les audio de mauvaise qualité
                'condition_on_previous_text': False,  # Ignore le contexte précédent
                'compression_ratio_threshold': 2.4,   # Plus tolérant aux bruits
                'log_prob_threshold': -1.0,           # Plus permissif sur la vraisemblance
                'no_speech_threshold': 0.6,           # Plus tolérant aux segments bruités
                
                # Détection et transcription de langue
                'language': language,  # Langue spécifiée (optionnel)
                'task': 'transcribe',  # Tâche de transcription
            }

            # Transcription avec gestion des segments
            segments, info = self.whisper_model.transcribe(
                processed_audio_path, 
                **options
            )

            # Compiler le texte avec des informations détaillées
            transcription_segments = []
            full_text = []
            
            for segment in segments:
                segment_info = {
                    'text': segment.text,
                    'start': segment.start,
                    'end': segment.end,
                    'probability': segment.probability
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
                'text': ' '.join(full_text),  # Texte complet
                'language': info.language,    # Langue détectée
                'language_probability': info.language_probability,  # Probabilité de la langue
                'segments': transcription_segments,  # Segments détaillés
                'duration': info.duration,    # Durée de l'audio
                'all_language_probs': info.all_language_probs  # Probabilités pour toutes les langues
            }
        
        except Exception as transcription_error:
            # Nettoyage en cas d'erreur
            if os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                except:
                    pass
            
            # Journalisation de l'erreur
            print(f"Erreur de transcription : {transcription_error}")
            raise

    def translate(self, text, source_lang='fr', target_lang='en'):
        """
        Traduire un texte
        
        :param text: Texte à traduire
        :param source_lang: Langue source
        :param target_lang: Langue cible
        :return: Texte traduit
        """
        if not self.translation_model:
            raise ValueError("Aucun modèle de traduction chargé")

        # Prompt de traduction
        prompt = f"""Translate the following text from {source_lang} to {target_lang}:

{text}

Translation:"""

        # Génération de la traduction
        inputs = self.translation_tokenizer(prompt, return_tensors='pt').to(self.translation_model.device)
        
        outputs = self.translation_model.generate(
            **inputs, 
            max_new_tokens=200,  # Limiter la longueur
            do_sample=True,      # Échantillonnage pour plus de variété
            temperature=0.7      # Contrôle de la créativité
        )

        # Décoder la réponse
        translation = self.translation_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        return translation

    def process_audio(self, audio_path, target_language=None):
        """
        Processus complet : transcription puis traduction
        
        :param audio_path: Chemin du fichier audio
        :param target_language: Langue de traduction (optionnel)
        :return: Résultats de traitement
        """
        # Transcrire
        transcription_result = self.transcribe(audio_path)
        
        # Traduire si demandé
        translated_text = None
        if target_language and self.translation_model:
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
    # Chemin vers vos modèles
    WHISPER_MODEL = 'base'  # ou le chemin local de votre modèle Whisper
    TRANSLATION_MODEL = '/chemin/vers/votre/modele/dolphin-mixtral'
    
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
