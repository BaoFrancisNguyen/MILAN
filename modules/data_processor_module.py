import pandas as pd
import numpy as np
import logging
import hashlib
import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from datetime import datetime

# Ajouter le chemin du module
sys.path.append('./modules')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Classe pour le traitement et la transformation des données CSV"""

    def __init__(self, upload_folder='uploads'):
        """
        Initialisation du processeur de données
        
        Args:
            upload_folder: Dossier de stockage des fichiers transformés
        """
        self.logger = logging.getLogger(f"{__name__}.DataProcessor")
        self.transformations_history = []
        self.upload_folder = upload_folder
        
        # Créer le dossier s'il n'existe pas
        os.makedirs(self.upload_folder, exist_ok=True)

    def process_dataframe(self, df, transformations=None, user_context=None):
        """
        Traite un DataFrame selon les transformations spécifiées.
        
        Args:
            df: DataFrame à traiter
            transformations: Dictionnaire des transformations à appliquer
            user_context: Contexte utilisateur pour l'analyse IA
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        # Copier le DataFrame pour ne pas modifier l'original
        processed_df = df.copy()
        original_shape = processed_df.shape
        
        # Générer un nom de fichier unique pour la sauvegarde
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"original_{timestamp}.csv"
        original_filepath = os.path.join(self.upload_folder, original_filename)
        df.to_csv(original_filepath, index=False)
        
        # Initialiser les métadonnées
        metadata = {
            "original_shape": original_shape,
            "original_filename": original_filename,
            "transformations_applied": [],
            "columns_added": [],
            "columns_removed": [],
            "missing_values": {
                "before": processed_df.isna().sum().sum()
            }
        }
        
        # Si aucune transformation n'est spécifiée, retourner le DataFrame tel quel
        if not transformations:
            metadata["analysis"] = "Aucune transformation appliquée."
            return processed_df, metadata
        
        # Appliquer les transformations
        for transform_type, transform_params in transformations.items():
            try:
                transform_method = {
                    "missing_values": self.handle_missing_values,
                    "standardization": self.standardize_data,
                    "encoding": self.encode_categorical,
                    "outliers": self.handle_outliers,
                    "feature_engineering": self.engineer_features,
                    "dimension_reduction": self.reduce_dimensions,
                    "drop_columns": self.drop_columns,
                    "merge_columns": self.merge_columns,
                    "replace_values": self.replace_values
                }.get(transform_type)
                
                if not transform_method:
                    self.logger.warning(f"Transformation inconnue: {transform_type}")
                    continue
                
                # Appeler la méthode de transformation
                processed_df, transform_meta = transform_method(
                    processed_df, 
                    **transform_params if transform_params else {}
                )
                
                # Ajouter les métadonnées de transformation
                metadata["transformations_applied"].append(transform_type)
                
                # Mettre à jour les colonnes ajoutées/supprimées
                if "columns_added" in transform_meta:
                    metadata["columns_added"].extend(transform_meta["columns_added"])
                if "columns_removed" in transform_meta:
                    metadata["columns_removed"].extend(transform_meta["columns_removed"])
                
                # Enregistrer l'historique de transformation
                self.transformations_history.append({
                    "transformation": transform_type,
                    "metadata": transform_meta
                })
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la transformation '{transform_type}': {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Continuer avec les autres transformations malgré l'erreur
        
        # Mettre à jour le nombre de valeurs manquantes après transformation
        metadata["missing_values"]["after"] = processed_df.isna().sum().sum()
        
        # Générer une analyse du DataFrame transformé si un contexte est fourni
        if user_context:
            try:
                from data_transformer_module import DataTransformer
                transformer = DataTransformer()
                _, analysis_meta = transformer.transform(processed_df, None, user_context)
                if "analysis" in analysis_meta:
                    metadata["analysis"] = analysis_meta["analysis"]
            except Exception as e:
                self.logger.error(f"Erreur lors de l'analyse IA: {e}")
                metadata["analysis"] = f"Erreur lors de l'analyse IA: {e}"
        else:
            # Analyser simplement les changements dans le DataFrame
            metadata["analysis"] = self.analyze_changes(df, processed_df, metadata)
        
        return processed_df, metadata

    # Les autres méthodes (handle_missing_values, standardize_data, etc.) 
    # restent identiques à l'implémentation précédente

    def analyze_changes(self, original_df, transformed_df, metadata):
        """
        Génère une analyse textuelle des changements appliqués au DataFrame.
        
        Args:
            original_df: DataFrame original
            transformed_df: DataFrame transformé
            metadata: Métadonnées des transformations
            
        Returns:
            str: Analyse textuelle
        """
        analysis = ["Rapport d'analyse des transformations appliquées:"]
        
        # Changements dans les dimensions
        original_shape = metadata["original_shape"]
        new_shape = transformed_df.shape
        
        analysis.append(f"• Dimensions: {original_shape[0]} lignes × {original_shape[1]} colonnes → {new_shape[0]} lignes × {new_shape[1]} colonnes")
        
        # Résumé des transformations
        if metadata["transformations_applied"]:
            analysis.append(f"• Transformations appliquées: {', '.join(metadata['transformations_applied'])}")
        
        # Traitement des valeurs manquantes
        if "missing_values" in metadata:
            before = metadata["missing_values"].get("before", 0)
            after = metadata["missing_values"].get("after", 0)
            if before > 0:
                analysis.append(f"• Valeurs manquantes: {before} → {after} ({100 - (after/before*100):.1f}% traitées)")
        
        # Colonnes modifiées
        if metadata.get("columns_added"):
            analysis.append(f"• Colonnes ajoutées: {len(metadata['columns_added'])}")
            analysis.append(f"  - {', '.join(metadata['columns_added'][:5])}" + ("..." if len(metadata['columns_added']) > 5 else ""))
        
        if metadata.get("columns_removed"):
            analysis.append(f"• Colonnes supprimées: {len(metadata['columns_removed'])}")
            analysis.append(f"  - {', '.join(metadata['columns_removed'][:5])}" + ("..." if len(metadata['columns_removed']) > 5 else ""))
        
        return "\n".join(analysis)

    # Les autres méthodes existantes restent identiques
    # (handle_missing_values, standardize_data, encode_categorical, etc.)
    # Elles sont copiées de l'implémentation précédente

    def handle_missing_values(self, df, strategy="auto", threshold=0.5, constant=None):
        """
        Gère les valeurs manquantes dans le DataFrame.
        
        Args:
            df: DataFrame à traiter
            strategy: Stratégie de traitement ('auto', 'drop_rows', 'drop_columns', 'fill_mean', 'fill_median', 'fill_mode', 'fill_constant')
            threshold: Seuil pour la suppression (si strategy est 'drop_rows' ou 'drop_columns')
            constant: Valeur constante pour le remplacement (si strategy est 'fill_constant')
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "strategy": strategy,
            "columns_affected": [],
            "total_missing_before": df.isna().sum().sum()
        }
        
        # Si aucune valeur manquante, retourner tel quel
        if metadata["total_missing_before"] == 0:
            return df, metadata
        
        # Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        # Appliquer la stratégie sélectionnée
        if strategy == "auto":
            # Stratégie automatique basée sur les données
            return self._handle_missing_auto(result_df, metadata)
        elif strategy == "drop_rows":
            # Supprimer les lignes avec plus de X% de valeurs manquantes
            rows_before = len(result_df)
            missing_rate = result_df.isna().mean(axis=1)
            result_df = result_df[missing_rate <= float(threshold)]
            metadata["rows_removed"] = rows_before - len(result_df)
        elif strategy == "drop_columns":
            # Supprimer les colonnes avec plus de X% de valeurs manquantes
            columns_before = len(result_df.columns)
            missing_rate = result_df.isna().mean(axis=0)
            columns_to_drop = missing_rate[missing_rate > float(threshold)].index.tolist()
            result_df = result_df.drop(columns=columns_to_drop)
            metadata["columns_removed"] = columns_to_drop
        elif strategy == "fill_mean":
            # Remplir par la moyenne (colonnes numériques uniquement)
            for column in df.select_dtypes(include=['number']).columns:
                if df[column].isna().any():
                    mean_value = df[column].mean()
                    result_df[column] = result_df[column].fillna(mean_value)
                    metadata["columns_affected"].append(column)
                    metadata.setdefault("fill_values", {})[column] = float(mean_value)
        elif strategy == "fill_median":
            # Remplir par la médiane (colonnes numériques uniquement)
            for column in df.select_dtypes(include=['number']).columns:
                if df[column].isna().any():
                    median_value = df[column].median()
                    result_df[column] = result_df[column].fillna(median_value)
                    metadata["columns_affected"].append(column)
                    metadata.setdefault("fill_values", {})[column] = float(median_value)
        elif strategy == "fill_mode":
            # Remplir par le mode (toutes colonnes)
            for column in df.columns:
                if df[column].isna().any():
                    mode_value = df[column].mode()[0] if not df[column].mode().empty else "NA"
                    result_df[column] = result_df[column].fillna(mode_value)
                    metadata["columns_affected"].append(column)
                    metadata.setdefault("fill_values", {})[column] = str(mode_value)
        elif strategy == "fill_constant":
            # Remplir par une valeur constante
            for column in df.columns:
                if df[column].isna().any():
                    result_df[column] = result_df[column].fillna(constant)
                    metadata["columns_affected"].append(column)
            metadata["constant_value"] = constant
        else:
            # Stratégie inconnue, retourner le DataFrame inchangé
            self.logger.warning(f"Stratégie de traitement des valeurs manquantes inconnue: {strategy}")
            return df, metadata
        
        # Mettre à jour les métadonnées
        metadata["total_missing_after"] = result_df.isna().sum().sum()
        
        return result_df, metadata

    def _handle_missing_auto(self, df, metadata):
        """
        Stratégie automatique pour gérer les valeurs manquantes.
        
        Args:
            df: DataFrame à traiter
            metadata: Métadonnées à compléter
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        # Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        # Pour chaque colonne avec des valeurs manquantes
        for column in df.columns[df.isna().any()]:
            missing_rate = df[column].isna().mean()
            metadata["columns_affected"].append(column)
            
            # Stratégie basée sur le taux de valeurs manquantes et le type de données
            if missing_rate > 0.5:
                # Plus de 50% de valeurs manquantes -> supprimer la colonne
                result_df = result_df.drop(columns=[column])
                metadata.setdefault("strategy", {})[column] = "drop_column"
                metadata.setdefault("columns_removed", []).append(column)
            elif pd.api.types.is_numeric_dtype(df[column]):
                # Pour les colonnes numériques, remplacer par la médiane
                median_value = df[column].median()
                result_df[column] = result_df[column].fillna(median_value)
                metadata.setdefault("strategy", {})[column] = f"median_fill:{median_value}"
            else:
                # Pour les colonnes non numériques, remplacer par la valeur la plus fréquente
                most_common = df[column].mode()[0] if not df[column].mode().empty else "NA"
                result_df[column] = result_df[column].fillna(most_common)
                metadata.setdefault("strategy", {})[column] = f"mode_fill:{most_common}"
        
        # Mettre à jour les métadonnées
        metadata["total_missing_after"] = result_df.isna().sum().sum()
        
        return result_df, metadata

    def standardize_data(self, df, columns=None, method="zscore"):
        """
        Standardise les colonnes numériques du DataFrame.
        
        Args:
            df: DataFrame à traiter
            columns: Liste des colonnes à standardiser (None pour toutes les colonnes numériques)
            method: Méthode de standardisation ('zscore', 'minmax', 'robust', 'maxabs')
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "standardized_columns": [],
            "method": method,
            "stats": {}
        }
        
        # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes numériques
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Vérifier s'il y a des colonnes numériques à standardiser
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) == 0:
            return df, metadata
        
        # Création d'une copie pour éviter# Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        # Standardisation selon la méthode choisie
        if method == "zscore":
            # Z-score (moyenne 0, écart-type 1)
            scaler = StandardScaler()
            result_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif method == "minmax":
            # MinMaxScaler (intervalle [0, 1])
            scaler = MinMaxScaler()
            result_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif method == "robust":
            # RobustScaler (robust aux valeurs aberrantes)
            scaler = RobustScaler()
            result_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif method == "maxabs":
            # MaxAbsScaler (conserve le signe, basé sur la valeur max absolue)
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
            result_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        else:
            # Méthode inconnue, retourner le DataFrame inchangé
            self.logger.warning(f"Méthode de standardisation inconnue: {method}")
            return df, metadata
        
        # Mise à jour des métadonnées
        for col in numeric_cols:
            metadata["standardized_columns"].append(col)
            metadata["stats"][col] = {
                "mean_before": float(df[col].mean()),
                "std_before": float(df[col].std()),
                "mean_after": float(result_df[col].mean()),
                "std_after": float(result_df[col].std())
            }
        
        return result_df, metadata

    def encode_categorical(self, df, columns=None, method="one_hot", drop_original=True):
        """
        Encode les variables catégorielles.
        
        Args:
            df: DataFrame à traiter
            columns: Liste des colonnes à encoder (None pour toutes les colonnes catégorielles)
            method: Méthode d'encodage ('one_hot', 'label', 'frequency')
            drop_original: Si True, supprime les colonnes originales après encodage
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "encoded_columns": {},
            "encoding_method": method,
            "columns_added": [],
            "columns_removed": []
        }
        
        # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes catégorielles
        if not columns:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Vérifier s'il y a des colonnes catégorielles à encoder
        cat_cols = [col for col in columns if col in df.columns]
        if len(cat_cols) == 0:
            return df, metadata
        
        # Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        # Pour chaque colonne catégorielle
        for col in cat_cols:
            if method == "one_hot":
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                
                # Ajout des nouvelles colonnes
                result_df = pd.concat([result_df, dummies], axis=1)
                
                # Suppression de la colonne originale si demandé
                if drop_original:
                    result_df = result_df.drop(columns=[col])
                    metadata["columns_removed"].append(col)
                
                # Mise à jour des métadonnées
                metadata["encoded_columns"][col] = {
                    "method": "one_hot",
                    "categories": df[col].unique().tolist(),
                    "new_columns": dummies.columns.tolist()
                }
                metadata["columns_added"].extend(dummies.columns.tolist())
            
            elif method == "label":
                # Label encoding pour convertir chaque catégorie en nombre entier
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                result_df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                
                # Suppression de la colonne originale si demandé
                if drop_original:
                    result_df = result_df.drop(columns=[col])
                    metadata["columns_removed"].append(col)
                
                # Mise à jour des métadonnées
                metadata["encoded_columns"][col] = {
                    "method": "label",
                    "mapping": dict(zip(le.classes_, range(len(le.classes_)))),
                    "new_columns": [f"{col}_encoded"]
                }
                metadata["columns_added"].append(f"{col}_encoded")
            
            elif method == "frequency":
                # Encodage fréquentiel (remplacer les catégories par leur fréquence)
                value_counts = df[col].value_counts(normalize=True)
                result_df[f"{col}_freq"] = df[col].map(value_counts)
                
                # Suppression de la colonne originale si demandé
                if drop_original:
                    result_df = result_df.drop(columns=[col])
                    metadata["columns_removed"].append(col)
                
                # Mise à jour des métadonnées
                metadata["encoded_columns"][col] = {
                    "method": "frequency",
                    "mapping": value_counts.to_dict(),
                    "new_columns": [f"{col}_freq"]
                }
                metadata["columns_added"].append(f"{col}_freq")
            
            else:
                # Méthode inconnue, ignorer cette colonne
                self.logger.warning(f"Méthode d'encodage inconnue: {method}")
                continue
        
        return result_df, metadata

    def handle_outliers(self, df, columns=None, method="iqr", treatment="tag"):
        """
        Détecte et traite les valeurs aberrantes.
        
        Args:
            df: DataFrame à traiter
            columns: Liste des colonnes à traiter (None pour toutes les colonnes numériques)
            method: Méthode de détection ('iqr', 'zscore', 'isolation_forest')
            treatment: Traitement à appliquer ('tag', 'winsorize', 'remove', 'impute')
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "outliers_detected": {},
            "method": method,
            "treatment": treatment,
            "columns_added": []
        }
        
        # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes numériques
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Vérifier s'il y a des colonnes numériques à traiter
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) == 0:
            return df, metadata
        
        # Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        # Dictionnaire pour stocker les masques d'outliers par colonne
        outlier_masks = {}
        
        # Détection des outliers selon la méthode choisie
        for col in numeric_cols:
            if method == "iqr":
                # Méthode IQR (Interquartile Range)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                
                metadata["outliers_detected"][col] = {
                    "method": "iqr",
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "count": int(outliers.sum()),
                    "percentage": float(outliers.mean() * 100)
                }
                
                outlier_masks[col] = outliers
            
            elif method == "zscore":
                # Méthode Z-score (écarts à la moyenne)
                from scipy import stats
                z_scores = stats.zscore(df[col], nan_policy='omit')
                outliers = (abs(z_scores) > 3)  # Seuil standard : 3 écarts-types
                
                metadata["outliers_detected"][col] = {
                    "method": "zscore",
                    "threshold": 3,
                    "count": int(outliers.sum()),
                    "percentage": float(outliers.mean() * 100)
                }
                
                outlier_masks[col] = outliers
            
            elif method == "isolation_forest":
                # Méthode Isolation Forest (apprentissage automatique)
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    # Reshape pour sklearn
                    X = df[col].values.reshape(-1, 1)
                    
                    # Ajuster le modèle
                    iso_forest = IsolationForest(contamination=0.05, random_state=42)
                    outliers = iso_forest.fit_predict(X) == -1  # -1 pour les outliers
                    
                    metadata["outliers_detected"][col] = {
                        "method": "isolation_forest",
                        "contamination": 0.05,
                        "count": int(outliers.sum()),
                        "percentage": float(outliers.mean() * 100)
                    }
                    
                    outlier_masks[col] = outliers
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la détection avec Isolation Forest: {e}")
                    # Utiliser IQR par défaut en cas d'erreur
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                    
                    metadata["outliers_detected"][col] = {
                        "method": "iqr (fallback)",
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "count": int(outliers.sum()),
                        "percentage": float(outliers.mean() * 100)
                    }
                    
                    outlier_masks[col] = outliers
            
            else:
                # Méthode inconnue, utiliser IQR par défaut
                self.logger.warning(f"Méthode de détection d'outliers inconnue: {method}. Utilisation d'IQR par défaut.")
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                
                metadata["outliers_detected"][col] = {
                    "method": "iqr (default)",
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "count": int(outliers.sum()),
                    "percentage": float(outliers.mean() * 100)
                }
                
                outlier_masks[col] = outliers
        
        # Traitement des outliers selon la méthode choisie
        if treatment == "tag":
            # Ajouter une colonne indicatrice pour chaque colonne analysée
            for col, outliers in outlier_masks.items():
                indicator_col = f"{col}_outlier"
                result_df[indicator_col] = outliers.astype(int)
                metadata["columns_added"].append(indicator_col)
            
            metadata["treatment_details"] = "Outliers tagged with indicator columns"
        
        elif treatment == "winsorize":
            # Winsorisation : remplacer les valeurs aberrantes par les bornes
            for col, outliers in outlier_masks.items():
                if method == "iqr":
                    # Les bornes sont déjà calculées
                    lower_bound = metadata["outliers_detected"][col]["lower_bound"]
                    upper_bound = metadata["outliers_detected"][col]["upper_bound"]
                else:
                    # Calculer les bornes IQR pour d'autres méthodes
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                # Remplacer les valeurs en-dessous de la borne inférieure
                result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                
                # Remplacer les valeurs au-dessus de la borne supérieure
                result_df.loc[result_df[col] > upper_bound, col] = upper_bound
            
            metadata["treatment_details"] = "Outliers winsorized to bounds"
        
        elif treatment == "remove":
            # Supprimer les lignes contenant des valeurs aberrantes
            combined_mask = pd.Series(False, index=df.index)
            
            for col, outliers in outlier_masks.items():
                combined_mask = combined_mask | outliers
            
            # Nombre de lignes supprimées
            rows_removed = combined_mask.sum()
            
            # Supprimer les lignes
            result_df = result_df[~combined_mask]
            
            metadata["treatment_details"] = f"Removed {rows_removed} rows containing outliers"
            metadata["rows_removed"] = int(rows_removed)
        
        elif treatment == "impute":
            # Remplacer les valeurs aberrantes par la médiane de la colonne
            for col, outliers in outlier_masks.items():
                # Calculer la médiane (en excluant les outliers)
                median_value = df.loc[~outliers, col].median()
                
                # Remplacer les outliers par la médiane
                result_df.loc[outliers, col] = median_value
            
            metadata["treatment_details"] = "Outliers replaced with median values"
        
        else:
            # Traitement inconnu, ne rien faire
            self.logger.warning(f"Méthode de traitement d'outliers inconnue: {treatment}")
            return df, metadata
        
        return result_df, metadata

    def engineer_features(self, df, type_fe="interaction", columns=None, operations=None):
        """
        Crée de nouvelles caractéristiques à partir des existantes.
        
        Args:
            df: DataFrame à traiter
            type_fe: Type d'ingénierie ('interaction', 'polynomial', 'binning', 'time', 'text')
            columns: Liste des colonnes à utiliser
            operations: Liste des opérations à appliquer (pour 'interaction')
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "feature_type": type_fe,
            "features_created": {},
            "columns_added": []
        }
        
        # Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        if type_fe == "interaction":
            # Interactions entre colonnes numériques
            if not columns or len(columns) < 2:
                # Utiliser toutes les colonnes numériques (max 5) si non spécifié
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                columns = numeric_cols[:min(5, len(numeric_cols))]
            
            if not operations:
                # Opérations par défaut
                operations = ["multiplication"]
            
            # Vérifier les colonnes valides
            valid_columns = [col for col in columns if col in df.columns]
            
            # Créer les interactions entre paires de colonnes
            import itertools
            for col1, col2 in itertools.combinations(valid_columns, 2):
                # Vérifier si les deux colonnes sont numériques
                if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
                    continue
                
                if "multiplication" in operations:
                    new_col = f"{col1}_times_{col2}"
                    result_df[new_col] = df[col1] * df[col2]
                    metadata["features_created"][new_col] = {
                        "type": "multiplication",
                        "source_columns": [col1, col2]
                    }
                    metadata["columns_added"].append(new_col)
                
                if "division" in operations:
                    # Éviter les divisions par zéro
                    if not (df[col2] == 0).any():
                        new_col = f"{col1}_div_{col2}"
                        result_df[new_col] = df[col1] / df[col2].replace(0, float('nan'))
                        metadata["features_created"][new_col] = {
                            "type": "division",
                            "source_columns": [col1, col2]
                        }
                        metadata["columns_added"].append(new_col)
                
                if "addition" in operations:
                    new_col = f"{col1}_plus_{col2}"
                    result_df[new_col] = df[col1] + df[col2]
                    metadata["features_created"][new_col] = {
                        "type": "addition",
                        "source_columns": [col1, col2]
                    }
                    metadata["columns_added"].append(new_col)
                
                if "subtraction" in operations:
                    new_col = f"{col1}_minus_{col2}"
                    result_df[new_col] = df[col1] - df[col2]
                    metadata["features_created"][new_col] = {
                        "type": "subtraction",
                        "source_columns": [col1, col2]
                    }
                    metadata["columns_added"].append(new_col)
        
        elif type_fe == "polynomial":
            # Transformations polynomiales (carrés, cubes...)
            if not columns:
                # Utiliser toutes les colonnes numériques (max 5) si non spécifié
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                columns = numeric_cols[:min(5, len(numeric_cols))]
            
            # Vérifier les colonnes valides
            valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            # Créer les termes polynomiaux
            for col in valid_columns:
                # Terme au carré
                squared_col = f"{col}_squared"
                result_df[squared_col] = df[col] ** 2
                metadata["features_created"][squared_col] = {
                    "type": "polynomial",
                    "source_column": col,
                    "degree": 2
                }
                metadata["columns_added"].append(squared_col)
                
                # Terme au cube
                cubed_col = f"{col}_cubed"
                result_df[cubed_col] = df[col] ** 3
                metadata["features_created"][cubed_col] = {
                    "type": "polynomial",
                    "source_column": col,
                    "degree": 3
                }
                metadata["columns_added"].append(cubed_col)
        
        elif type_fe == "binning":
            # Discrétisation des variables numériques
            if not columns:
                # Utiliser toutes les colonnes numériques (max 5) si non spécifié
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                columns = numeric_cols[:min(5, len(numeric_cols))]
            
            # Vérifier les colonnes valides
            valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            # Discrétiser chaque colonne
            for col in valid_columns:
                # Créer 5 bins
                binned_col = f"{col}_binned"
                result_df[binned_col] = pd.cut(df[col], bins=5, labels=False)
                metadata["features_created"][binned_col] = {
                    "type": "binning",
                    "source_column": col,
                    "num_bins": 5
                }
                metadata["columns_added"].append(binned_col)
                
                # Créer également une version one-hot
                dummies = pd.get_dummies(result_df[binned_col], prefix=f"{col}_bin")
                result_df = pd.concat([result_df, dummies], axis=1)
                metadata["features_created"][f"{col}_bin_dummies"] = {
                    "type": "one_hot_binning",
                    "source_column": col,
                    "new_columns": dummies.columns.tolist()
                }
                metadata["columns_added"].extend(dummies.columns.tolist())
        
        elif type_fe == "time":
            # Extraction de caractéristiques temporelles à partir de dates
            # Identifier les colonnes de date
            if not columns:
                # Essayer de détecter les colonnes de date
                date_columns = []
                for col in df.columns:
                    try:
                        # Vérifier si la colonne peut être convertie en datetime
                        if df[col].dtype == 'object':
                            test_dates = pd.to_datetime(df[col], errors='coerce')
                            if test_dates.notna().mean() > 0.5:  # Plus de 50% sont des dates valides
                                date_columns.append(col)
                    except:
                        continue
                
                columns = date_columns
            
            # Vérifier les colonnes valides
            valid_columns = [col for col in columns if col in df.columns]
            
            # Extraire les caractéristiques de date pour chaque colonne
            for col in valid_columns:
                try:
                    # Convertir en datetime
                    dates = pd.to_datetime(df[col], errors='coerce')
                    
                    # Créer les caractéristiques
                    result_df[f"{col}_year"] = dates.dt.year
                    result_df[f"{col}_month"] = dates.dt.month
                    result_df[f"{col}_day"] = dates.dt.day
                    result_df[f"{col}_dayofweek"] = dates.dt.dayofweek
                    result_df[f"{col}_quarter"] = dates.dt.quarter
                    
                    # Mise à jour des métadonnées
                    for component in ['year', 'month', 'day', 'dayofweek', 'quarter']:
                        new_col = f"{col}_{component}"
                        metadata["features_created"][new_col] = {
                            "type": "date_component",
                            "source_column": col,
                            "component": component
                        }
                        metadata["columns_added"].append(new_col)
                    
                except Exception as e:
                    self.logger.warning(f"Erreur lors de l'extraction de caractéristiques temporelles pour {col}: {e}")
                    continue
        
        elif type_fe == "text":
            # Extraction de caractéristiques textuelles
            # Identifier les colonnes de texte
            if not columns:
                # Utiliser toutes les colonnes objet (texte)
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                columns = text_columns
            
            # Vérifier les colonnes valides
            valid_columns = [col for col in columns if col in df.columns and df[col].dtype == 'object']
            
            # Extraire les caractéristiques de texte pour chaque colonne
            for col in valid_columns:
                # Longueur du texte
                result_df[f"{col}_length"] = df[col].astype(str).apply(len)
                metadata["features_created"][f"{col}_length"] = {
                    "type": "text_length",
                    "source_column": col
                }
                metadata["columns_added"].append(f"{col}_length")
                
                # Nombre de mots
                result_df[f"{col}_word_count"] = df[col].astype(str).apply(lambda x: len(x.split()))
                metadata["features_created"][f"{col}_word_count"] = {
                    "type": "word_count",
                    "source_column": col
                }
                metadata["columns_added"].append(f"{col}_word_count")
        
        else:
            # Type d'ingénierie inconnu
            self.logger.warning(f"Type d'ingénierie inconnu: {type_fe}")
            return df, metadata
        
        return result_df, metadata

    def reduce_dimensions(self, df, n_components=None, variance_threshold=0.95):
        """
        Réduit la dimensionnalité des données.
        
        Args:
            df: DataFrame à traiter
            n_components: Nombre de composantes à conserver (None pour utiliser variance_threshold)
            variance_threshold: Seuil de variance à conserver (0.95 = 95%)
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "method": "pca",
            "components_kept": 0,
            "variance_explained": 0.0,
            "columns_added": [],
            "columns_removed": []
        }
        
        # Vérifier s'il y a des colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 3:  # La réduction de dimension n'a de sens qu'avec plusieurs colonnes
            return df, metadata
        
        # Création d'une copie pour éviter la modification de l'original
        result_df = df.copy()
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardiser les données avant PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Appliquer PCA
            if n_components is not None:
                pca = PCA(n_components=n_components)
            else:
                pca = PCA(n_components=variance_threshold)
                
            pca_result = pca.fit_transform(scaled_data)
            
            # Créer les nouvelles colonnes PCA
            for i in range(pca_result.shape[1]):
                new_col = f"PCA_{i+1}"
                result_df[new_col] = pca_result[:, i]
                metadata["columns_added"].append(new_col)
            
            # Mise à jour des métadonnées
            metadata["components_kept"] = pca_result.shape[1]
            metadata["variance_explained"] = float(sum(pca.explained_variance_ratio_))
            metadata["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
            
            # Supprimer les colonnes numériques originales si demandé
            # result_df = result_df.drop(columns=numeric_cols)
            # metadata["columns_removed"] = numeric_cols.tolist()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la réduction de dimensionnalité: {e}")
            # Retourner le DataFrame inchangé en cas d'erreur
        
        return result_df, metadata

    def drop_columns(self, df, columns_to_drop):
        """
        Supprime les colonnes spécifiées du DataFrame.
        
        Args:
            df: DataFrame à traiter
            columns_to_drop: Liste des colonnes à supprimer
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "columns_removed": columns_to_drop,
            "original_shape": df.shape
        }
        
        # Vérifier que les colonnes existent
        missing_columns = [col for col in columns_to_drop if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Colonnes non trouvées dans le DataFrame: {missing_columns}")
            # Filtrer pour ne garder que les colonnes existantes
            columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        # Copier le DataFrame pour ne pas modifier l'original
        result_df = df.copy()
        
        if columns_to_drop:
            # Supprimer les colonnes
            result_df = result_df.drop(columns=columns_to_drop)
            
            # Mettre à jour les métadonnées
            metadata["new_shape"] = result_df.shape
            metadata["columns_count_before"] = df.shape[1]
            metadata["columns_count_after"] = result_df.shape[1]
            metadata["columns_actually_removed"] = columns_to_drop
        else:
            self.logger.warning("Aucune colonne valide à supprimer")
            metadata["message"] = "Aucune colonne valide à supprimer"
        
        return result_df, metadata

    def merge_columns(self, df, columns_to_merge, new_column, method="concat", separator=", ", drop_original=False):
        """
        Fusionne plusieurs colonnes en une seule.
        
        Args:
            df: DataFrame à traiter
            columns_to_merge: Liste des colonnes à fusionner
            new_column: Nom de la nouvelle colonne fusionnée
            method: Méthode de fusion ('concat', 'sum', 'mean', 'max', 'min')method: Méthode de fusion ('concat', 'sum', 'mean', 'max', 'min')
            separator: Séparateur pour la concaténation (si method est 'concat')
            drop_original: Si True, supprime les colonnes originales après fusion
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "columns_merged": columns_to_merge,
            "new_column": new_column,
            "method": method,
            "columns_added": [new_column]
        }
        
        # Vérifier que les colonnes existent
        missing_columns = [col for col in columns_to_merge if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Colonnes non trouvées dans le DataFrame: {missing_columns}")
            # Filtrer pour ne garder que les colonnes existantes
            columns_to_merge = [col for col in columns_to_merge if col in df.columns]
        
        if len(columns_to_merge) < 2:
            self.logger.error("Au moins deux colonnes sont nécessaires pour la fusion")
            # Retourner le DataFrame inchangé avec un message d'erreur
            metadata["error"] = "Au moins deux colonnes sont nécessaires pour la fusion"
            return df.copy(), metadata
        
        # Copier le DataFrame pour ne pas modifier l'original
        result_df = df.copy()
        
        try:
            # Effectuer la fusion selon la méthode choisie
            if method == "concat":
                # Pour la concaténation, convertir toutes les colonnes en chaînes
                result_df[new_column] = result_df[columns_to_merge].astype(str).agg(separator.join, axis=1)
            elif method == "sum":
                # Somme (pour colonnes numériques)
                result_df[new_column] = result_df[columns_to_merge].sum(axis=1)
            elif method == "mean":
                # Moyenne (pour colonnes numériques)
                result_df[new_column] = result_df[columns_to_merge].mean(axis=1)
            elif method == "max":
                # Maximum (pour colonnes numériques)
                result_df[new_column] = result_df[columns_to_merge].max(axis=1)
            elif method == "min":
                # Minimum (pour colonnes numériques)
                result_df[new_column] = result_df[columns_to_merge].min(axis=1)
            else:
                self.logger.warning(f"Méthode de fusion non reconnue: {method}")
                metadata["error"] = f"Méthode de fusion non reconnue: {method}"
                return df.copy(), metadata
            
            # Supprimer les colonnes originales si demandé
            if drop_original:
                result_df = result_df.drop(columns=columns_to_merge)
                metadata["columns_removed"] = columns_to_merge
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la fusion des colonnes: {e}")
            metadata["error"] = str(e)
            # Retourner le DataFrame inchangé en cas d'erreur
            return df.copy(), metadata
        
        return result_df, metadata

    def replace_values(self, df, column, replacements, replace_all=True):
        """
        Remplace des valeurs spécifiques dans une colonne.
        
        Args:
            df: DataFrame à traiter
            column: Nom de la colonne à traiter
            replacements: Dictionnaire {valeur_originale: nouvelle_valeur}
            replace_all: Si True, remplace toutes les occurrences. Sinon, uniquement la première.
            
        Returns:
            tuple: (DataFrame transformé, métadonnées)
        """
        metadata = {
            "column": column,
            "replacements": replacements,
            "count_replaced": {}
        }
        
        # Vérifier que la colonne existe
        if column not in df.columns:
            self.logger.error(f"Colonne '{column}' non trouvée dans le DataFrame")
            metadata["error"] = f"Colonne '{column}' non trouvée"
            return df.copy(), metadata
        
        # Copier le DataFrame pour ne pas modifier l'original
        result_df = df.copy()
        
        try:
            # Compter les occurrences avant remplacement pour chaque valeur
            value_counts = df[column].value_counts().to_dict()
            
            # Effectuer les remplacements
            for original_val, new_val in replacements.items():
                # Gérer les valeurs NULL (None) et chaînes vides
                if original_val == "NULL":
                    original_val = None
                
                # Compter les occurrences avant remplacement
                count = value_counts.get(original_val, 0)
                
                # Remplacer les valeurs
                if replace_all:
                    result_df[column] = result_df[column].replace(original_val, new_val)
                else:
                    # Ne remplacer que la première occurrence pour chaque ligne
                    mask = result_df[column] == original_val
                    result_df.loc[mask, column] = new_val
                
                # Enregistrer le nombre de remplacements
                metadata["count_replaced"][str(original_val)] = int(count)
            
            # Résumé des modifications
            metadata["total_modified"] = sum(metadata["count_replaced"].values())
            metadata["success"] = True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du remplacement des valeurs: {e}")
            metadata["error"] = str(e)
            metadata["success"] = False
            # Retourner le DataFrame inchangé en cas d'erreur
            return df.copy(), metadata
        
        return result_df, metadata

    def analyze_changes(self, original_df, transformed_df, metadata):
        """
        Génère une analyse textuelle des changements appliqués au DataFrame.
        
        Args:
            original_df: DataFrame original
            transformed_df: DataFrame transformé
            metadata: Métadonnées des transformations
            
        Returns:
            str: Analyse textuelle
        """
        analysis = ["Rapport d'analyse des transformations appliquées:"]
        
        # Changements dans les dimensions
        original_shape = metadata["original_shape"]
        new_shape = transformed_df.shape
        
        analysis.append(f"• Dimensions: {original_shape[0]} lignes × {original_shape[1]} colonnes → {new_shape[0]} lignes × {new_shape[1]} colonnes")
        
        # Résumé des transformations
        if metadata["transformations_applied"]:
            analysis.append(f"• Transformations appliquées: {', '.join(metadata['transformations_applied'])}")
        
        # Traitement des valeurs manquantes
        if "missing_values" in metadata:
            before = metadata["missing_values"].get("before", 0)
            after = metadata["missing_values"].get("after", 0)
            if before > 0:
                analysis.append(f"• Valeurs manquantes: {before} → {after} ({100 - (after/before*100):.1f}% traitées)")
        
        # Colonnes modifiées
        if metadata.get("columns_added"):
            analysis.append(f"• Colonnes ajoutées: {len(metadata['columns_added'])}")
            analysis.append(f"  - {', '.join(metadata['columns_added'][:5])}" + ("..." if len(metadata['columns_added']) > 5 else ""))
        
        if metadata.get("columns_removed"):
            analysis.append(f"• Colonnes supprimées: {len(metadata['columns_removed'])}")
            analysis.append(f"  - {', '.join(metadata['columns_removed'][:5])}" + ("..." if len(metadata['columns_removed']) > 5 else ""))
        
        # Informations sur l'encodage
        if "encoding" in metadata["transformations_applied"]:
            for hist in self.transformations_history:
                if hist["transformation"] == "encoding":
                    encoded_count = len(hist["metadata"]["encoded_columns"])
                    analysis.append(f"• Variables catégorielles encodées: {encoded_count}")
        
        # Informations sur les outliers
        if "outliers" in metadata["transformations_applied"]:
            for hist in self.transformations_history:
                if hist["transformation"] == "outliers":
                    outlier_cols = len(hist["metadata"]["outliers_detected"])
                    if outlier_cols > 0:
                        total_outliers = sum(info["count"] for info in hist["metadata"]["outliers_detected"].values())
                        analysis.append(f"• Valeurs aberrantes traitées: {total_outliers} dans {outlier_cols} colonnes")
        
        # Informations sur la fusion de colonnes
        if "merge_columns" in metadata["transformations_applied"]:
            for hist in self.transformations_history:
                if hist["transformation"] == "merge_columns":
                    columns_merged = len(hist["metadata"]["columns_merged"])
                    new_column = hist["metadata"]["new_column"]
                    method = hist["metadata"]["method"]
                    analysis.append(f"• Fusion de colonnes: {columns_merged} colonnes fusionnées en '{new_column}' (méthode: {method})")
        
        # Informations sur le remplacement de valeurs
        if "replace_values" in metadata["transformations_applied"]:
            for hist in self.transformations_history:
                if hist["transformation"] == "replace_values":
                    if "count_replaced" in hist["metadata"]:
                        total_replaced = sum(hist["metadata"]["count_replaced"].values())
                        column = hist["metadata"]["column"]
                        analysis.append(f"• Valeurs remplacées: {total_replaced} dans la colonne '{column}'")
        
        return "\n".join(analysis)

    def get_dataframe_hash(self, df):
        """
        Génère un hash pour un DataFrame afin de l'identifier de manière unique.
        
        Args:
            df: DataFrame à identifier
            
        Returns:
            str: Hash MD5 du DataFrame
        """
        column_str = "_".join(df.columns)
        shape_str = f"{df.shape[0]}_{df.shape[1]}"
        sample_data = df.head(5).to_json()
        
        hash_input = column_str + shape_str + sample_data
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def analyze_with_ai(self, df, transformations=None, user_context='', data_transformer=None):
        """
        Analyse un DataFrame avec l'IA
        
        Args:
            df: DataFrame à analyser
            transformations: Liste des transformations à appliquer avant l'analyse
            user_context: Contexte utilisateur ou instructions
            data_transformer: Instance de DataTransformer (optionnel)
            
        Returns:
            tuple: (résultat de l'analyse, métadonnées)
        """
        try:
            # Importer DataTransformer ici pour éviter les imports circulaires
            from modules.data_transformer_module import DataTransformer
            if data_transformer is None:
                data_transformer = DataTransformer(
                    os.environ.get('AI_MODEL', 'mistral:latest'),
                    int(os.environ.get('CONTEXT_SIZE', '4096'))
                )

            # Log détaillé au début de la méthode
            self.logger.info(f"Début de l'analyse IA avec le contexte: {user_context}")
            self.logger.info(f"Nombre de transformations: {len(transformations) if transformations else 0}")
            
            # Filtrer les transformations pour exclure 'ai_analysis'
            if transformations:
                transformations = [t for t in transformations if t != 'ai_analysis']
                self.logger.info(f"Transformations à appliquer après filtrage: {transformations}")
            
            # Appliquer les transformations si nécessaire pour l'analyse
            temp_df = df.copy()
            if transformations:
                for transform in transformations:
                    self.logger.info(f"Application de la transformation: {transform}")
                    # Appliquer chaque transformation individuellement
                    if transform == "missing_values":
                        temp_df, _ = self.handle_missing_values(temp_df)
                    elif transform == "standardization":
                        temp_df, _ = self.standardize_data(temp_df)
                    elif transform == "encoding":
                        temp_df, _ = self.encode_categorical(temp_df)
                    elif transform == "outliers":
                        temp_df, _ = self.handle_outliers(temp_df)
                    elif transform == "feature_engineering":
                        temp_df, _ = self.engineer_features(temp_df)
            
            # Collecter des informations sur le dataset
            stats_info = [
                f"Dimensions: {temp_df.shape[0]} lignes, {temp_df.shape[1]} colonnes",
                f"Valeurs manquantes: {temp_df.isna().sum().sum()} ({temp_df.isna().sum().sum()/(temp_df.size)*100:.1f}%)"
            ]
            
            # Informations sur les colonnes
            numeric_cols = temp_df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = temp_df.select_dtypes(exclude=['number']).columns.tolist()
            
            # Log des informations sur le dataset
            self.logger.info(f"Statistiques du dataset: {stats_info}")
            self.logger.info(f"Colonnes numériques: {numeric_cols}")
            self.logger.info(f"Colonnes catégorielles: {categorical_cols}")
            
            # Appel de la génération d'analyse
            self.logger.info("Génération de l'analyse avec le modèle IA...")
            analysis_text = data_transformer.generate_dataset_analysis(temp_df, user_context)
            
            # Log du résultat de l'analyse
            self.logger.info(f"Longueur de l'analyse générée: {len(analysis_text) if analysis_text else 0}")
            
            if not analysis_text:
                self.logger.warning("Aucun texte d'analyse généré.")
                analysis_text = "L'analyse IA n'a pas pu être générée. Veuillez réessayer."
            
            self.logger.info("Analyse IA générée avec succès.")
            
            # Créer des métadonnées pour l'analyse
            metadata = {
                'dataset_info': {
                    'shape': temp_df.shape,
                    'columns': temp_df.columns.tolist(),
                    'numeric_columns': numeric_cols,
                    'categorical_columns': categorical_cols,
                    'missing_values': int(temp_df.isna().sum().sum())
                },
                'transformations_applied': transformations if transformations else [],
                'user_context': user_context
            }
            
            return analysis_text, metadata
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse IA: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"Une erreur s'est produite lors de l'analyse: {str(e)}", {"error": str(e)}