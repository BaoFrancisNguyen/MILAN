import pandas as pd
import numpy as np
import json
import logging
import base64
from io import BytesIO

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_visualization(df, chart_type, x_var=None, y_var=None, color_var=None, **kwargs):
    """
    Crée une visualisation basée sur le type de graphique spécifié.
    Retourne les données au format JSON pour Plotly.js.
    
    Args:
        df: DataFrame pandas
        chart_type: Type de graphique (histogramme, boxplot, etc.)
        x_var: Variable pour l'axe X
        y_var: Variable pour l'axe Y
        color_var: Variable pour la couleur
        **kwargs: Arguments supplémentaires pour la visualisation
    
    Returns:
        dict: Données de visualisation au format JSON pour Plotly.js
    """
    try:
        if chart_type == "Histogramme":
            return create_histogram(df, x_var, color_var, **kwargs)
        elif chart_type == "Boîte à moustaches":
            return create_boxplot(df, x_var, y_var, color_var, **kwargs)
        elif chart_type == "Scatter plot":
            return create_scatterplot(df, x_var, y_var, color_var, **kwargs)
        elif chart_type == "Carte de chaleur":
            return create_heatmap(df, x_var, y_var, color_var, **kwargs)
        elif chart_type == "Graphique en barres":
            return create_barplot(df, x_var, y_var, color_var, **kwargs)
        elif chart_type == "Camembert":
            return create_piechart(df, x_var, y_var, **kwargs)
        elif chart_type == "Dashboard interactif":
            return create_dashboard_data(df, **kwargs)
        else:
            logger.warning(f"Type de graphique non reconnu: {chart_type}")
            return {"error": f"Type de graphique non reconnu: {chart_type}"}
    except Exception as e:
        logger.error(f"Erreur lors de la création de la visualisation: {e}")
        return {"error": str(e)}

def create_histogram(df, x_var, color_var=None, bins=20, **kwargs):
    """
    Crée un histogramme.
    
    Args:
        df: DataFrame pandas
        x_var: Variable pour l'axe X
        color_var: Variable pour la couleur
        bins: Nombre de bins
        
    Returns:
        dict: Données pour Plotly.js
    """
    if not x_var:
        return {"error": "Variable X non spécifiée"}
    
    try:
        # Vérifier si la colonne X existe
        if x_var not in df.columns:
            return {"error": f"La colonne '{x_var}' n'existe pas dans le DataFrame"}
        
        # Convertir la colonne en numérique et supprimer les valeurs non numériques
        x_numeric = pd.to_numeric(df[x_var], errors='coerce')
        
        if x_numeric.isna().all():
            return {"error": f"La colonne '{x_var}' ne contient pas de valeurs numériques valides"}
        
        # Remplacer la colonne originale par la version numérique pour l'histogramme
        df_temp = df.copy()
        df_temp[x_var] = x_numeric
        
        data = []
        
        if color_var and color_var != "Aucune":
            # Histogramme groupé par couleur
            if color_var not in df.columns:
                return {"error": f"La colonne de couleur '{color_var}' n'existe pas dans le DataFrame"}
                
            color_categories = df[color_var].dropna().unique()
            
            for category in color_categories:
                subset = df_temp[(df_temp[color_var] == category) & (~df_temp[x_var].isna())]
                
                if subset.empty:
                    continue
                
                # Calculer l'histogramme
                hist, bin_edges = np.histogram(subset[x_var].dropna(), bins=bins)
                
                # Créer la trace
                trace = {
                    "type": "bar",
                    "x": [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)],
                    "y": hist.tolist(),
                    "name": str(category),
                    "marker": {"opacity": 0.7}
                }
                data.append(trace)
        else:
            # Histogramme simple
            numeric_values = df_temp[x_var].dropna()
            
            if numeric_values.empty:
                return {"error": f"Après filtrage des valeurs non valides, la colonne '{x_var}' est vide"}
                
            hist, bin_edges = np.histogram(numeric_values, bins=bins)
            
            trace = {
                "type": "bar",
                "x": [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)],
                "y": hist.tolist(),
                "marker": {"color": "rgba(70, 130, 180, 0.7)"}
            }
            data.append(trace)
        
        layout = {
            "title": f"Histogramme de {x_var}",
            "xaxis": {"title": x_var},
            "yaxis": {"title": "Fréquence"},
            "barmode": "overlay" if color_var else "group",
            "legend": {"orientation": "h", "y": -0.2}
        }
        
        return {
            "data": data,
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'histogramme: {e}")
        return {"error": str(e)}

def create_boxplot(df, x_var, y_var, color_var=None, **kwargs):
    """
    Crée une boîte à moustaches.
    
    Args:
        df: DataFrame pandas
        x_var: Variable pour l'axe X (groupement)
        y_var: Variable pour l'axe Y (valeurs numériques)
        color_var: Variable pour la couleur
        
    Returns:
        dict: Données pour Plotly.js
    """
    if not y_var:
        return {"error": "Variable Y non spécifiée"}
    
    try:
        # Vérifier si les colonnes existent
        if y_var not in df.columns:
            return {"error": f"La colonne Y '{y_var}' n'existe pas dans le DataFrame"}
        
        # Convertir la colonne Y en numérique
        df_temp = df.copy()
        df_temp[y_var] = pd.to_numeric(df_temp[y_var], errors='coerce')
        
        if df_temp[y_var].isna().all():
            return {"error": f"La colonne Y '{y_var}' ne contient pas de valeurs numériques valides"}
        
        data = []
        
        if x_var and x_var != "Aucune":
            # Vérifier si la colonne X existe
            if x_var not in df.columns:
                return {"error": f"La colonne X '{x_var}' n'existe pas dans le DataFrame"}
                
            # Boîte à moustaches groupée par X
            x_categories = df[x_var].dropna().unique()
            
            if color_var and color_var != "Aucune":
                # Vérifier si la colonne de couleur existe
                if color_var not in df.columns:
                    return {"error": f"La colonne de couleur '{color_var}' n'existe pas dans le DataFrame"}
                    
                # Avec coloration
                color_categories = df[color_var].dropna().unique()
                
                for color_cat in color_categories:
                    y_values = []
                    x_values = []
                    
                    for x_cat in x_categories:
                        subset = df_temp[(df_temp[x_var] == x_cat) & (df_temp[color_var] == color_cat)]
                        y_vals = subset[y_var].dropna().tolist()
                        if y_vals:  # Vérifier si la liste n'est pas vide
                            y_values.extend(y_vals)
                            x_values.extend([str(x_cat)] * len(y_vals))
                    
                    if not y_values:  # Passer cette catégorie si aucune valeur valide
                        continue
                        
                    trace = {
                        "type": "box",
                        "y": y_values,
                        "x": x_values,
                        "name": str(color_cat),
                        "marker": {"opacity": 0.7}
                    }
                    data.append(trace)
            else:
                # Sans coloration
                # Filtrer les valeurs non valides
                valid_data = df_temp[~df_temp[y_var].isna()]
                
                if valid_data.empty:
                    return {"error": "Après filtrage des valeurs non valides, aucune donnée n'est disponible"}
                    
                trace = {
                    "type": "box",
                    "y": valid_data[y_var].tolist(),
                    "x": valid_data[x_var].tolist(),
                    "marker": {"color": "rgba(70, 130, 180, 0.7)"}
                }
                data.append(trace)
        else:
            # Boîte à moustaches simple
            valid_data = df_temp[~df_temp[y_var].isna()]
            
            if valid_data.empty:
                return {"error": "Après filtrage des valeurs non valides, aucune donnée n'est disponible"}
                
            trace = {
                "type": "box",
                "y": valid_data[y_var].tolist(),
                "name": y_var,
                "marker": {"color": "rgba(70, 130, 180, 0.7)"}
            }
            data.append(trace)
        
        layout = {
            "title": f"Boîte à moustaches de {y_var}" + (f" par {x_var}" if x_var and x_var != "Aucune" else ""),
            "xaxis": {"title": x_var if x_var and x_var != "Aucune" else ""},
            "yaxis": {"title": y_var},
            "boxmode": "group",
            "legend": {"orientation": "h", "y": -0.2}
        }
        
        return {
            "data": data,
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création de la boîte à moustaches: {e}")
        return {"error": str(e)}

def create_scatterplot(df, x_var, y_var, color_var=None, size_var=None, **kwargs):
    """
    Crée un nuage de points.
    
    Args:
        df: DataFrame pandas
        x_var: Variable pour l'axe X
        y_var: Variable pour l'axe Y
        color_var: Variable pour la couleur
        size_var: Variable pour la taille des points
        
    Returns:
        dict: Données pour Plotly.js
    """
    if not x_var or not y_var:
        return {"error": "Variables X et Y non spécifiées"}
    
    try:
        # Vérifier si les colonnes existent
        if x_var not in df.columns:
            return {"error": f"La colonne X '{x_var}' n'existe pas dans le DataFrame"}
        if y_var not in df.columns:
            return {"error": f"La colonne Y '{y_var}' n'existe pas dans le DataFrame"}
        
        # Créer une copie du DataFrame pour les transformations
        df_temp = df.copy()
        
        # Convertir les colonnes en numérique
        df_temp[x_var] = pd.to_numeric(df_temp[x_var], errors='coerce')
        df_temp[y_var] = pd.to_numeric(df_temp[y_var], errors='coerce')
        
        # Vérifier s'il reste des données après conversion
        valid_mask = ~df_temp[x_var].isna() & ~df_temp[y_var].isna()
        if not valid_mask.any():
            return {"error": f"Après conversion, aucune paire de valeurs (X, Y) valide n'a été trouvée"}
        
        data = []
        
        if color_var and color_var != "Aucune":
            # Vérifier si la colonne de couleur existe
            if color_var not in df.columns:
                return {"error": f"La colonne de couleur '{color_var}' n'existe pas dans le DataFrame"}
                
            # Scatter avec coloration
            color_categories = df[color_var].dropna().unique()
            
            for category in color_categories:
                subset = df_temp[(df_temp[color_var] == category) & valid_mask]
                
                if subset.empty:
                    continue
                
                # Configuration de base
                trace = {
                    "type": "scatter",
                    "mode": "markers",
                    "x": subset[x_var].tolist(),
                    "y": subset[y_var].tolist(),
                    "name": str(category),
                    "marker": {"opacity": 0.7}
                }
                
                # Ajouter la taille variable si spécifiée
                if size_var and size_var != "Uniforme" and size_var in df.columns:
                    # Convertir la colonne de taille en numérique
                    subset[size_var] = pd.to_numeric(subset[size_var], errors='coerce')
                    sizes = subset[size_var].dropna()
                    
                    # Normaliser les tailles entre 5 et 20
                    if not sizes.empty:
                        min_size = sizes.min()
                        max_size = sizes.max()
                        if min_size != max_size:  # Éviter la division par zéro
                            normalized_sizes = 5 + ((sizes - min_size) / (max_size - min_size)) * 15
                            trace["marker"]["size"] = normalized_sizes.tolist()
                
                data.append(trace)
        else:
            # Scatter simple
            valid_data = df_temp[valid_mask]
            
            trace = {
                "type": "scatter",
                "mode": "markers",
                "x": valid_data[x_var].tolist(),
                "y": valid_data[y_var].tolist(),
                "marker": {"color": "rgba(70, 130, 180, 0.7)"}
            }
            
            # Ajouter la taille variable si spécifiée
            if size_var and size_var != "Uniforme" and size_var in df.columns:
                # Convertir la colonne de taille en numérique
                valid_data[size_var] = pd.to_numeric(valid_data[size_var], errors='coerce')
                sizes = valid_data[size_var].dropna()
                
                # Normaliser les tailles entre 5 et 20
                if not sizes.empty:
                    min_size = sizes.min()
                    max_size = sizes.max()
                    if min_size != max_size:  # Éviter la division par zéro
                        normalized_sizes = 5 + ((sizes - min_size) / (max_size - min_size)) * 15
                        trace["marker"]["size"] = normalized_sizes.tolist()
            
            data.append(trace)
        
        layout = {
            "title": f"Nuage de points: {x_var} vs {y_var}",
            "xaxis": {"title": x_var},
            "yaxis": {"title": y_var},
            "legend": {"orientation": "h", "y": -0.2}
        }
        
        return {
            "data": data,
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création du nuage de points: {e}")
        return {"error": str(e)}

def create_heatmap(df, x_var, y_var, color_var=None, **kwargs):
    """
    Crée une carte de chaleur.
    
    Args:
        df: DataFrame pandas
        x_var: Variable pour l'axe X
        y_var: Variable pour l'axe Y (valeurs à agréger)
        color_var: Non utilisé pour les cartes de chaleur
        
    Returns:
        dict: Données pour Plotly.js
    """
    try:
        # Si x_var et y_var sont spécifiés, créer un pivot
        if x_var and y_var and x_var != "Aucune" and y_var != "Aucune":
            # Vérifier si les colonnes existent
            if x_var not in df.columns:
                return {"error": f"La colonne X '{x_var}' n'existe pas dans le DataFrame"}
            if y_var not in df.columns:
                return {"error": f"La colonne Y '{y_var}' n'existe pas dans le DataFrame"}
            
            # Convertir la colonne Y en numérique si nécessaire
            df_temp = df.copy()
            if not pd.api.types.is_numeric_dtype(df[y_var]):
                df_temp[y_var] = pd.to_numeric(df_temp[y_var], errors='coerce')
                
                if df_temp[y_var].isna().all():
                    return {"error": f"La colonne Y '{y_var}' ne contient pas de valeurs numériques valides"}
            
            # Agrégation selon la moyenne par défaut
            agg_func = kwargs.get("agg_func", "mean")
            
            # Filtrer les NaN avant de créer le pivot
            df_filtered = df_temp.dropna(subset=[x_var, y_var])
            
            if df_filtered.empty:
                return {"error": "Après filtrage des valeurs non valides, aucune donnée n'est disponible"}
            
            # Créer le pivot
            try:
                if y_var != x_var:
                    pivot_df = df_filtered.pivot_table(
                        values=y_var,
                        index=y_var,
                        columns=x_var,
                        aggfunc=agg_func
                    )
                else:
                    # Si les variables X et Y sont identiques, utiliser une autre approche
                    # Par exemple, calculer une matrice de corrélation pour cette variable
                    return {"error": "Les variables X et Y ne peuvent pas être identiques pour une carte de chaleur"}
                
                # Convertir en matrice
                z_values = pivot_df.values.tolist()
                x_labels = pivot_df.columns.tolist()
                y_labels = pivot_df.index.tolist()
                
                trace = {
                    "type": "heatmap",
                    "z": z_values,
                    "x": x_labels,
                    "y": y_labels,
                    "colorscale": "Viridis"
                }
                
                layout = {
                    "title": f"Carte de chaleur: {x_var} vs {y_var} ({agg_func})",
                    "xaxis": {"title": x_var},
                    "yaxis": {"title": y_var}
                }
            except Exception as e:
                logger.error(f"Erreur lors de la création du pivot pour la carte de chaleur: {e}")
                return {"error": f"Impossible de créer une carte de chaleur avec ces variables: {str(e)}"}
            
        else:
            # Corrélation par défaut si variables non spécifiées
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) < 2:
                # Tentative de conversion des colonnes non numériques
                df_temp = df.copy()
                converted_cols = []
                
                for col in df.columns:
                    if col not in numeric_cols:
                        try:
                            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                            if not df_temp[col].isna().all():
                                converted_cols.append(col)
                        except:
                            pass
                
                # Mettre à jour la liste des colonnes numériques
                numeric_cols = df_temp.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) < 2:
                    return {"error": "Pas assez de colonnes numériques pour une carte de corrélation"}
                    
                # Utiliser le DataFrame converti
                df = df_temp
            
            # Calcul de la matrice de corrélation
            corr_matrix = df[numeric_cols].corr().round(2)
            
            z_values = corr_matrix.values.tolist()
            labels = corr_matrix.columns.tolist()
            
            trace = {
                "type": "heatmap",
                "z": z_values,
                "x": labels,
                "y": labels,
                "colorscale": "RdBu",
                "zmid": 0,  # Centre la colorscale sur 0
                "zmin": -1,
                "zmax": 1,
                "text": [[str(round(val, 2)) for val in row] for row in z_values],
                "hoverinfo": "text",
                "colorbar": {"title": "Corrélation"}
            }
            
            layout = {
                "title": "Matrice de corrélation",
                "xaxis": {"title": "Variables"},
                "yaxis": {"title": "Variables"}
            }
        
        return {
            "data": [trace],
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création de la carte de chaleur: {e}")
        return {"error": str(e)}

def create_barplot(df, x_var, y_var=None, color_var=None, **kwargs):
    """
    Crée un graphique en barres.
    
    Args:
        df: DataFrame pandas
        x_var: Variable pour l'axe X
        y_var: Variable pour l'axe Y (valeurs numériques)
        color_var: Variable pour la couleur
        
    Returns:
        dict: Données pour Plotly.js
    """
    if not x_var:
        return {"error": "Variable X non spécifiée"}
    
    try:
        # Vérifier si la colonne X existe
        if x_var not in df.columns:
            return {"error": f"La colonne X '{x_var}' n'existe pas dans le DataFrame"}
        
        data = []
        
        if y_var and y_var != "Aucune":
            # Vérifier si la colonne Y existe
            if y_var not in df.columns:
                return {"error": f"La colonne Y '{y_var}' n'existe pas dans le DataFrame"}
                
            # Convertir la colonne Y en numérique
            df_temp = df.copy()
            df_temp[y_var] = pd.to_numeric(df_temp[y_var], errors='coerce')
            
            if df_temp[y_var].isna().all():
                return {"error": f"La colonne Y '{y_var}' ne contient pas de valeurs numériques valides"}
            
            # Graphique en barres avec une variable Y numérique
            if color_var and color_var != "Aucune":
                # Vérifier si la colonne de couleur existe
                if color_var not in df.columns:
                    return {"error": f"La colonne de couleur '{color_var}' n'existe pas dans le DataFrame"}
                    
                # Avec coloration
                color_categories = df[color_var].dropna().unique()
                
                for category in color_categories:
                    subset = df_temp[(df_temp[color_var] == category) & (~df_temp[y_var].isna())]
                    
                    if subset.empty:
                        continue
                    
                    # Agréger les données
                    try:
                        grouped = subset.groupby(x_var)[y_var].mean().reset_index()
                        
                        trace = {
                            "type": "bar",
                            "x": grouped[x_var].tolist(),
                            "y": grouped[y_var].tolist(),
                            "name": str(category),
                            "marker": {"opacity": 0.7}
                        }
                        data.append(trace)
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'agrégation pour la catégorie {category}: {e}")
                        continue
            else:
                # Sans coloration
                # Filtrer les valeurs non valides
                valid_data = df_temp[~df_temp[y_var].isna()]
                
                if valid_data.empty:
                    return {"error": "Après filtrage des valeurs non valides, aucune donnée n'est disponible"}
                
                try:
                    grouped = valid_data.groupby(x_var)[y_var].mean().reset_index()
                    
                    trace = {
                        "type": "bar",
                        "x": grouped[x_var].tolist(),
                        "y": grouped[y_var].tolist(),
                        "marker": {"color": "rgba(70, 130, 180, 0.7)"}
                    }
                    data.append(trace)
                except Exception as e:
                    return {"error": f"Erreur lors de l'agrégation des données: {e}"}
            
            layout = {
                "title": f"Moyenne de {y_var} par {x_var}",
                "xaxis": {"title": x_var},
                "yaxis": {"title": f"Moyenne de {y_var}"},
                "barmode": "group",
                "legend": {"orientation": "h", "y": -0.2}
            }
        else:
            # Graphique en barres de comptage
            if color_var and color_var != "Aucune":
                # Vérifier si la colonne de couleur existe
                if color_var not in df.columns:
                    return {"error": f"La colonne de couleur '{color_var}' n'existe pas dans le DataFrame"}
                    
                # Avec coloration
                try:
                    cross_tab = pd.crosstab(df[x_var], df[color_var])
                    
                    for category in cross_tab.columns:
                        trace = {
                            "type": "bar",
                            "x": cross_tab.index.tolist(),
                            "y": cross_tab[category].tolist(),
                            "name": str(category),
                            "marker": {"opacity": 0.7}
                        }
                        data.append(trace)
                except Exception as e:
                    return {"error": f"Erreur lors de la création du tableau croisé: {e}"}
            else:
                # Sans coloration
                try:
                    counts = df[x_var].value_counts().reset_index()
                    counts.columns = [x_var, "count"]
                    
                    trace = {
                        "type": "bar",
                        "x": counts[x_var].tolist(),
                        "y": counts["count"].tolist(),
                        "marker": {"color": "rgba(70, 130, 180, 0.7)"}
                    }
                    data.append(trace)
                except Exception as e:
                    return {"error": f"Erreur lors du comptage des valeurs: {e}"}
            
            layout = {
                "title": f"Décompte de {x_var}",
                "xaxis": {"title": x_var},
                "yaxis": {"title": "Nombre d'occurrences"},
                "barmode": "group",
                "legend": {"orientation": "h", "y": -0.2}
            }
        
        return {
            "data": data,
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création du graphique en barres: {e}")
        return {"error": str(e)}


def create_piechart(df, x_var, y_var=None, **kwargs):
    """
    Crée un graphique en camembert.
    
    Args:
        df: DataFrame pandas
        x_var: Variable pour les segments
        y_var: Variable pour les valeurs (optionnel)
        
    Returns:
        dict: Données pour Plotly.js
    """
    if not x_var:
        return {"error": "Variable pour les segments non spécifiée"}
    
    try:
        # Vérifier si la colonne X existe
        if x_var not in df.columns:
            return {"error": f"La colonne '{x_var}' n'existe pas dans le DataFrame"}
        
        if y_var and y_var != "Aucune":
            # Vérifier si la colonne Y existe
            if y_var not in df.columns:
                return {"error": f"La colonne Y '{y_var}' n'existe pas dans le DataFrame"}
                
            # Convertir la colonne Y en numérique
            df_temp = df.copy()
            df_temp[y_var] = pd.to_numeric(df_temp[y_var], errors='coerce')
            
            if df_temp[y_var].isna().all():
                return {"error": f"La colonne Y '{y_var}' ne contient pas de valeurs numériques valides"}
            
            # Agréger les données par somme
            try:
                grouped = df_temp.groupby(x_var)[y_var].sum().reset_index()
                
                # Filtrer les valeurs nulles ou négatives
                grouped = grouped[grouped[y_var] > 0]
                
                if grouped.empty:
                    return {"error": "Après filtrage des valeurs non positives, aucune donnée n'est disponible"}
                
                labels = grouped[x_var].tolist()
                values = grouped[y_var].tolist()
                title = f"Répartition de {x_var} par somme de {y_var}"
            except Exception as e:
                return {"error": f"Erreur lors de l'agrégation des données: {e}"}
        else:
            # Comptage simple
            try:
                counts = df[x_var].value_counts().reset_index()
                counts.columns = [x_var, "count"]
                
                labels = counts[x_var].tolist()
                values = counts["count"].tolist()
                title = f"Répartition de {x_var}"
            except Exception as e:
                return {"error": f"Erreur lors du comptage des valeurs: {e}"}
        
        trace = {
            "type": "pie",
            "labels": labels,
            "values": values,
            "textinfo": "percent+label",
            "insidetextorientation": "radial",
            "marker": {
                "line": {"width": 2}
            }
        }
        
        # Ajouter un trou pour faire un donut chart si demandé
        hole_size = kwargs.get("hole_size", 0.3)
        if hole_size > 0:
            trace["hole"] = hole_size
        
        layout = {
            "title": title,
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.2}
        }
        
        return {
            "data": [trace],
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création du camembert: {e}")
        return {"error": str(e)}

def create_dashboard_data(df, **kwargs):
    """
    Crée les données pour un dashboard interactif.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        dict: Données pour le dashboard
    """
    try:
        # Statistiques générales
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "missing_percentage": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        }
        
        # Types de données
        dtypes = df.dtypes.astype(str).value_counts().to_dict()
        dtypes = {str(k): int(v) for k, v in dtypes.items()}
        
        # Statistiques par colonne
        column_stats = {}
        for col in df.columns:
            col_stats = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_percentage": float(df[col].isna().sum() / len(df) * 100)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Statistiques numériques
                try:
                    col_stats.update({
                        "min": float(df[col].min()) if not df[col].empty else None,
                        "max": float(df[col].max()) if not df[col].empty else None,
                        "mean": float(df[col].mean()) if not df[col].empty else None,
                        "median": float(df[col].median()) if not df[col].empty else None,
                        "std": float(df[col].std()) if not df[col].empty else None
                    })
                except Exception as e:
                    logger.warning(f"Erreur lors du calcul des statistiques pour la colonne {col}: {e}")
            else:
                # Tentative de conversion en numérique
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_col.isna().all():
                        col_stats.update({
                            "min": float(numeric_col.min()),
                            "max": float(numeric_col.max()),
                            "mean": float(numeric_col.mean()),
                            "median": float(numeric_col.median()),
                            "std": float(numeric_col.std())
                        })
                except Exception:
                    # Statistiques catégorielles si la conversion échoue
                    try:
                        value_counts = df[col].value_counts().head(5).to_dict()
                        col_stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
                        col_stats["unique_count"] = int(df[col].nunique())
                    except Exception as e:
                        logger.warning(f"Erreur lors du calcul des statistiques pour la colonne {col}: {e}")
            
            column_stats[col] = col_stats
        
        # Création des visualisations de base pour le dashboard
        visualizations = []
        
        # 1. Matrice de corrélation (si suffisamment de colonnes numériques)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            try:
                visualizations.append({
                    "id": "correlation_matrix",
                    "title": "Matrice de corrélation",
                    "type": "heatmap",
                    "data": create_heatmap(df, None, None)
                })
            except Exception as e:
                logger.warning(f"Erreur lors de la création de la matrice de corrélation: {e}")
        
        # 2. Histogrammes pour les variables numériques principales
        for col in numeric_cols[:min(3, len(numeric_cols))]:
            try:
                viz_data = create_histogram(df, col)
                if "error" not in viz_data:
                    visualizations.append({
                        "id": f"histogram_{col}",
                        "title": f"Distribution de {col}",
                        "type": "histogram",
                        "data": viz_data
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la création de l'histogramme pour {col}: {e}")
        
        # 3. Camemberts pour les variables catégorielles principales
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols[:min(3, len(categorical_cols))]:
            try:
                viz_data = create_piechart(df, col)
                if "error" not in viz_data:
                    visualizations.append({
                        "id": f"pie_{col}",
                        "title": f"Répartition de {col}",
                        "type": "pie",
                        "data": viz_data
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de la création du camembert pour {col}: {e}")
        
        return {
            "stats": stats,
            "dtypes": dtypes,
            "column_stats": column_stats,
            "visualizations": visualizations
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création des données du dashboard: {e}")
        return {"error": str(e)}

def get_df_preview(df, max_rows=10):
    """
    Renvoie un aperçu du DataFrame au format JSON.
    
    Args:
        df: DataFrame pandas
        max_rows: Nombre maximum de lignes à inclure
        
    Returns:
        dict: Aperçu du DataFrame
    """
    try:
        preview = df.head(max_rows).to_dict(orient='records')
        columns = df.columns.tolist()
        
        return {
            "columns": columns,
            "data": preview,
            "total_rows": len(df),
            "preview_rows": min(max_rows, len(df))
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'aperçu du DataFrame: {e}")
        return {"error": str(e)}

def dataframe_to_csv_download_link(df, filename="data.csv"):
    """
    Convertit un DataFrame en lien de téléchargement CSV.
    
    Args:
        df: DataFrame pandas
        filename: Nom du fichier CSV
        
    Returns:
        str: HTML de lien de téléchargement
    """
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        
        return href
    except Exception as e:
        logger.error(f"Erreur lors de la création du lien de téléchargement: {e}")
        return None

def dataframe_to_excel_download_link(df, filename="data.xlsx"):
    """
    Convertit un DataFrame en lien de téléchargement Excel.
    
    Args:
        df: DataFrame pandas
        filename: Nom du fichier Excel
        
    Returns:
        str: HTML de lien de téléchargement
    """
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
        
        return href
    except Exception as e:
        logger.error(f"Erreur lors de la création du lien de téléchargement Excel: {e}")
        return None

def plot_correlation_matrix(df):
    """
    Crée une matrice de corrélation pour les colonnes numériques.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        dict: Données pour Plotly.js
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        # Tentative de conversion des colonnes non numériques
        df_temp = df.copy()
        converted_cols = []
        
        for col in df.columns:
            if col not in numeric_cols:
                try:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                    if not df_temp[col].isna().all():
                        converted_cols.append(col)
                except:
                    pass
        
        # Mettre à jour la liste des colonnes numériques
        numeric_cols = df_temp.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return {"error": "Pas assez de colonnes numériques pour une matrice de corrélation"}
            
        # Utiliser le DataFrame converti
        df = df_temp
    
    try:
        # Calcul de la matrice de corrélation
        corr_matrix = df[numeric_cols].corr().round(2)
        
        z_values = corr_matrix.values.tolist()
        labels = corr_matrix.columns.tolist()
        
        trace = {
            "type": "heatmap",
            "z": z_values,
            "x": labels,
            "y": labels,
            "colorscale": "RdBu",
            "zmid": 0,
            "zmin": -1,
            "zmax": 1,
            "text": [[str(round(val, 2)) for val in row] for row in z_values],
            "hoverinfo": "text",
            "colorbar": {"title": "Corrélation"}
        }
        
        layout = {
            "title": "Matrice de corrélation",
            "xaxis": {"title": "Variables"},
            "yaxis": {"title": "Variables"}
        }
        
        return {
            "data": [trace],
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création de la matrice de corrélation: {e}")
        return {"error": str(e)}

def plot_missing_values(df):
    """
    Crée une visualisation des valeurs manquantes.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        dict: Données pour Plotly.js
    """
    try:
        # Calculer le nombre de valeurs manquantes par colonne
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        
        if missing.empty:
            return {"info": "Aucune valeur manquante trouvée dans le dataset"}
        
        # Créer le graphique en barres
        trace = {
            "type": "bar",
            "x": missing.index.tolist(),
            "y": missing.values.tolist(),
            "marker": {
                "color": "rgba(255, 99, 132, 0.7)",
                "line": {"color": "rgba(255, 99, 132, 1)", "width": 1.5}
            }
        }
        
        layout = {
            "title": "Valeurs manquantes par colonne",
            "xaxis": {"title": "Colonnes"},
            "yaxis": {"title": "Nombre de valeurs manquantes"},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 100}
        }
        
        return {
            "data": [trace],
            "layout": layout
        }
    except Exception as e:
        logger.error(f"Erreur lors de la création du graphique des valeurs manquantes: {e}")
        return {"error": str(e)}

def generate_report(df):
    """
    Génère un rapport complet sur le DataFrame.
    
    Args:
        df: DataFrame pandas
        
    Returns:
        dict: Rapport d'analyse complet
    """
    try:
        # Informations générales
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_cells": df.isna().sum().sum(),
            "missing_percentage": df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        }
        
        # Analyse par colonne
        columns_analysis = {}
        for col in df.columns:
            col_data = {
                "type": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_percentage": float(df[col].isna().sum() / len(df) * 100),
                "unique_values": int(df[col].nunique())
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Statistiques pour colonnes numériques
                try:
                    col_data.update({
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                        "skewness": float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                        "kurtosis": float(df[col].kurt()) if not pd.isna(df[col].kurt()) else None,
                        "is_numeric": True
                    })
                    
                    # Détection des valeurs aberrantes
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    
                    col_data["outliers"] = int(outliers)
                    col_data["outliers_percentage"] = float(outliers / len(df) * 100)
                except Exception as e:
                    logger.warning(f"Erreur lors du calcul des statistiques pour la colonne {col}: {e}")
            else:
                # Tentative de conversion en numérique
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_col.isna().all():
                        # La colonne peut être convertie en numérique
                        col_data["is_numeric"] = True
                        col_data.update({
                            "min": float(numeric_col.min()),
                            "max": float(numeric_col.max()),
                            "mean": float(numeric_col.mean()),
                            "median": float(numeric_col.median()),
                            "std": float(numeric_col.std()),
                            "skewness": float(numeric_col.skew()),
                            "kurtosis": float(numeric_col.kurt())
                        })
                        
                        # Détection des valeurs aberrantes
                        Q1 = numeric_col.quantile(0.25)
                        Q3 = numeric_col.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = ((numeric_col < lower_bound) | (numeric_col > upper_bound)).sum()
                        
                        col_data["outliers"] = int(outliers)
                        col_data["outliers_percentage"] = float(outliers / len(df) * 100)
                    else:
                        # Statistiques pour colonnes non-numériques
                        col_data["is_numeric"] = False
                        
                        # Fréquence des valeurs (top 10)
                        value_counts = df[col].value_counts().head(10).to_dict()
                        col_data["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
                except Exception:
                    # Statistiques pour colonnes non-numériques
                    col_data["is_numeric"] = False
                    
                    # Fréquence des valeurs (top 10)
                    try:
                        value_counts = df[col].value_counts().head(10).to_dict()
                        col_data["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
                    except Exception as e:
                        logger.warning(f"Erreur lors du calcul des fréquences pour la colonne {col}: {e}")
            
            columns_analysis[col] = col_data
        
        # Création d'un résumé global
        summary = {
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(exclude=['number']).columns),
            "has_missing_values": info["missing_cells"] > 0,
            "columns_with_missing": [col for col in df.columns if df[col].isna().sum() > 0],
            "dataset_completeness": float(100 - info["missing_percentage"])
        }
        
        # Suggestions et recommandations
        recommendations = []
        
        # Recommandations sur les valeurs manquantes
        if info["missing_cells"] > 0:
            recommendations.append({
                "type": "missing_values",
                "description": "Le dataset contient des valeurs manquantes qui nécessitent un traitement",
                "details": f"{info['missing_cells']} cellules ({info['missing_percentage']:.2f}%) sont manquantes",
                "suggestion": "Envisagez d'imputer les valeurs manquantes ou de supprimer les lignes/colonnes concernées"
            })
        
        # Recommandations sur les outliers
        columns_with_outliers = []
        for col, analysis in columns_analysis.items():
            if analysis.get("is_numeric", False) and analysis.get("outliers", 0) > 0:
                columns_with_outliers.append((col, analysis["outliers"], analysis["outliers_percentage"]))
        
        if columns_with_outliers:
            outliers_detail = ", ".join([f"{col}: {count} ({pct:.2f}%)" for col, count, pct in columns_with_outliers[:5]])
            if len(columns_with_outliers) > 5:
                outliers_detail += f"... et {len(columns_with_outliers) - 5} autres colonnes"
                
            recommendations.append({
                "type": "outliers",
                "description": "Certaines colonnes contiennent des valeurs aberrantes",
                "details": outliers_detail,
                "suggestion": "Examinez les valeurs aberrantes et envisagez de les traiter (winsorisation, suppression, etc.)"
            })
        
        # Recommandations sur l'encodage des données catégorielles
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        if len(categorical_cols) > 0:
            recommendations.append({
                "type": "categorical_encoding",
                "description": f"Le dataset contient {len(categorical_cols)} variables catégorielles qui devront être encodées pour des modèles ML",
                "details": f"Colonnes: {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}",
                "suggestion": "Utilisez l'encodage one-hot pour les variables avec peu de catégories, et l'encodage ordinal pour les autres"
            })
        
        # Rapport final
        report = {
            "info": info,
            "summary": summary,
            "columns": columns_analysis,
            "recommendations": recommendations
        }
        
        return report
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport: {e}")
        return {"error": str(e)}