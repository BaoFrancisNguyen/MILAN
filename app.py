import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
import os
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importer les modules personnalisés
from modules.audio_processor_module import LocalAudioProcessor
from modules.data_processor_module import DataProcessor
from modules.pdf_processor_module import PDFProcessor  
from modules.visualization_module import create_visualization
from modules.data_transformer_module import DataTransformer
from modules.history_manager_module import AnalysisHistory, PDFAnalysisHistory
from modules.signal_processor import process_audio_signal

def convert_numpy_types(obj):
    """Convertit les types NumPy en types Python standards pour la sérialisation JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# Configuration de l'application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'orbital')  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['HISTORY_FOLDER'] = 'analysis_history'

# Configuration de la session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
Session(app)

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTORY_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['HISTORY_FOLDER'], 'pdf'), exist_ok=True)

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation des gestionnaires d'historique
csv_history = AnalysisHistory(app.config['HISTORY_FOLDER'])
pdf_history = PDFAnalysisHistory(os.path.join(app.config['HISTORY_FOLDER'], 'pdf'))

# Initialisation du transformateur de données
data_transformer = DataTransformer(
    os.environ.get('AI_MODEL', 'mistral:latest'),
    int(os.environ.get('CONTEXT_SIZE', '4096'))
)

def allowed_file(filename):
    """Vérifie si le type de fichier est autorisé"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.before_request
def make_session_permanent():
    """Rend la session permanente"""
    session.permanent = True

@app.route('/apply_transformation', methods=['POST'])
def apply_transformation():
    """
    Route pour appliquer une transformation sur le DataFrame
    """
    # Vérifier l'authentification et la session
    if 'current_file' not in session or session['current_file']['type'] != 'csv':
        return jsonify({
            'success': False, 
            'error': 'Aucun fichier CSV chargé',
            'redirect': url_for('data_processing')
        }), 400

    try:
        # Récupérer les données de transformation
        transformation_data = request.json
        
        # Vérifier la présence des données requises
        if not transformation_data or 'type' not in transformation_data:
            return jsonify({
                'success': False, 
                'error': 'Données de transformation invalides'
            }), 400

        # Charger le DataFrame actuel
        file_path = session['current_file']['path']
        df = pd.read_csv(file_path)

        # Initialiser le processeur de données
        processor = DataProcessor()

        # Préparer les paramètres de transformation
        transform_params = {
            transformation_data['type']: transformation_data.get('details', {})
        }

        # Appliquer la transformation
        try:
            transformed_df, metadata = processor.process_dataframe(
                df, 
                transformations=transform_params
            )

            # Générer un nom de fichier unique pour la version transformée
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"transformed_{timestamp}_{os.path.basename(file_path)}"
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            # Sauvegarder le DataFrame transformé
            transformed_df.to_csv(new_file_path, index=False)

            # Mettre à jour la session avec le nouveau fichier
            session['current_file'] = {
                'name': new_filename,
                'path': new_file_path,
                'type': 'csv'
            }
            session.modified = True

            # Gérer l'historique des transformations
            if 'transformations_history' not in session:
                session['transformations_history'] = []
            
            session['transformations_history'].append({
                'type': transformation_data['type'],
                'details': metadata,
                'timestamp': timestamp
            })

            # Préparer la réponse
            return jsonify({
                'success': True,
                'metadata': {
                    'transformation_type': transformation_data['type'],
                    'details': metadata,
                    'new_file': new_filename
                },
                'redirect': url_for('data_preview')  # Rediriger vers l'aperçu
            })

        except Exception as transform_error:
            logger.error(f"Erreur de transformation : {transform_error}")
            return jsonify({
                'success': False, 
                'error': f'Erreur lors de la transformation : {str(transform_error)}'
            }), 500

    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        return jsonify({
            'success': False, 
            'error': f'Erreur inattendue : {str(e)}'
        }), 500

@app.route('/get_transformations', methods=['GET'])
def get_transformations():
    """
    Récupère l'historique des transformations
    """
    return jsonify({
        'history': session.get('transformations_history', [])
    })

@app.route('/rollback_transformation', methods=['POST'])
def rollback_transformation():
    """
    Annule la dernière transformation
    """
    try:
        # Vérifier s'il y a un historique de transformations
        if 'transformations_history' not in session or not session['transformations_history']:
            return jsonify({
                'success': False,
                'error': 'Aucune transformation à annuler'
            }), 400

        # Supprimer la dernière transformation
        last_transformation = session['transformations_history'].pop()
        session.modified = True

        # Charger le fichier précédent (si possible)
        if len(session['transformations_history']) > 0:
            # Revenir au fichier avant la dernière transformation
            previous_file = os.path.join(
                app.config['UPLOAD_FOLDER'], 
                session['transformations_history'][-1]['details'].get('original_filename', '')
            )
        else:
            # Revenir au fichier original
            previous_file = session.get('original_file', {}).get('path')

        if not previous_file or not os.path.exists(previous_file):
            return jsonify({
                'success': False,
                'error': 'Impossible de restaurer le fichier précédent'
            }), 500

        # Mettre à jour le fichier courant
        session['current_file'] = {
            'name': os.path.basename(previous_file),
            'path': previous_file,
            'type': 'csv'
        }
        session.modified = True

        return jsonify({
            'success': True,
            'message': 'Dernière transformation annulée',
            'redirect': url_for('data_preview')
        })

    except Exception as e:
        logger.error(f"Erreur lors de l'annulation : {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# 📌 ROUTES PRINCIPALES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_processing', methods=['GET', 'POST'])
def data_processing():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Aucun fichier trouvé', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Aucun fichier sélectionné', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            session['current_file'] = {'name': filename, 'path': file_path, 'type': 'csv'}

            try:
                # Code pour charger le CSV avec le délimiteur
                delimiter = request.form.get('delimiter', 'auto')
                if delimiter == 'auto':
                    df = pd.read_csv(file_path, sep=None, engine='python')
                else:
                    df = pd.read_csv(file_path, sep=delimiter)
                
                # Convertir les types NumPy pour la sérialisation JSON
                df_info = {
                    'shape': list(df.shape),
                    'columns': df.columns.tolist(),
                    'dtypes': {col: str(df[col].dtype) for col in df.columns},
                    'missing_values': int(df.isna().sum().sum()),
                    'has_numeric': bool(any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns))
                }
                
                print(f"Colonnes détectées: {df.columns.tolist()}")
                # Convertir toutes les valeurs NumPy
                session['df_info'] = convert_numpy_types(df_info)
                
                flash(f'Fichier "{filename}" chargé avec succès!', 'success')
                return redirect(url_for('data_preview'))
            except Exception as e:
                logger.error(f"Erreur lors du chargement du fichier: {e}")
                flash(f'Erreur lors du chargement du fichier: {e}', 'error')
                return redirect(request.url)  # Ajoutez cette ligne pour gérer l'erreur
    
    # N'oubliez pas de retourner quelque chose à la fin de la fonction
    return render_template('data_processing.html')

@app.route('/data_preview')
def data_preview():
    if 'current_file' not in session or session['current_file']['type'] != 'csv':
        flash('Veuillez d\'abord charger un fichier CSV', 'warning')
        return redirect(url_for('data_processing'))

    file_path = session['current_file']['path']

    try:
        df = pd.read_csv(file_path)
        preview_data = df.head(100)
        
        # Récupérer les informations sur le dataframe
        df_info = session['df_info']
        
        # Identifier et compter les colonnes numériques
        numeric_columns = []
        categorical_columns = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        # Mettre à jour les informations
        df_info['numeric_count'] = len(numeric_columns)
        df_info['has_numeric'] = len(numeric_columns) > 0
        df_info['numeric_columns'] = numeric_columns
        df_info['categorical_columns'] = categorical_columns
        
        # Sauvegarder les modifications dans la session
        session['df_info'] = df_info

        return render_template(
            'data_preview.html',
            filename=session['current_file']['name'],
            df_info=df_info,
            preview_data=preview_data.to_html(classes='table table-striped table-hover', index=False),
            columns=df.columns.tolist(),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'aperçu: {e}")
        flash(f'Erreur lors de la génération de l\'aperçu: {e}', 'error')
        return redirect(url_for('data_processing'))

# Ajouter cette route pour obtenir les valeurs uniques d'une colonne
@app.route('/api/column_values', methods=['POST'])
def get_column_values():
    """
    API pour récupérer les valeurs uniques d'une colonne avec leurs fréquences.
    """
    if 'current_file' not in session or session['current_file']['type'] != 'csv':
        return jsonify({"error": "Aucun fichier CSV chargé"}), 400
    
    # Récupérer les paramètres de la requête
    params = request.json
    column_name = params.get('column_name')
    
    if not column_name:
        return jsonify({"error": "Nom de colonne non spécifié"}), 400
    
    try:
        # Charger le DataFrame
        file_path = session['current_file']['path']
        df = pd.read_csv(file_path)
        
        # Vérifier si la colonne existe
        if column_name not in df.columns:
            return jsonify({"error": f"Colonne '{column_name}' introuvable"}), 404
        
        # Calculer les valeurs uniques et leurs fréquences
        value_counts = df[column_name].value_counts(dropna=False).reset_index()
        value_counts.columns = ['value', 'count']
        
        # Gérer les valeurs NULL explicitement
        null_rows = value_counts[value_counts['value'].isna()]
        if not null_rows.empty:
            # Remplacer NaN par "NULL" pour le JSON
            null_idx = null_rows.index[0]
            value_counts.at[null_idx, 'value'] = "NULL"
        
        # Convertir en liste de dictionnaires pour le JSON
        values_data = []
        for _, row in value_counts.iterrows():
            # Gérer les différents types de données
            if pd.isna(row['value']):
                value = "NULL"
            elif row['value'] == "":
                value = ""  # Chaîne vide
            else:
                value = row['value']
            
            values_data.append({
                "value": value,
                "count": int(row['count'])
            })
        
        # Ajouter des statistiques supplémentaires
        stats = {
            "total_rows": len(df),
            "unique_values": df[column_name].nunique(dropna=False),
            "missing_values": int(df[column_name].isna().sum()),
            "data_type": str(df[column_name].dtype)
        }
        
        return jsonify({
            "values": values_data,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des valeurs uniques: {e}")
        return jsonify({"error": str(e)}), 500

# API pour générer une visualisation
@app.route('/api/generate_visualization', methods=['POST'])
def generate_visualization():
    """
    API pour générer une visualisation à partir des paramètres fournis.
    """
    if 'current_file' not in session or session['current_file']['type'] != 'csv':
        return jsonify({"error": "Aucun fichier CSV chargé"}), 400
    
    # Récupérer les paramètres de la requête
    params = request.json
    chart_type = params.get('chart_type')
    x_var = params.get('x_var')
    y_var = params.get('y_var')
    color_var = params.get('color_var')
    
    # Paramètres optionnels
    additional_params = {}
    for key, value in params.items():
        if key not in ['chart_type', 'x_var', 'y_var', 'color_var']:
            additional_params[key] = value
    
    try:
        # Charger le DataFrame
        file_path = session['current_file']['path']
        df = pd.read_csv(file_path)
        
        # Générer la visualisation
        visualization_data = create_visualization(
            df, 
            chart_type, 
            x_var=x_var, 
            y_var=y_var, 
            color_var=color_var, 
            **additional_params
        )
        
        return jsonify(visualization_data)
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la visualisation: {e}")
        return jsonify({"error": str(e)}), 500
    

# Vérifiez dans app.py que vous avez bien la route suivante :
@app.route('/visualizations')
def visualizations():
    if 'current_file' not in session:
        flash('Veuillez d\'abord charger un fichier', 'warning')
        return redirect(url_for('index'))

    file_path = session['current_file']['path']
    file_type = session['current_file']['type']

    if file_type != 'csv':
        flash('Les visualisations ne sont disponibles que pour les fichiers CSV', 'warning')
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(file_path)
        return render_template(
            'visualizations.html',
            filename=session['current_file']['name'],
            columns=df.columns.tolist(),
            numeric_columns=df.select_dtypes(include=['number']).columns.tolist(),
            categorical_columns=df.select_dtypes(exclude=['number']).columns.tolist()
        )
    except Exception as e:
        logger.error(f"Erreur lors du chargement pour visualisation: {e}")
        flash(f'Erreur lors du chargement pour visualisation: {e}', 'error')
        return redirect(url_for('index'))

# 📌 ROUTE : Transformation des données CSV
@app.route('/data_transform', methods=['GET', 'POST'])
def data_transform():
    """Page de transformation des données"""
    if 'current_file' not in session or session['current_file']['type'] != 'csv':
        flash('Veuillez d\'abord charger un fichier CSV', 'warning')
        return redirect(url_for('data_processing'))

    analysis_result = None  # Variable pour stocker le résultat de l'analyse IA
    transformations_data = {}  # Variable pour stocker les détails des transformations

    logger.info("Début de la route data_transform")
    logger.info(f"Méthode de requête : {request.method}")

    if request.method == 'POST':
        logger.info("Contenu complet du formulaire :")
        transformations = request.form.getlist('transformations')
        user_context = request.form.get('user_context', '')
        is_ai_analysis = request.form.get('is_ai_analysis') == 'true'

        # Extraire les détails des transformations
        for transform in transformations:
            if transform != 'ai_analysis':
                transform_data = {}
                
                # Filtrer les clés spécifiques à cette transformation
                transform_keys = [key for key in request.form.keys() if key.startswith(f"{transform}_")]
                
                for key in transform_keys:
                    # Supprimer le préfixe de transformation
                    param_name = key[len(transform)+1:]
                    values = request.form.getlist(key)
                    
                    # Si c'est un tableau, le convertir en liste
                    if param_name.endswith('[]'):
                        param_name = param_name[:-2]
                        transform_data[param_name] = values
                    # Sinon, prendre la première valeur
                    else:
                        transform_data[param_name] = values[0] if values else None
                
                # Traitement spécial pour les remplacements de valeurs
                if transform == 'replace_values':
                    # Récupérer les valeurs originales et nouvelles
                    original_values = request.form.getlist('original_values[]')
                    new_values = request.form.getlist('new_values[]')
                    
                    # Créer un dictionnaire de remplacements
                    replacements = {}
                    for i in range(min(len(original_values), len(new_values))):
                        if original_values[i]:  # Ignorer les entrées vides
                            replacements[original_values[i]] = new_values[i]
                    
                    transform_data['replacements'] = replacements
                    transform_data['replace_all'] = request.form.get('replace_all_occurrences') == 'on'
                
                # Ne stocker que les transformations non vides
                if transform_data:
                    transformations_data[transform] = transform_data

        #log pour déboguer
        logger.info(f"Transformations détectées : {transformations_data}")

        file_path = session['current_file']['path']

        try:
            df = pd.read_csv(file_path)
            processor = DataProcessor()

            # Si c'est une analyse IA, effectuer l'analyse en conservant les transformations
            if is_ai_analysis and user_context:
                # Appeler la méthode d'analyse IA
                analysis_result, metadata = processor.analyze_with_ai(df, transformations, user_context)
                logger.info(f"résultats d'analyse {analysis_result}")
                
                # Sauvegarder l'analyse dans l'historique si demandé
                if request.form.get('save_history') == 'true':
                    dataset_name = session['current_file']['name']
                    csv_history.add_analysis(
                        dataset_name=dataset_name,
                        dataset_description=f"Analyse IA: {user_context[:50]}{'...' if len(user_context) > 50 else ''}",
                        analysis_text=analysis_result,
                        metadata={'transformations': transformations, 'user_context': user_context}
                    )
                    flash('Analyse IA sauvegardée dans l\'historique!', 'success')
                
                # Ne pas rediriger - continuer pour afficher la page avec les résultats
            
            # Si ce n'est pas une analyse IA ou si on veut appliquer les transformations
            elif not is_ai_analysis:
                transformed_df, metadata = processor.process_dataframe(df, transformations_data, user_context)
                
                # Sauvegarder le DataFrame transformé
                transformed_filename = f"transformed_{session['current_file']['name']}"
                transformed_path = os.path.join(app.config['UPLOAD_FOLDER'], transformed_filename)
                transformed_df.to_csv(transformed_path, index=False)
                
                # Mettre à jour les informations du fichier en session
                session['current_file'] = {
                    'name': transformed_filename,
                    'path': transformed_path,
                    'type': 'csv'
                }
                
                # Effacer les transformations car elles ont été appliquées
                transformations_data = {}
                
                # Sauvegarder dans l'historique si demandé
                if request.form.get('save_history') == 'true':
                    dataset_name = session['current_file']['name']
                    analysis_text = metadata.get('analysis', 'Analyse non disponible')
                    csv_history.add_analysis(
                        dataset_name=dataset_name,
                        dataset_description=f"Transformation avec {len(transformations)} opérations",
                        analysis_text=analysis_text,
                        metadata={'transformations': transformations}
                    )
                
                flash('Transformations appliquées avec succès!', 'success')
                return redirect(url_for('data_preview'))
                
        except Exception as e:
            logger.error(f"Erreur lors de la transformation/analyse: {e}")
            flash(f'Erreur lors de la transformation/analyse: {e}', 'error')

    # Récupérer les données pour le formulaire - méthode GET ou après analyse IA
    try:
        # Charger le fichier CSV
        file_path = session['current_file']['path']
        df = pd.read_csv(file_path)
        
        # Récupérer df_info de la session ou le recréer
        if 'df_info' in session:
            df_info = session['df_info']
        else:
            # Créer les informations de base sur le dataframe
            df_info = {
                'shape': list(df.shape),
                'columns': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'missing_values': int(df.isna().sum().sum())
            }
        
        # Identifier les colonnes numériques et catégorielles
        numeric_columns = []
        categorical_columns = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        # Mettre à jour les informations
        df_info['numeric_count'] = len(numeric_columns)
        df_info['has_numeric'] = len(numeric_columns) > 0
        
        # Passer les variables au template, y compris le résultat d'analyse IA s'il existe
        return render_template(
            'data_transform.html', 
            df_info=df_info,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            analysis_result=analysis_result,  # Passer le résultat d'analyse
            transformations_data=transformations_data  # Passer les détails des transformations
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement pour transformation: {e}")
        flash(f'Erreur lors du chargement pour transformation: {e}', 'error')
        return redirect(url_for('data_preview'))
    


# 📌 ROUTE : Page de traitement des PDF
@app.route('/pdf_processing', methods=['GET', 'POST'])
def pdf_processing():
    return render_template('pdf_processing.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """Traitement du fichier PDF via le formulaire principal"""
    if 'pdf_file' not in request.files:
        flash('Aucun fichier trouvé', 'error')
        return redirect(url_for('pdf_processing'))
    
    file = request.files['pdf_file']
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('pdf_processing'))
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Stocker les informations du fichier en session
    session['current_file'] = {'name': filename, 'path': file_path, 'type': 'pdf'}
    
    # Récupérer le type d'analyse demandé
    analysis_type = request.form.get('analysis_type', 'summary')
    
    try:
        processor = PDFProcessor()
        pdf_result = processor.process_pdf(file_path, context=analysis_type)
        
        if not pdf_result.get("success", False):
            flash(f"Erreur lors de l'analyse: {pdf_result.get('error', 'Erreur inconnue')}", 'error')
            return redirect(url_for('pdf_processing'))
        
        # Stocker les résultats en session
        session['pdf_analysis'] = pdf_result
        
        # Enregistrer dans l'historique si réussi
        if pdf_result.get("success", False):
            pdf_history.add_pdf_analysis(
                pdf_id=pdf_result.get("pdf_id", str(uuid.uuid4())),
                pdf_name=filename,
                analysis_result=pdf_result.get("analysis", {}),
                metadata=pdf_result.get("metadata", {})
            )
        
        return redirect(url_for('pdf_results'))
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du PDF: {e}")
        flash(f"Erreur lors du traitement: {e}", 'error')
        return redirect(url_for('pdf_processing'))

# 📌 ROUTE : Analyse des PDFs
@app.route('/pdf_analysis', methods=['GET', 'POST'])
def pdf_analysis():
    # Si c'est une méthode GET, rediriger vers le formulaire
    if request.method == 'GET':
        return redirect(url_for('pdf_processing'))
    
    # Le reste du code reste identique
    logger.info("📥 Requête reçue pour analyse PDF")

    if 'file' not in request.files:
        logger.error("❌ Aucun fichier reçu")
        return jsonify({"error": "Aucun fichier reçu"}), 400
    
    # ... le reste de votre code ...

    file = request.files['file']

    if file.filename == '':
        logger.error("❌ Aucun fichier sélectionné")
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    logger.info(f"✅ Fichier reçu : {file.filename}")


    # Sauvegarde du fichier
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logger.info(f"✅ Fichier enregistré sous : {file_path}")

    # Vérifier si le fichier est bien un PDF
    is_valid_pdf = False
    with open(file_path, 'rb') as f:
        if f.read(4).startswith(b'%PDF'):
            is_valid_pdf = True

    if not is_valid_pdf:
        logger.error("❌ Le fichier ne semble pas être un PDF valide")
        return jsonify({"error": "Le fichier ne semble pas être un PDF valide"}), 400

    # Charger le contexte utilisateur
    user_context = request.form.get('user_context', '')

    try:
        processor = PDFProcessor()
        logger.info("🛠 Début de l'analyse du PDF...")

        pdf_result = processor.process_pdf(file_path, context=user_context)

        # Vérifier si l'analyse a réussi
        if not pdf_result.get("success", False):
            logger.error(f"❌ Échec de l'analyse : {pdf_result.get('error', 'Erreur inconnue')}")
            return jsonify({"error": pdf_result.get("error", "Erreur inconnue")}), 500

        # Ajouter cette partie ici:
        session['current_file'] = {'name': filename, 'path': file_path, 'type': 'pdf'}
        session['pdf_analysis'] = pdf_result
        
        # Enregistrer dans l'historique
        if pdf_result.get("success", False):
            pdf_id = pdf_result.get("pdf_id", str(uuid.uuid4()))
            pdf_history.add_pdf_analysis(
                pdf_id=pdf_id,
                pdf_name=filename,
                analysis_result=pdf_result.get("analysis", {}),
                metadata=pdf_result.get("metadata", {})
            )
        
        logger.info("✅ Analyse terminée avec succès !")
        return jsonify(pdf_result)

    except Exception as e:
        logger.exception("❌ Erreur critique lors de l'analyse du PDF")
        return jsonify({"error": f"Exception: {str(e)}"}), 500

# 📌 ROUTE : Affichage des résultats PDF
@app.route('/pdf_results')
def pdf_results():
    if 'pdf_analysis' not in session:
        flash('Aucune analyse PDF en cours', category='warning')
        return redirect(url_for('pdf_analysis'))

    analysis = session['pdf_analysis']
    filename = session['current_file']['name']

    return render_template(
        'pdf_results.html',
        filename=filename,
        metadata=analysis['metadata'],
        analysis=analysis['analysis'],
        tables=analysis['tables']
    )

# 📌 ROUTES HISTORIQUE ET PARAMÈTRES
@app.route('/history')
def history():
    csv_analyses = csv_history.get_recent_analyses(limit=10)
    pdf_analyses = pdf_history.get_recent_pdf_analyses(limit=10)
    
    # Séparer les analyses normales des analyses IA pour mieux les présenter
    ai_analyses = []
    regular_analyses = []
    
    for analysis in csv_analyses:
        if "Analyse IA" in analysis.get("dataset_description", ""):
            ai_analyses.append(analysis)
        else:
            regular_analyses.append(analysis)
    
    return render_template(
        'history.html',
        csv_analyses=regular_analyses,
        ai_analyses=ai_analyses,
        pdf_analyses=pdf_analyses
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        session['settings'] = {
            'model_name': request.form.get('model_name', 'mistral:latest'),
            'context_size': int(request.form.get('context_size', '4096')),
            'use_history': request.form.get('use_history') == 'true',
            'max_history': int(request.form.get('max_history', '3'))
        }
        
        flash('Paramètres mis à jour avec succès!', 'success')
        return redirect(url_for('settings'))

    current_settings = session.get('settings', {
        'model_name': os.environ.get('AI_MODEL', 'mistral:latest'),
        'context_size': int(os.environ.get('CONTEXT_SIZE', '4096')),
        'use_history': True,
        'max_history': 3
    })
    
    return render_template('settings.html', settings=current_settings)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    # Logique pour effacer l'historique ici
    csv_history.clear_history()  # Supposons que vous ayez une méthode pour cela
    pdf_history.clear_history()  # De même pour l'historique PDF
    flash('L\'historique a été effacé avec succès.', 'success')
    return redirect(url_for('settings'))

@app.route('/cartography')
def cartography():
    """Page de cartographie pour visualiser les données géoréférencées."""
    if 'current_file' not in session:
        flash('Veuillez d\'abord charger un fichier', 'warning')
        return redirect(url_for('index'))

    file_path = session['current_file']['path']
    file_type = session['current_file']['type']

    if file_type != 'csv':
        flash('La cartographie n\'est disponible que pour les fichiers CSV', 'warning')
        return redirect(url_for('index'))

    try:
        # Récupérer les informations du fichier depuis la session
        df_info = session.get('df_info', {})
        
        # Utiliser le délimiteur stocké dans la session ou utiliser un point-virgule par défaut
        delimiter = df_info.get('delimiter_used', ';')
        
        logger.info(f"Tentative de chargement du fichier avec délimiteur '{delimiter}'")
        
        # Charger le DataFrame avec le bon délimiteur
        try:
            # Tenter d'abord avec le délimiteur stocké
            df = pd.read_csv(file_path, sep=delimiter)
            
            # Vérifier que plusieurs colonnes sont détectées
            if len(df.columns) <= 1 and delimiter != ';':
                # Si une seule colonne est détectée, essayer avec le point-virgule
                logger.info("Une seule colonne détectée, tentative avec délimiteur ';'")
                df = pd.read_csv(file_path, sep=';')
                delimiter = ';'
        except Exception as e:
            # En cas d'erreur, essayer avec le point-virgule
            logger.warning(f"Erreur avec le délimiteur initial: {e}")
            df = pd.read_csv(file_path, sep=';')
            delimiter = ';'
        
        # Vérifier à nouveau le nombre de colonnes
        if len(df.columns) <= 1:
            # Si toujours une seule colonne, essayer avec l'auto-détection
            logger.warning("Toujours une seule colonne détectée, tentative avec auto-détection")
            df = pd.read_csv(file_path, sep=None, engine='python')
            delimiter = 'auto'
        
        # Déterminer les colonnes potentiellement géographiques
        geo_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['lat', 'latitude', 'lng', 'longitude', 'lon', 'coord', 'gps', 'x', 'y']):
                geo_columns.append(col)
        
        # Déterminer les colonnes numériques (pour des visualisations potentielles)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Si aucune colonne numérique n'est détectée, essayer de convertir
        if not numeric_columns:
            logger.info("Aucune colonne numérique détectée, tentative de conversion")
            for col in df.columns:
                try:
                    # Ignorer les colonnes déjà identifiées comme géographiques
                    if col not in geo_columns:
                        # Essayer de convertir en numérique
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        # Si au moins 50% des valeurs sont numériques, considérer comme colonne numérique
                        if numeric_series.notna().mean() >= 0.5:
                            numeric_columns.append(col)
                except:
                    pass
        
        # Journaliser les informations pour le débogage
        logger.info(f"Délimiteur utilisé: '{delimiter}'")
        logger.info(f"Nombre total de colonnes: {len(df.columns)}")
        logger.info(f"Colonnes: {df.columns.tolist()}")
        logger.info(f"Colonnes géographiques probables: {geo_columns}")
        logger.info(f"Colonnes numériques: {numeric_columns}")
        
        # Retourner le template avec les colonnes disponibles
        return render_template(
            'cartography.html',
            filename=session['current_file']['name'],
            columns=df.columns.tolist(),
            geo_columns=geo_columns,
            numeric_columns=numeric_columns,
            num_rows=len(df),
            delimiter_used=delimiter
        )
    except Exception as e:
        logger.error(f"Erreur lors du chargement pour cartographie: {e}", exc_info=True)
        flash(f'Erreur lors du chargement pour cartographie: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/api/generate_map', methods=['POST'])
def generate_map():
    """API pour générer des données cartographiques."""
    if 'current_file' not in session or session['current_file']['type'] != 'csv':
        return jsonify({"error": "Aucun fichier CSV chargé"}), 400
    
    # Récupérer les paramètres de la requête
    params = request.json
    lat_col = params.get('lat_col')
    lng_col = params.get('lng_col')
    label_col = params.get('label_col')
    
    if not lat_col or not lng_col:
        return jsonify({"error": "Veuillez sélectionner les colonnes de latitude et longitude"}), 400
    
    try:
        # Charger le DataFrame
        file_path = session['current_file']['path']
        
        # Essayer avec une virgule comme délimiteur (pour les fichiers CSV standards)
        try:
            df = pd.read_csv(file_path, sep=',')
            logger.info(f"Fichier chargé avec délimiteur ',' - {len(df.columns)} colonnes détectées")
        except Exception as e:
            # Si ça échoue, essayer avec le point-virgule
            logger.warning(f"Échec de chargement avec ',': {e}")
            df = pd.read_csv(file_path, sep=';')
            logger.info(f"Fichier chargé avec délimiteur ';' - {len(df.columns)} colonnes détectées")
        
        # Vérifier les colonnes disponibles (pour le débogage)
        logger.info(f"Colonnes disponibles: {df.columns.tolist()}")
        
        # Créer un dictionnaire pour faire correspondre les noms de colonnes insensibles à la casse
        col_map = {col.lower(): col for col in df.columns}
        
        # Trouver les colonnes réelles correspondant aux noms demandés
        real_lat_col = col_map.get(lat_col.lower())
        real_lng_col = col_map.get(lng_col.lower())
        
        # Si les colonnes exactes ne sont pas trouvées, chercher des alternatives
        if not real_lat_col:
            # Essayer de trouver une colonne qui contient "lat"
            for col in df.columns:
                if "lat" in col.lower():
                    real_lat_col = col
                    break
        
        if not real_lng_col:
            # Essayer de trouver une colonne qui contient "lon"
            for col in df.columns:
                if any(x in col.lower() for x in ["lon", "lng", "long"]):
                    real_lng_col = col
                    break
        
        # Vérifier si on a trouvé les colonnes
        if not real_lat_col:
            return jsonify({"error": f"Colonne de latitude '{lat_col}' introuvable. Colonnes disponibles: {', '.join(df.columns)}"}), 400
        if not real_lng_col:
            return jsonify({"error": f"Colonne de longitude '{lng_col}' introuvable. Colonnes disponibles: {', '.join(df.columns)}"}), 400
        
        # Utiliser les noms réels des colonnes
        lat_col = real_lat_col
        lng_col = real_lng_col
        
        # Traiter la colonne d'étiquette
        real_label_col = None
        if label_col and label_col != "Aucune":
            real_label_col = col_map.get(label_col.lower())
            if not real_label_col:
                # Si l'étiquette n'est pas trouvée, utiliser la première colonne de texte disponible
                for col in df.columns:
                    if col != lat_col and col != lng_col and df[col].dtype == 'object':
                        real_label_col = col
                        break
            label_col = real_label_col
        
        # Convertir les colonnes de coordonnées en numérique
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lng_col] = pd.to_numeric(df[lng_col], errors='coerce')
        
        # Filtrer les valeurs non valides
        valid_mask = ~df[lat_col].isna() & ~df[lng_col].isna()
        
        # Ajouter des vérifications de plage pour les coordonnées
        valid_mask = valid_mask & (df[lat_col] >= -90) & (df[lat_col] <= 90)
        valid_mask = valid_mask & (df[lng_col] >= -180) & (df[lng_col] <= 180)
        
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            return jsonify({
                "error": "Aucune coordonnée valide trouvée. Vérifiez que vos colonnes contiennent bien des coordonnées géographiques."
            }), 400
        
        # Informations pour le débogage
        logger.info(f"Points valides: {len(df_valid)} sur {len(df)} lignes")
        logger.info(f"Colonnes utilisées - Latitude: {lat_col}, Longitude: {lng_col}, Étiquette: {label_col}")
        
        # Préparer les données pour la carte (format GeoJSON)
        features = []
        for idx, row in df_valid.iterrows():
            try:
                lat = float(row[lat_col])
                lng = float(row[lng_col])
                
                # Propriétés de base
                properties = {
                    "id": int(idx),
                    "lat": lat,
                    "lng": lng
                }
                
                # Ajouter l'étiquette si spécifiée
                if label_col and label_col != "Aucune":
                    properties["label"] = str(row[label_col]) if pd.notna(row[label_col]) else f"Point {idx}"
                else:
                    properties["label"] = f"Point {idx}"
                
                # Ajouter d'autres informations utiles
                for col in df.columns:
                    if col != lat_col and col != lng_col and col != label_col:
                        properties[col] = str(row[col]) if pd.notna(row[col]) else ""
                
                # Ajouter le point au GeoJSON
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lng, lat]
                    },
                    "properties": properties
                }
                
                features.append(feature)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du point {idx}: {e}")
                continue
        
        if not features:
            return jsonify({
                "error": "Impossible de générer des points valides avec les colonnes sélectionnées."
            }), 400
        
        # Calculer le centre de la carte
        lats = [f["geometry"]["coordinates"][1] for f in features]
        lngs = [f["geometry"]["coordinates"][0] for f in features]
        
        center = [
            sum(lngs) / len(lngs),
            sum(lats) / len(lats)
        ]
        
        # Renvoyer les données
        return jsonify({
            "map_data": {
                "type": "FeatureCollection",
                "features": features
            },
            "center": center,
            "stats": {
                "total_points": len(df),
                "valid_points": len(features),
                "invalid_points": len(df) - len(features)
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la carte: {e}", exc_info=True)
        return jsonify({"error": f"Erreur: {str(e)}"}), 500

# 📌 ROUTE : Transcription audio

@app.route('/audio_transcription', methods=['GET', 'POST'])
def audio_transcription():
    if request.method == 'POST':
        # Vérifier si un fichier audio est uploadé
        if 'audio_file' not in request.files:
            flash('Aucun fichier audio trouvé', 'error')
            return redirect(request.url)
        
        audio_file = request.files['audio_file']
        target_language = request.form.get('target_language', None)
        
        # Sauvegarder temporairement le fichier
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(file_path)
        
        try:
            # Chemins à configurer selon votre installation
            WHISPER_MODEL = 'base'  # ou chemin local
            TRANSLATION_MODEL = 'Mistral:latest'
            
            # Initialiser le processeur audio
            processor = LocalAudioProcessor(
                whisper_model_size='base',
                translation_model_path= 'Mistral:latest'
            )
            
            # Traiter l'audio
            result = processor.process_audio(
                file_path, 
                target_language=target_language
            )
            
            # Nettoyer le fichier temporaire
            os.remove(file_path)
            
            return render_template(
                'audio_transcription_results.html', 
                result=result
            )
        
        except Exception as e:
            logger.error(f"Erreur de transcription : {e}")
            flash(f'Erreur lors du traitement audio : {e}', 'error')
            return redirect(request.url)
    
    # Page de formulaire pour l'upload
    return render_template('audio_transcription.html')


@app.template_filter('wordcount')
def wordcount(text):
    """Compte le nombre de mots dans un texte"""
    return len(str(text).split())


@app.route('/signal_analysis')
def signal_analysis():
    return render_template('signal_analysis.html')

#analyse du signal audio

@app.route('/analyze_audio_signal', methods=['POST'])
def analyze_audio_signal():
    logger.info("Début de l'analyse de signal audio")
    
    if 'audio_file' not in request.files:
        logger.error("Aucun fichier audio trouvé")
        return jsonify({"error": "Aucun fichier audio trouvé"}), 400
    
    audio_file = request.files['audio_file']
    logger.info(f"Fichier reçu : {audio_file.filename}")
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
    audio_file.save(file_path)
    
    try:
        # Analyse du signal
        logger.info(f"Début du traitement du fichier : {file_path}")
        analysis_result = process_audio_signal(file_path)
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du signal : {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(file_path):
            os.remove(file_path)

# 📌 LANCEMENT DE L'APPLICATION
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)