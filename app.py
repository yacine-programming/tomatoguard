"""
TomatoGuard - Serveur Flask (version robuste)
==============================================
✅ Mode test : 20 secondes = 1 jour (7 jours en 2 min 20 sec)
✅ Mode normal : 1 jour réel = 1 jour
✅ Le compteur de périodes ne diminue JAMAIS (sauf reset manuel)
✅ Fenêtre glissante : utilise toujours les 7 dernières périodes
✅ Conserve les 50 dernières périodes (auto-nettoyage des plus vieilles)
✅ Heure Algérie (GMT+1) sur tous les timestamps
✅ Bouton de reset manuel disponible

LSTM utilise 5 variables dans cet ordre EXACT :
  [temperature, humidity, precipitation, wind_speed, leaf_wetness]
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import numpy as np
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
import joblib

app = Flask(__name__, static_folder='static')
CORS(app)

# ==================== ÉTAT GLOBAL ====================

APP_MODE = os.environ.get('APP_MODE', 'test')
REQUIRED_PERIODS = 7
MAX_PERIODS = 50  # On garde les 50 dernières périodes

# Mode test : 20 secondes = 1 jour | Mode normal : 1 jour = 1 jour
TEST_PERIOD_SECONDS = 20
NORMAL_PERIOD_SECONDS = 86400  # 24h

# Algérie = GMT+1 (CET, pas de daylight saving)
ALGERIA_TZ = timezone(timedelta(hours=1))

yolo_model = None
lstm_model = None
scaler = None

DISEASE_CLASSES = {
    0: 'Tomato_Bacterial_Spot',
    1: 'Tomato_Early_Blight',
    2: 'Tomato_Late_Blight',
    3: 'Tomato_Leaf_Mold',
    4: 'Tomato_Septoria_Leaf_Spot',
    5: 'Tomato_Spider_Mites',
    6: 'Tomato_Target_Spot',
    7: 'Tomato_Yellow_Leaf_Curl_Virus',
    8: 'Tomato_Healthy',
    9: 'Tomato_Mosaic_Virus'
}

LSTM_SUPPORTED = ['Tomato_Early_Blight', 'Tomato_Spider_Mites']

TREATMENT_RECOMMENDATIONS = {
    'Tomato_Bacterial_Spot': {'fr': "Appliquer un fongicide à base de cuivre. Éviter l'arrosage par aspersion. Retirer les feuilles infectées.", 'severity': 'modérée'},
    'Tomato_Early_Blight': {'fr': "Traiter avec un fongicide à base de chlorothalonil ou mancozèbe. Améliorer la circulation d'air entre les plants.", 'severity': 'élevée'},
    'Tomato_Late_Blight': {'fr': "Action urgente requise. Appliquer un fongicide systémique (métalaxyl). Détruire les plants gravement atteints.", 'severity': 'critique'},
    'Tomato_Leaf_Mold': {'fr': "Réduire l'humidité de la serre. Améliorer la ventilation. Appliquer un fongicide à base de cuivre.", 'severity': 'modérée'},
    'Tomato_Septoria_Leaf_Spot': {'fr': "Retirer les feuilles infectées. Traiter avec un fongicide (chlorothalonil). Éviter l'arrosage du feuillage.", 'severity': 'modérée'},
    'Tomato_Spider_Mites': {'fr': "Pulvériser de l'eau savonneuse ou un acaricide. Augmenter l'humidité (les acariens préfèrent le sec).", 'severity': 'élevée'},
    'Tomato_Target_Spot': {'fr': "Appliquer un fongicide (azoxystrobine). Espacer les plants. Pailler le sol pour éviter les éclaboussures.", 'severity': 'modérée'},
    'Tomato_Yellow_Leaf_Curl_Virus': {'fr': "Aucun traitement curatif. Détruire les plants infectés. Lutter contre les aleurodes (vecteur du virus).", 'severity': 'critique'},
    'Tomato_Mosaic_Virus': {'fr': "Aucun traitement curatif. Arracher et brûler les plants infectés. Désinfecter les outils.", 'severity': 'critique'},
    'Tomato_Healthy': {'fr': "Plante en bonne santé. Continuer les bonnes pratiques de surveillance.", 'severity': 'aucune'}
}

# ==================== UTILS DATETIME ====================

def now_algeria():
    """Datetime actuelle en Algérie (GMT+1)"""
    return datetime.now(ALGERIA_TZ)

def now_algeria_str():
    """Format ISO pour stockage en DB"""
    return now_algeria().strftime('%Y-%m-%d %H:%M:%S')

def get_period_index_now():
    """
    Retourne un index de période entier basé sur l'heure actuelle.
    En mode test : un nouvel index toutes les 20 secondes.
    En mode normal : un nouvel index par jour.
    """
    period_seconds = TEST_PERIOD_SECONDS if APP_MODE == 'test' else NORMAL_PERIOD_SECONDS
    epoch_seconds = int(now_algeria().timestamp())
    return epoch_seconds // period_seconds

# ==================== CHARGEMENT MODÈLES ====================

def load_models():
    global yolo_model, lstm_model, scaler
    import traceback
    models_dir = os.environ.get('MODELS_DIR', './models')
    print(f"📂 Recherche des modèles dans : {os.path.abspath(models_dir)}")
    if not os.path.exists(models_dir):
        print(f"❌ Le dossier {models_dir} n'existe pas !")
        return
    files = os.listdir(models_dir)
    print(f"📋 Fichiers trouvés : {files}")
    try:
        from ultralytics import YOLO
        for f in files:
            if f.endswith('.pt'):
                yolo_model = YOLO(os.path.join(models_dir, f))
                print(f"✅ YOLO chargé: {f}")
                break
    except Exception as e:
        print(f"❌ Erreur YOLO: {e}"); traceback.print_exc()
    try:
        from tensorflow.keras.models import load_model
        for f in files:
            if f.endswith('.h5'):
                lstm_model = load_model(os.path.join(models_dir, f))
                print(f"✅ LSTM chargé: {f}")
                break
    except Exception as e:
        print(f"❌ Erreur LSTM: {e}"); traceback.print_exc()
    try:
        for f in files:
            if f.endswith('.pkl'):
                scaler = joblib.load(os.path.join(models_dir, f))
                print(f"✅ Scaler chargé: {f}")
                break
    except Exception as e:
        print(f"❌ Erreur Scaler: {e}"); traceback.print_exc()
    print(f"📊 État final : YOLO={yolo_model is not None} LSTM={lstm_model is not None} Scaler={scaler is not None}")

# ==================== BASE DE DONNÉES ====================

def init_db():
    """
    2 tables :
    - raw_readings : chaque envoi ESP32 (avec son period_index calculé au moment de la réception)
    - On agrège par period_index quand on a besoin
    """
    conn = sqlite3.connect('climate.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS raw_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_index INTEGER NOT NULL,
            timestamp_algeria TEXT NOT NULL,
            mode TEXT NOT NULL,
            temperature REAL, humidity REAL, precipitation REAL,
            wind_speed REAL, wind_direction INTEGER, wind_direction_label TEXT,
            leaf_wetness REAL, pressure_hpa REAL,
            soil_cb INTEGER, soil_resistance REAL,
            npk_n INTEGER, npk_p INTEGER, npk_k INTEGER
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_period ON raw_readings(period_index)')
    conn.commit()
    conn.close()
    print("✅ Base de données prête")

def cleanup_old_periods():
    """Garde uniquement les MAX_PERIODS dernières périodes uniques"""
    conn = sqlite3.connect('climate.db')
    c = conn.cursor()
    c.execute('SELECT DISTINCT period_index FROM raw_readings ORDER BY period_index DESC LIMIT ?', (MAX_PERIODS,))
    keep = [r[0] for r in c.fetchall()]
    if keep:
        c.execute(f'DELETE FROM raw_readings WHERE period_index NOT IN ({",".join("?"*len(keep))})', keep)
        deleted = c.rowcount
        if deleted > 0:
            print(f"🧹 Nettoyage : {deleted} ancien(nes) lignes supprimées")
    conn.commit()
    conn.close()

def get_aggregated_periods(limit=None):
    """
    Retourne les périodes agrégées (moyenne par period_index).
    Triées du plus récent au plus ancien.
    JAMAIS de filtre temporel - on compte les périodes UNIQUES en DB.
    """
    conn = sqlite3.connect('climate.db')
    c = conn.cursor()

    query = '''
        SELECT
            period_index,
            MIN(timestamp_algeria) as first_ts,
            MAX(timestamp_algeria) as last_ts,
            AVG(temperature), AVG(humidity), AVG(precipitation),
            AVG(wind_speed), AVG(wind_direction), AVG(leaf_wetness),
            AVG(pressure_hpa), AVG(soil_cb), AVG(soil_resistance),
            AVG(npk_n), AVG(npk_p), AVG(npk_k),
            COUNT(*) as samples
        FROM raw_readings
        GROUP BY period_index
        ORDER BY period_index DESC
    '''
    if limit:
        query += f' LIMIT {int(limit)}'

    c.execute(query)
    rows = c.fetchall()
    conn.close()
    return rows

def get_last_reading():
    """Dernière mesure brute reçue"""
    conn = sqlite3.connect('climate.db')
    c = conn.cursor()
    c.execute('SELECT * FROM raw_readings ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    cols = ['id', 'period_index', 'timestamp_algeria', 'mode',
            'temperature', 'humidity', 'precipitation', 'wind_speed',
            'wind_direction', 'wind_direction_label', 'leaf_wetness',
            'pressure_hpa', 'soil_cb', 'soil_resistance',
            'npk_n', 'npk_p', 'npk_k']
    return dict(zip(cols, row))

# ==================== ROUTES ====================

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    periods = get_aggregated_periods(REQUIRED_PERIODS)
    return jsonify({
        'status': 'online',
        'mode': APP_MODE,
        'mode_description': '20 sec = 1 jour' if APP_MODE == 'test' else '1 jour réel = 1 jour',
        'period_seconds': TEST_PERIOD_SECONDS if APP_MODE == 'test' else NORMAL_PERIOD_SECONDS,
        'models_loaded': {
            'yolo': yolo_model is not None,
            'lstm': lstm_model is not None,
            'scaler': scaler is not None
        },
        'periods_available': len(periods),
        'periods_required': REQUIRED_PERIODS,
        'max_periods_kept': MAX_PERIODS,
        'ready_for_fusion': len(periods) >= REQUIRED_PERIODS,
        'server_time_algeria': now_algeria_str()
    }), 200

@app.route('/api/mode', methods=['POST'])
def switch_mode():
    global APP_MODE
    data = request.get_json()
    new_mode = data.get('mode', '').lower()
    if new_mode not in ['normal', 'test']:
        return jsonify({'error': "mode doit être 'normal' ou 'test'"}), 400
    APP_MODE = new_mode
    print(f"🔄 Mode changé: {APP_MODE}")
    return jsonify({'status': 'ok', 'mode': APP_MODE}), 200

@app.route('/api/climate', methods=['POST'])
def receive_climate():
    """Reçoit les données capteurs de l'ESP32"""
    try:
        data = request.get_json()
        required = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'leaf_wetness']
        for f in required:
            if f not in data:
                return jsonify({'error': f'Champ manquant: {f}'}), 400

        period_idx = get_period_index_now()
        ts = now_algeria_str()

        conn = sqlite3.connect('climate.db')
        conn.execute('''
            INSERT INTO raw_readings (period_index, timestamp_algeria, mode,
                temperature, humidity, precipitation, wind_speed, wind_direction,
                wind_direction_label, leaf_wetness, pressure_hpa,
                soil_cb, soil_resistance, npk_n, npk_p, npk_k)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            period_idx, ts, APP_MODE,
            data['temperature'], data['humidity'], data['precipitation'],
            data['wind_speed'], data.get('wind_direction'), data.get('wind_direction_label'),
            data['leaf_wetness'], data.get('pressure_hpa'),
            data.get('soil_cb'), data.get('soil_resistance'),
            data.get('npk_n'), data.get('npk_p'), data.get('npk_k')
        ))
        conn.commit()
        conn.close()

        # Nettoyage : ne garder que MAX_PERIODS
        cleanup_old_periods()

        print(f"📡 ESP32 [{APP_MODE}] période #{period_idx} → T={data['temperature']}°C H={data['humidity']}%")
        return jsonify({
            'status': 'ok',
            'period_index': period_idx,
            'timestamp': ts,
            'mode': APP_MODE
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    """Toutes les périodes pour affichage (jusqu'à MAX_PERIODS)"""
    try:
        rows = get_aggregated_periods(MAX_PERIODS)
        keys = ['period_index', 'first_ts', 'last_ts',
                'temperature', 'humidity', 'precipitation',
                'wind_speed', 'wind_direction', 'leaf_wetness',
                'pressure_hpa', 'soil_cb', 'soil_resistance',
                'npk_n', 'npk_p', 'npk_k', 'samples']

        history = []
        for r in rows:
            entry = {}
            for k, v in zip(keys, r):
                if v is None:
                    entry[k] = None
                elif isinstance(v, float):
                    entry[k] = round(v, 2)
                else:
                    entry[k] = v
            history.append(entry)

        return jsonify({
            'periods': history,
            'count': len(history),
            'required': REQUIRED_PERIODS,
            'max_kept': MAX_PERIODS,
            'mode': APP_MODE,
            'last_reading': get_last_reading(),
            'server_time': now_algeria_str()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Efface toutes les données capteurs (avec confirmation côté client)"""
    try:
        data = request.get_json() or {}
        if data.get('confirm') != 'YES_DELETE_ALL':
            return jsonify({'error': 'Confirmation requise'}), 400

        conn = sqlite3.connect('climate.db')
        c = conn.cursor()
        c.execute('DELETE FROM raw_readings')
        deleted = c.rowcount
        conn.commit()
        conn.close()

        print(f"🗑️  Reset effectué : {deleted} lignes supprimées")
        return jsonify({'status': 'ok', 'deleted': deleted}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== DÉTECTION SEULE ====================

@app.route('/api/detect', methods=['POST'])
def detect_only():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image reçue'}), 400
        if yolo_model is None:
            return jsonify({'error': 'Modèle YOLO non chargé'}), 500

        image_file = request.files['image']
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        results = yolo_model(tmp_path, verbose=False)
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    conf = float(box.conf[0])
                    disease = DISEASE_CLASSES.get(cid, 'Unknown')
                    rec = TREATMENT_RECOMMENDATIONS.get(disease, {})
                    detections.append({
                        'disease': disease,
                        'disease_label': disease.replace('Tomato_', '').replace('_', ' '),
                        'confidence': round(conf * 100, 1),
                        'is_diseased': disease != 'Tomato_Healthy',
                        'severity': rec.get('severity', 'inconnue'),
                        'treatment': rec.get('fr', 'Aucune recommandation disponible')
                    })

        os.unlink(tmp_path)

        if not detections:
            return jsonify({'status': 'no_detection',
                          'message': 'Aucune détection sur cette photo'}), 200
        return jsonify({'status': 'success', 'detections': detections}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== FUSION (STRICT 7 PÉRIODES) ====================

@app.route('/api/predict', methods=['POST'])
def predict_fusion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image reçue'}), 400
        if not all([yolo_model, lstm_model, scaler]):
            return jsonify({'error': 'Tous les modèles ne sont pas chargés'}), 500

        # Prendre les 7 dernières périodes UNIQUES (pas filtre temporel)
        rows = get_aggregated_periods(REQUIRED_PERIODS)

        if len(rows) < REQUIRED_PERIODS:
            return jsonify({
                'error': 'Pas assez de données climatiques',
                'detail': f'{REQUIRED_PERIODS} périodes requises. Actuellement : {len(rows)}/{REQUIRED_PERIODS}.',
                'periods_available': len(rows),
                'periods_required': REQUIRED_PERIODS,
                'mode': APP_MODE
            }), 400

        # rows[i] = (period_index, first_ts, last_ts, temp, humidity, precip, wind_speed,
        #            wind_dir, leaf_wetness, pressure, soil_cb, ...)
        # Index pour LSTM : 3=temp, 4=humidity, 5=precip, 6=wind_speed, 8=leaf_wetness
        climate_data = []
        for r in rows:
            climate_data.append([
                r[3] or 0,  # temperature
                r[4] or 0,  # humidity
                r[5] or 0,  # precipitation
                r[6] or 0,  # wind_speed
                r[8] or 0   # leaf_wetness (index 8 car 7 = wind_direction)
            ])

        # Du plus ancien au plus récent (le LSTM doit voir l'ordre chronologique)
        climate_data.reverse()

        image_file = request.files['image']
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        result = run_fusion(tmp_path, climate_data)
        os.unlink(tmp_path)

        if result is None:
            return jsonify({'status': 'no_detection'}), 200

        result['periods_used'] = len(rows)
        result['mode'] = APP_MODE
        return jsonify({'status': 'success', 'results': result}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def run_fusion(image_path, climate_data_7):
    yolo_results = yolo_model(image_path, verbose=False)
    detections = []
    for r in yolo_results:
        if r.boxes:
            for box in r.boxes:
                d = DISEASE_CLASSES.get(int(box.cls[0]), 'Unknown')
                detections.append({
                    'disease': d,
                    'confidence': float(box.conf[0]),
                    'is_diseased': d != 'Tomato_Healthy'
                })
    if not detections:
        return None

    arr = np.array(climate_data_7)
    arr_n = scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(1, arr.shape[0], arr.shape[-1])
    pred = lstm_model.predict(arr_n, verbose=0)
    eb_risk = float(pred[0][0])
    sm_risk = float(pred[0][1])

    risks = {
        'Tomato_Early_Blight': eb_risk > 0.5,
        'Tomato_Spider_Mites': sm_risk > 0.5
    }

    results_list = []
    yolo_healthy = any(d['disease'] == 'Tomato_Healthy' for d in detections)

    if yolo_healthy:
        alerts = [name for name, key in
                  [('Early Blight', 'Tomato_Early_Blight'),
                   ('Spider Mites', 'Tomato_Spider_Mites')]
                  if risks[key]]
        if alerts:
            results_list.append({
                'type': 'potential_risk', 'disease': ' & '.join(alerts),
                'message': f"Feuille saine mais climat favorable à : {', '.join(alerts)}",
                'action': 'TRAITEMENT PRÉVENTIF RECOMMANDÉ', 'confidence': None
            })
        else:
            results_list.append({
                'type': 'safe', 'disease': 'Aucune',
                'message': 'Feuille saine et climat sans risque',
                'action': 'Aucune action nécessaire', 'confidence': None
            })
    else:
        for det in detections:
            if not det['is_diseased']: continue
            d, conf = det['disease'], det['confidence']
            label = d.replace('Tomato_', '').replace('_', ' ')
            rec = TREATMENT_RECOMMENDATIONS.get(d, {})
            if d in LSTM_SUPPORTED:
                if risks.get(d, False):
                    results_list.append({
                        'type': 'critical', 'disease': label,
                        'message': f'{label} détectée ET climat favorable au développement',
                        'action': 'TRAITEMENT IMMÉDIAT NÉCESSAIRE',
                        'treatment': rec.get('fr'),
                        'confidence': round(conf * 100, 1)
                    })
                else:
                    results_list.append({
                        'type': 'moderate', 'disease': label,
                        'message': f'{label} détectée mais climat défavorable',
                        'action': 'SURVEILLANCE RECOMMANDÉE',
                        'treatment': rec.get('fr'),
                        'confidence': round(conf * 100, 1)
                    })
            else:
                results_list.append({
                    'type': 'standard', 'disease': label,
                    'message': f'{label} détectée (pas de modèle climatique)',
                    'action': 'TRAITEMENT STANDARD',
                    'treatment': rec.get('fr'),
                    'confidence': round(conf * 100, 1)
                })

    return {
        'detections': results_list,
        'climate_summary': {
            'early_blight_risk': round(eb_risk * 100, 1),
            'spider_mites_risk': round(sm_risk * 100, 1)
        }
    }

# ==================== LANCEMENT ====================

def background_init():
    print("\n🍅 TomatoGuard — Initialisation en arrière-plan\n")
    init_db()
    load_models()
    print(f"\n✅ Initialisation terminée — Heure Algérie : {now_algeria_str()}\n")

threading.Thread(target=background_init, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
