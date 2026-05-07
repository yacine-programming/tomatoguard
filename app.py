"""
TomatoGuard - Serveur Flask
============================
Endpoints:
  - POST /api/climate          → ESP32 envoie ses données
  - POST /api/detect           → Détection YOLO seule
  - POST /api/predict          → Fusion YOLO + LSTM
  - GET  /api/status           → Statut + mode actuel
  - POST /api/mode             → Basculer entre normal et test
  - GET  /api/climate/history  → Historique 7 jours

Mode TEST: 2 minutes simulent 1 jour (7 jours = 14 min)
Mode NORMAL: 1 jour réel = 1 jour
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
import joblib

app = Flask(__name__, static_folder='static')
CORS(app)

# ==================== ÉTAT GLOBAL ====================

# Mode actuel: 'normal' ou 'test'
# - normal : 1 jour réel = 1 jour climat (groupement par DATE())
# - test   : 2 minutes = 1 jour (groupement par fenêtres de 2 min)
APP_MODE = os.environ.get('APP_MODE', 'test')

# Modèles chargés en mémoire
yolo_model = None
lstm_model = None
scaler = None

# Liste des classes YOLO
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

# Recommandations de traitement standard par maladie
TREATMENT_RECOMMENDATIONS = {
    'Tomato_Bacterial_Spot': {
        'fr': "Appliquer un fongicide à base de cuivre. Éviter l'arrosage par aspersion. Retirer les feuilles infectées.",
        'severity': 'modérée'
    },
    'Tomato_Early_Blight': {
        'fr': "Traiter avec un fongicide à base de chlorothalonil ou mancozèbe. Améliorer la circulation d'air entre les plants.",
        'severity': 'élevée'
    },
    'Tomato_Late_Blight': {
        'fr': "Action urgente requise. Appliquer un fongicide systémique (métalaxyl). Détruire les plants gravement atteints.",
        'severity': 'critique'
    },
    'Tomato_Leaf_Mold': {
        'fr': "Réduire l'humidité de la serre. Améliorer la ventilation. Appliquer un fongicide à base de cuivre.",
        'severity': 'modérée'
    },
    'Tomato_Septoria_Leaf_Spot': {
        'fr': "Retirer les feuilles infectées. Traiter avec un fongicide (chlorothalonil). Éviter l'arrosage du feuillage.",
        'severity': 'modérée'
    },
    'Tomato_Spider_Mites': {
        'fr': "Pulvériser de l'eau savonneuse ou un acaricide. Augmenter l'humidité (les acariens préfèrent le sec).",
        'severity': 'élevée'
    },
    'Tomato_Target_Spot': {
        'fr': "Appliquer un fongicide (azoxystrobine). Espacer les plants. Pailler le sol pour éviter les éclaboussures.",
        'severity': 'modérée'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'fr': "Aucun traitement curatif. Détruire les plants infectés. Lutter contre les aleurodes (vecteur du virus).",
        'severity': 'critique'
    },
    'Tomato_Mosaic_Virus': {
        'fr': "Aucun traitement curatif. Arracher et brûler les plants infectés. Désinfecter les outils.",
        'severity': 'critique'
    },
    'Tomato_Healthy': {
        'fr': "Plante en bonne santé. Continuer les bonnes pratiques de surveillance.",
        'severity': 'aucune'
    }
}

# ==================== CHARGEMENT DES MODÈLES ====================

def load_models():
    """Charge YOLO, LSTM et scaler une seule fois au démarrage"""
    global yolo_model, lstm_model, scaler

    try:
        from ultralytics import YOLO
        from tensorflow.keras.models import load_model

        models_dir = os.environ.get('MODELS_DIR', './models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for f in os.listdir(models_dir):
            path = os.path.join(models_dir, f)
            if f.endswith('.pt'):
                yolo_model = YOLO(path)
                print(f"✅ YOLO chargé: {f}")
            elif f.endswith('.h5'):
                lstm_model = load_model(path)
                print(f"✅ LSTM chargé: {f}")
            elif f.endswith('.pkl'):
                scaler = joblib.load(path)
                print(f"✅ Scaler chargé: {f}")

        if not all([yolo_model, lstm_model, scaler]):
            print("⚠️  Certains modèles sont manquants dans ./models/")
    except Exception as e:
        print(f"❌ Erreur chargement modèles: {e}")

# ==================== BASE DE DONNÉES ====================

def init_db():
    conn = sqlite3.connect('climate.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS climate (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL, humidity REAL, precipitation REAL,
            wind_speed REAL, wind_direction INTEGER, leaf_wetness REAL,
            pressure_hpa REAL, soil_cb INTEGER, soil_resistance REAL,
            npk_n INTEGER, npk_p INTEGER, npk_k INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Base de données prête")

def get_last_7_days():
    """
    Récupère 7 'jours' de données selon le mode actuel.
    - Mode normal : 7 derniers jours réels, groupés par date
    - Mode test   : 14 dernières minutes, groupées par fenêtres de 2 min
    """
    conn = sqlite3.connect('climate.db')
    c = conn.cursor()

    if APP_MODE == 'test':
        # 14 minutes = 7 "jours" de 2 min
        since = (datetime.now() - timedelta(minutes=14)).strftime('%Y-%m-%d %H:%M:%S')
        # Grouper par fenêtre de 2 minutes (timestamp tronqué)
        c.execute('''
            SELECT
                strftime('%Y-%m-%d %H:', timestamp) ||
                  printf('%02d', (CAST(strftime('%M', timestamp) AS INTEGER) / 2) * 2) AS window,
                AVG(temperature), AVG(humidity), AVG(precipitation),
                AVG(wind_speed),  AVG(leaf_wetness)
            FROM climate
            WHERE timestamp >= ?
            GROUP BY window
            ORDER BY window DESC
            LIMIT 7
        ''', (since,))
    else:
        # Mode normal : 7 derniers jours
        since = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''
            SELECT DATE(timestamp) AS day,
                AVG(temperature), AVG(humidity), AVG(precipitation),
                AVG(wind_speed),  AVG(leaf_wetness)
            FROM climate
            WHERE timestamp >= ?
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
        ''', (since,))

    rows = c.fetchall()
    conn.close()
    return rows

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Sert la page web mobile"""
    return send_from_directory('static', 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    rows = get_last_7_days()
    return jsonify({
        'status': 'online',
        'mode': APP_MODE,
        'mode_description': '2 min = 1 jour' if APP_MODE == 'test' else '1 jour réel = 1 jour',
        'models_loaded': {
            'yolo':   yolo_model is not None,
            'lstm':   lstm_model is not None,
            'scaler': scaler is not None
        },
        'climate_days_available': len(rows),
        'minimum_days_required': 3
    }), 200

@app.route('/api/mode', methods=['POST'])
def switch_mode():
    """Bascule entre 'normal' et 'test'"""
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

        conn = sqlite3.connect('climate.db')
        conn.execute('''
            INSERT INTO climate (timestamp, temperature, humidity, precipitation,
                wind_speed, wind_direction, leaf_wetness, pressure_hpa,
                soil_cb, soil_resistance, npk_n, npk_p, npk_k)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data['temperature'], data['humidity'], data['precipitation'],
            data['wind_speed'],  data.get('wind_direction'),
            data['leaf_wetness'],data.get('pressure_hpa'),
            data.get('soil_cb'), data.get('soil_resistance'),
            data.get('npk_n'),   data.get('npk_p'), data.get('npk_k')
        ))
        conn.commit()
        conn.close()

        print(f"📡 ESP32 → T={data['temperature']}°C "
              f"H={data['humidity']}% P={data['precipitation']}mm "
              f"[{APP_MODE}]")
        return jsonify({'status': 'ok', 'mode': APP_MODE}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== DÉTECTION SEULE (YOLO) ====================

@app.route('/api/detect', methods=['POST'])
def detect_only():
    """
    Détection YOLO seule, sans données climatiques.
    Retourne maladie + confiance + recommandation de traitement standard.
    """
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
                    cid  = int(box.cls[0])
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

# ==================== FUSION YOLO + LSTM ====================

@app.route('/api/predict', methods=['POST'])
def predict_fusion():
    """Prédiction complète : fusion YOLO (image) + LSTM (climat 7 jours)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image reçue'}), 400
        if not all([yolo_model, lstm_model, scaler]):
            return jsonify({'error': 'Tous les modèles ne sont pas chargés'}), 500

        rows = get_last_7_days()
        if len(rows) < 3:
            return jsonify({
                'error': 'Pas assez de données climatiques',
                'detail': f'Seulement {len(rows)} période(s) disponible(s). Minimum 3 requis.',
                'mode': APP_MODE
            }), 400

        climate_data = [[r[1], r[2], r[3], r[4], r[5]] for r in rows]
        while len(climate_data) < 7:
            climate_data.insert(0, climate_data[0])
        climate_data = climate_data[:7]

        image_file = request.files['image']
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        result = run_fusion(tmp_path, climate_data)
        os.unlink(tmp_path)

        if result is None:
            return jsonify({'status': 'no_detection'}), 200

        result['climate_periods'] = len(rows)
        result['mode'] = APP_MODE
        return jsonify({'status': 'success', 'results': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_fusion(image_path, climate_data_7days):
    """Logique de fusion YOLO + LSTM"""
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

    arr = np.array(climate_data_7days)
    arr_n = scaler.transform(arr.reshape(-1, arr.shape[-1])) \
                  .reshape(1, arr.shape[0], arr.shape[-1])
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
                'type': 'potential_risk',
                'disease': ' & '.join(alerts),
                'message': f"Feuille saine mais climat favorable à : {', '.join(alerts)}",
                'action': 'TRAITEMENT PRÉVENTIF RECOMMANDÉ',
                'confidence': None
            })
        else:
            results_list.append({
                'type': 'safe',
                'disease': 'Aucune',
                'message': 'Feuille saine et climat sans risque',
                'action': 'Aucune action nécessaire',
                'confidence': None
            })
    else:
        for det in detections:
            if not det['is_diseased']:
                continue
            d, conf = det['disease'], det['confidence']
            label = d.replace('Tomato_', '').replace('_', ' ')
            rec = TREATMENT_RECOMMENDATIONS.get(d, {})

            if d in LSTM_SUPPORTED:
                if risks.get(d, False):
                    results_list.append({
                        'type': 'critical',
                        'disease': label,
                        'message': f'{label} détectée ET climat favorable au développement',
                        'action': 'TRAITEMENT IMMÉDIAT NÉCESSAIRE',
                        'treatment': rec.get('fr'),
                        'confidence': round(conf * 100, 1)
                    })
                else:
                    results_list.append({
                        'type': 'moderate',
                        'disease': label,
                        'message': f'{label} détectée mais climat défavorable',
                        'action': 'SURVEILLANCE RECOMMANDÉE',
                        'treatment': rec.get('fr'),
                        'confidence': round(conf * 100, 1)
                    })
            else:
                results_list.append({
                    'type': 'standard',
                    'disease': label,
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

@app.route('/api/climate/history', methods=['GET'])
def climate_history():
    rows = get_last_7_days()
    keys = ['period', 'temperature', 'humidity', 'precipitation',
            'wind_speed', 'leaf_wetness']
    history = [
        {k: (round(v, 2) if isinstance(v, float) else v)
         for k, v in zip(keys, r)}
        for r in rows
    ]
    return jsonify({'days': history, 'count': len(history), 'mode': APP_MODE}), 200

# ==================== LANCEMENT ====================

if __name__ == '__main__':
    print("\n🍅 TomatoGuard — Démarrage du serveur\n")
    init_db()
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
