# 🍅 TomatoGuard

Système de détection des maladies de la tomate par fusion YOLO + LSTM.

## Architecture

```
ESP32 (capteurs) ──→ Render.com (Flask + DB) ──→ App mobile web
                            ↑
                    YOLO (.pt) + LSTM (.h5) + Scaler (.pkl)
```

## Modes de fonctionnement

- **Mode normal** : 1 jour réel = 1 jour de données climatiques
- **Mode test** : 2 minutes = 1 jour (7 "jours" en 14 minutes)

Bascule via le bouton dans l'app mobile.

## Endpoints

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | App mobile web |
| `/api/status` | GET | Statut serveur + mode |
| `/api/mode` | POST | Changer mode (`{"mode": "test"}` ou `"normal"`) |
| `/api/climate` | POST | ESP32 envoie ses données |
| `/api/detect` | POST | Détection YOLO seule (image) |
| `/api/predict` | POST | Fusion YOLO + LSTM (image + 7 jours climat) |
| `/api/climate/history` | GET | Historique 7 jours |

## Déploiement sur Render.com

### Étape 1 — Préparer le repo GitHub

```bash
cd tomatoguard
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TON_USER/tomatoguard.git
git push -u origin main
```

### Étape 2 — Ajouter les modèles

Place tes fichiers dans `models/` :
```
models/
├── yolov8_tomato.pt
├── lstm_climate.h5
└── scaler.pkl
```

⚠️ Si tes modèles sont volumineux (>100 MB), utilise **Git LFS** :
```bash
git lfs install
git lfs track "*.pt" "*.h5" "*.pkl"
git add .gitattributes
git add models/
git commit -m "Add models"
git push
```

### Étape 3 — Déployer sur Render

1. Va sur [render.com](https://render.com)
2. Clique sur **New + → Web Service**
3. Connecte ton repo GitHub
4. Render détecte automatiquement `render.yaml` ✅
5. Clique sur **Create Web Service**

Ton serveur sera disponible à : `https://tomatoguard-XXXX.onrender.com`

### Étape 4 — Mettre à jour l'URL côté ESP32

Dans `esp32_main.cpp`, change :
```cpp
const char* SERVER_URL = "https://tomatoguard-XXXX.onrender.com/api/climate";
```

## Test local

```bash
pip install -r requirements.txt
mkdir models
# Place .pt, .h5, .pkl dans models/
python app.py
```

Ouvre `http://localhost:5000` sur ton téléphone (même WiFi que le PC).

## Structure projet

```
tomatoguard/
├── app.py              # Serveur Flask
├── requirements.txt    # Dépendances Python
├── render.yaml         # Config Render
├── .gitignore
├── README.md
├── static/
│   └── index.html      # App mobile
└── models/             # Modèles ML (à ajouter)
    ├── *.pt
    ├── *.h5
    └── *.pkl
```
