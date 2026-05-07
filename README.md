---
title: TomatoGuard
emoji: 🍅
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🍅 TomatoGuard

Système de détection des maladies de la tomate par fusion YOLO + LSTM.

## Architecture

ESP32 (capteurs) → Hugging Face Spaces (Flask + ML) → App mobile web

## Endpoints

- `GET /` — App mobile
- `GET /api/status` — Statut serveur
- `POST /api/climate` — ESP32 envoie ses données
- `POST /api/detect` — Détection YOLO seule
- `POST /api/predict` — Fusion YOLO + LSTM