# ♻️ EcoVision — AI-Powered Garbage Classification

A full-stack machine learning application that classifies waste images into **10 categories** using a pre-trained EfficientNet model, with recyclability suggestions and Grad-CAM visualizations.

---

## 🏗️ Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Streamlit UI      │  HTTP   │   FastAPI Backend    │
│   (port 8501)       │───────▶│   (port 8000)        │
│                     │◀───────│                       │
│  • Image upload     │  JSON   │  • Model inference   │
│  • Result display   │         │  • Grad-CAM          │
│  • Plotly charts    │         │  • CSV logging        │
└─────────────────────┘         └─────────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  EfficientNet   │
                                │  (.h5 model)    │
                                └─────────────────┘
```

## 📁 Project Structure

```
ecovision/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app + endpoints
│   │   ├── model_loader.py    # Singleton model loading
│   │   ├── predict.py         # Prediction orchestration
│   │   ├── schemas.py         # Pydantic models
│   │   ├── config.py          # Settings & environment variables
│   │   └── utils.py           # Preprocessing, Grad-CAM, logging
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── frontend/
│   ├── app.py                 # Streamlit application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── garbage_efficientnet_model.h5   # Trained model
├── docker-compose.yml
├── .env.example
└── README.md                       # ← You are here
```

## 🏷️ Classes

| # | Class       | Recyclability |
|---|-------------|---------------|
| 0 | Batteries   | ♻️ Recyclable (special drop-off) |
| 1 | Biological  | 🌿 Compostable |
| 2 | Cardboard   | ♻️ Recyclable |
| 3 | Clothes     | 👗 Donatable |
| 4 | Glass       | ♻️ Recyclable |
| 5 | Metal       | ♻️ Recyclable |
| 6 | Paper       | ♻️ Recyclable |
| 7 | Plastic     | ♻️ Recyclable |
| 8 | Shoes       | 👟 Donatable |
| 9 | Trash       | 🗑️ Non-recyclable |

---

## 🚀 Quick Start

### Option 1 — Local (recommended for development)

**1. Install backend dependencies**

```bash
cd backend
pip install -r requirements.txt
```

**2. Start the backend**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Install frontend dependencies** (new terminal)

```bash
cd frontend
pip install -r requirements.txt
```

**4. Start the frontend**

```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

### Option 2 — Docker

```bash
# Build and start both services
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

> The model file `garbage_efficientnet_model.h5` must be in the project root.

---

## 📡 API Reference

### `GET /` — Health Check

```bash
curl http://localhost:8000/
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "garbage_efficientnet_model.h5"
}
```

### `POST /predict` — Classify Image

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

```json
{
  "predicted_class": "plastic",
  "confidence": 0.9213,
  "all_probabilities": {
    "batteries": 0.0051,
    "biological": 0.0182,
    "cardboard": 0.0073,
    "clothes": 0.0024,
    "glass": 0.0091,
    "metal": 0.0134,
    "paper": 0.0112,
    "plastic": 0.9213,
    "shoes": 0.0038,
    "trash": 0.0082
  },
  "recyclability_suggestion": "♻️ Recyclable — Check the resin code.",
  "below_confidence_threshold": false
}
```

### `POST /predict/gradcam` — Classify + Grad-CAM

```bash
curl -X POST http://localhost:8000/predict/gradcam \
  -F "file=@test_image.jpg"
```

Returns the same JSON as `/predict` plus a `gradcam_image` field (base64 PNG).

### Swagger UI

Interactive API docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## ⚙️ Environment Variables

| Variable                | Default                               | Description              |
|-------------------------|---------------------------------------|--------------------------|
| `MODEL_PATH`            | `../garbage_efficientnet_model.h5`    | Path to the model file   |
| `LOG_LEVEL`             | `INFO`                                | Logging level            |
| `CONFIDENCE_THRESHOLD`  | `0.5`                                 | Low-confidence warning   |
| `ALLOWED_ORIGINS`       | `["*"]`                               | CORS allowed origins     |
| `BACKEND_URL`           | `http://localhost:8000`               | Backend URL (frontend)   |

Copy `.env.example` to `.env` and customise as needed.

---

## 🚢 Deployment

### Render / Railway (Backend)

1. Push the `backend/` directory to a Git repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Upload `garbage_efficientnet_model.h5` as a persistent disk or use cloud storage

### Streamlit Cloud (Frontend)

1. Push the `frontend/` directory to a Git repo
2. Set `BACKEND_URL` in Streamlit Cloud secrets
3. Deploy — Streamlit Cloud auto-detects `app.py`

### Docker (any cloud)

```bash
docker-compose up --build -d
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model     | TensorFlow / Keras (EfficientNet) |
| Backend   | FastAPI + Uvicorn |
| Frontend  | Streamlit + Plotly |
| Container | Docker + Docker Compose |

---

## 📜 License

This project is for educational and research purposes.
