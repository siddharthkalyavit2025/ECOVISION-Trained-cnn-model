# EcoVision — Backend (FastAPI)

Production-ready REST API for garbage image classification using a pre-trained EfficientNet model.

## Features

- 🔬 **Image classification** into 10 waste categories
- 📊 **Full probability distribution** for all classes
- ♻️ **Recyclability suggestions** for each class
- 🔥 **Grad-CAM visualization** (optional endpoint)
- 📝 **CSV prediction logging**
- ⚡ **Confidence threshold** alerts
- 🐳 **Docker-ready**

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set environment variables (optional)

```bash
# Default: looks for model in parent directory
export MODEL_PATH=../garbage_efficientnet_model.h5
export LOG_LEVEL=INFO
export CONFIDENCE_THRESHOLD=0.5
```

### 3. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Endpoints

| Method | Endpoint            | Description                        |
|--------|---------------------|------------------------------------|
| GET    | `/`                 | Health check                       |
| POST   | `/predict`          | Classify uploaded image            |
| POST   | `/predict/gradcam`  | Classify + Grad-CAM heatmap        |

### Example: `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

**Response:**

```json
{
  "predicted_class": "plastic",
  "confidence": 0.92,
  "all_probabilities": {
    "batteries": 0.01,
    "biological": 0.02,
    "cardboard": 0.01,
    "clothes": 0.0,
    "glass": 0.01,
    "metal": 0.01,
    "paper": 0.01,
    "plastic": 0.92,
    "shoes": 0.0,
    "trash": 0.01
  },
  "recyclability_suggestion": "♻️ Recyclable — Check the resin code. Rinse and place in plastics recycling.",
  "below_confidence_threshold": false
}
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app + endpoints
│   ├── model_loader.py   # Singleton model loading
│   ├── predict.py        # Prediction orchestration
│   ├── schemas.py        # Pydantic models
│   ├── config.py         # Settings & env vars
│   └── utils.py          # Preprocessing, Grad-CAM, logging
├── requirements.txt
├── Dockerfile
└── README.md
```
