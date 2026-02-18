# EcoVision — Frontend (Streamlit)

Modern web interface for the EcoVision garbage classification system.

## Features

- 🖼️ **Image upload** with preview
- 🔬 **One-click classification** with loading spinner
- 📊 **Interactive bar chart** of class probabilities (Plotly)
- ♻️ **Recyclability suggestions** per class
- 🔥 **Grad-CAM visualization** toggle
- ⚠️ **Low-confidence warnings**
- 🌑 **Dark gradient UI** with glassmorphism

## Quick Start

### 1. Install dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Set backend URL (optional)

```bash
export BACKEND_URL=http://localhost:8000
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** The FastAPI backend must be running for predictions to work.

## Project Structure

```
frontend/
├── app.py              # Streamlit application
├── requirements.txt
├── Dockerfile
└── README.md
```
