"""
EcoVision — Streamlit Frontend

A modern UI for uploading waste images and receiving AI-powered
classification results from the FastAPI backend.
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
GRADCAM_ENDPOINT = f"{BACKEND_URL}/predict/gradcam"

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoVision ♻️ Garbage Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* ── Header ────────────────────────── */
    .eco-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .eco-header h1 {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .eco-header p {
        color: #b0b0c0;
        font-size: 1.1rem;
    }

    /* ── Result card ───────────────────── */
    .result-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
    }
    .result-card h2 {
        color: #92fe9d;
        margin-top: 0;
    }

    /* ── Prediction badge ──────────────── */
    .pred-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00c9ff, #92fe9d);
        color: #0f0c29;
        font-size: 1.5rem;
        font-weight: 700;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        margin: 0.5rem 0;
    }

    /* ── Confidence meter ──────────────── */
    .confidence-meter {
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 12px;
        border-radius: 6px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    /* ── Suggestion box ────────────────── */
    .suggestion-box {
        background: rgba(146, 254, 157, 0.1);
        border-left: 4px solid #92fe9d;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #e0e0e0;
        font-size: 1rem;
    }

    /* ── Warning box ───────────────────── */
    .warning-box {
        background: rgba(255, 193, 7, 0.12);
        border-left: 4px solid #ffc107;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #fff3cd;
    }

    /* ── Footer ────────────────────────── */
    .eco-footer {
        text-align: center;
        color: #666;
        padding: 2rem 0 1rem;
        font-size: 0.85rem;
    }

    /* ── Upload area ───────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(146,254,157,0.3);
        border-radius: 16px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  UI Components
# ═══════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    """Render the page header."""
    st.markdown(
        """
        <div class="eco-header">
            <h1>♻️ EcoVision</h1>
            <p>AI-Powered Garbage Classification — Drop an image and let the model decide!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probabilities: dict[str, float]) -> None:
    """Render a horizontal bar chart of class probabilities using Plotly."""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]

    colors = [
        "#92fe9d" if i == 0 else "rgba(0,201,255,0.5)"
        for i in range(len(labels))
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(color="white", size=12),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=13),
        xaxis=dict(
            title="Confidence (%)",
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=20, t=10, b=40),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: dict) -> None:
    """Render the prediction result card."""
    predicted = result["predicted_class"]
    confidence = result["confidence"]
    suggestion = result.get("recyclability_suggestion", "")
    below_threshold = result.get("below_confidence_threshold", False)

    # ── Confidence color ─────────────────────────────────────────────
    if confidence >= 0.8:
        bar_color = "#92fe9d"
    elif confidence >= 0.5:
        bar_color = "#ffc107"
    else:
        bar_color = "#ff4d4f"

    st.markdown(
        f"""
        <div class="result-card">
            <h2>🔍 Prediction Result</h2>
            <div class="pred-badge">{predicted.upper()}</div>
            <div class="confidence-meter">
                <p style="color:#b0b0c0; margin-bottom:4px;">
                    Confidence: <strong style="color:{bar_color}">{confidence*100:.1f}%</strong>
                </p>
                <div class="confidence-bar">
                    <div class="confidence-fill"
                         style="width:{confidence*100:.1f}%; background:{bar_color};"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if below_threshold:
        st.markdown(
            '<div class="warning-box">⚠️ <strong>Low confidence.</strong> '
            "The model is not very sure about this prediction. "
            "Try a clearer or closer-up image.</div>",
            unsafe_allow_html=True,
        )

    if suggestion:
        st.markdown(
            f'<div class="suggestion-box">💡 <strong>Recyclability:</strong> {suggestion}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Main App
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the Streamlit application."""
    render_header()

    st.markdown("---")

    # ── File uploader ────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📤 Upload a waste image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        help="Supported formats: JPEG, PNG, WebP, BMP, TIFF",
    )

    if uploaded_file is not None:
        # ── Preview ──────────────────────────────────────────────────
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # ── Classify button ──────────────────────────────────────────
        enable_gradcam = st.checkbox("🔥 Enable Grad-CAM visualization", value=False)
        classify_btn = st.button("🔬 Classify Image", type="primary", use_container_width=True)

        if classify_btn:
            with st.spinner("🧠 Analyzing image…"):
                try:
                    # Reset file pointer for upload
                    uploaded_file.seek(0)

                    endpoint = GRADCAM_ENDPOINT if enable_gradcam else PREDICT_ENDPOINT
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    response = requests.post(endpoint, files=files, timeout=60)

                    if response.status_code == 200:
                        result = response.json()

                        with col2:
                            render_result(result)

                        # ── Probability chart ─────────────────────────
                        st.markdown("### 📊 Class Probabilities")
                        render_probability_chart(result["all_probabilities"])

                        # ── Grad-CAM ──────────────────────────────────
                        if enable_gradcam and result.get("gradcam_image"):
                            st.markdown("### 🔥 Grad-CAM Heatmap")
                            import base64
                            from io import BytesIO

                            gradcam_bytes = base64.b64decode(result["gradcam_image"])
                            gradcam_img = Image.open(BytesIO(gradcam_bytes))
                            st.image(
                                gradcam_img,
                                caption="Grad-CAM — Areas the model focused on",
                                use_container_width=True,
                            )

                        # ── Raw JSON (expandable) ─────────────────────
                        with st.expander("📋 Raw API Response"):
                            display_result = {
                                k: v
                                for k, v in result.items()
                                if k != "gradcam_image"
                            }
                            st.json(display_result)
                    else:
                        st.error(
                            f"❌ API Error ({response.status_code}): "
                            f"{response.json().get('detail', response.text)}"
                        )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ **Could not connect to the backend.**\n\n"
                        f"Make sure the FastAPI server is running at `{BACKEND_URL}`.\n\n"
                        "```bash\ncd backend && uvicorn app.main:app --reload\n```"
                    )
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The model may be loading — try again.")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="eco-footer">
            Built with ❤️ using Streamlit & FastAPI &bull;
            EcoVision Garbage Classification &bull;
            10 waste categories
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
