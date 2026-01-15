import streamlit as st
import joblib
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
import re
import nltk
import os
import sqlite3
from datetime import datetime
import json
import hashlib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException

# ===========================
# Groq imports
# ===========================
from groq import Groq

# =========================================================
# 0) CONFIG: You only need to change this secret number
# =========================================================
CLEAR_HISTORY_SECRET = "123456"   # <-- CHANGE THIS to your own secret number

# =========================================================
# 1) NLTK Resource Downloads (Robust Version)
# =========================================================
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()

NLTK_DATA_DIR = os.path.join(APP_DIR, "nltk_data")
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# These downloads are kept as in your original code
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR)

# =========================================================
# 2) CONFIGURATION
# =========================================================
CONFIG = {
    "model_paths": {
        "without_emotion": {"pipeline": 'xgb_model_condition1.joblib'},
        "with_emotion": {"pipeline": 'xgb_model_condition2.joblib'}
    },
    "emotion_labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "sentiment_order": ['Negative', 'Neutral', 'Positive'],
    "hugging_face_model": "j-hartmann/emotion-english-distilroberta-base",
    "sentiment_color_map": {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#a1a1aa'},
    "emotion_color_map": {
        'sadness': '#3b82f6',
        'joy': '#facc15',
        'anger': '#ef4444',
        'fear': '#a855f7',
        'surprise': '#22d3ee',
        'disgust': '#84cc16',
        'neutral': '#a1a1aa'
    },

    # ✅ Groq models try order (fallback if first fails)
    "groq_models_try_order": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ],
    "groq_temperature": 0,
}

# =========================================================
# 3) DEMO SCENARIOS
# =========================================================
demo_options = {
    "Select an example...": "",
    "Standard Positive": "The battery life of this phone is amazing, I'm so happy with my purchase!",
    "Standard Negative": "Terrible service. The package arrived late and the item was broken.",
    "Sarcastic (Tricky)": "Oh great, another update that breaks everything. Just what I needed!",
    "Mixed Feelings": "I love the camera quality, but the battery drains way too fast.",
    "Short/Slang": "Omg best purchase everrr! <3",
    "Ambiguous/Neutral": "The product arrived on Tuesday. It is blue.",
    "Non-English (Language Check)": "Barang ini sangat bagus dan berkualiti tinggi."
}

# =========================================================
# 4) Page Configuration
# =========================================================
st.set_page_config(
    page_title="Sentiment Classification with Emotion Features",
    page_icon="🤖",
    layout="wide",
)

# =========================================================
# 5) Session State
# =========================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = "The battery life of this phone is amazing, I'm so happy with my purchase!"

# username gate
if "username" not in st.session_state:
    st.session_state.username = ""

# visitor count session guard
if "visitor_counted" not in st.session_state:
    st.session_state.visitor_counted = False

# LLM debug toggle
if "llm_debug" not in st.session_state:
    st.session_state.llm_debug = False

# =========================================================
# 6) Persistent Storage: SQLite (History + Visitor Count + LLM Cache)
# =========================================================
DB_PATH = os.path.join(APP_DIR, "app_storage.db")

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()

    # Shared history (visible to everyone)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            username TEXT NOT NULL,
            text TEXT NOT NULL,
            model1_pred TEXT NOT NULL,
            model2_pred TEXT NOT NULL,
            top_emotion TEXT NOT NULL
        )
    """)

    # Counters (visitor count)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS counters (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)

    # LLM cache (avoid repeated calls)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            text_hash TEXT PRIMARY KEY,
            audit_json TEXT,
            sentiment_json TEXT,
            ts TEXT NOT NULL
        )
    """)

    # Initialize visitor counter if not exists
    cur.execute("INSERT OR IGNORE INTO counters (key, value) VALUES (?, ?)", ("visitors", 0))
    conn.commit()
    return conn

def get_visitor_count(conn):
    cur = conn.cursor()
    cur.execute("SELECT value FROM counters WHERE key = ?", ("visitors",))
    row = cur.fetchone()
    return int(row[0]) if row else 0

def increment_visitor_count_once_per_session(conn):
    if not st.session_state.visitor_counted:
        cur = conn.cursor()
        cur.execute("UPDATE counters SET value = value + 1 WHERE key = ?", ("visitors",))
        conn.commit()
        st.session_state.visitor_counted = True

def add_history_entry(conn, entry: dict):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history (ts, username, text, model1_pred, model2_pred, top_emotion)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        entry["ts"],
        entry["username"],
        entry["text"],
        entry["model1_pred"],
        entry["model2_pred"],
        entry["top_emotion"]
    ))
    conn.commit()

def read_shared_history(conn, limit=200):
    cur = conn.cursor()
    cur.execute("""
        SELECT ts, username, text, model1_pred, model2_pred, top_emotion
        FROM history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    results = []
    for r in rows:
        results.append({
            "timestamp": r[0],
            "username": r[1],
            "text": r[2],
            "model1_pred": r[3],
            "model2_pred": r[4],
            "top_emotion": r[5],
        })
    return results

def clear_shared_history(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()

# -------- LLM cache helpers --------
def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def read_llm_cache(conn, text_hash: str):
    cur = conn.cursor()
    cur.execute("SELECT audit_json, sentiment_json FROM llm_cache WHERE text_hash = ?", (text_hash,))
    row = cur.fetchone()
    if not row:
        return None
    audit_json, sentiment_json = row
    return {
        "audit": json.loads(audit_json) if audit_json else None,
        "sentiment": json.loads(sentiment_json) if sentiment_json else None
    }

def write_llm_cache(conn, text_hash: str, audit_obj: dict = None, sentiment_obj: dict = None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO llm_cache (text_hash, audit_json, sentiment_json, ts)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(text_hash) DO UPDATE SET
            audit_json = COALESCE(excluded.audit_json, llm_cache.audit_json),
            sentiment_json = COALESCE(excluded.sentiment_json, llm_cache.sentiment_json),
            ts = excluded.ts
    """, (
        text_hash,
        json.dumps(audit_obj) if audit_obj else None,
        json.dumps(sentiment_obj) if sentiment_obj else None,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

# =========================================================
# 7) Asset Loading
# =========================================================
@st.cache_resource
def load_all_models():
    """Loads all joblib model files."""
    try:
        models = {
            "without_emotion": joblib.load(CONFIG["model_paths"]["without_emotion"]["pipeline"]),
            "with_emotion": joblib.load(CONFIG["model_paths"]["with_emotion"]["pipeline"])
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Error: A model file was not found. Please ensure all .joblib files are present. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None

@st.cache_resource
def load_emotion_model():
    """Loads the emotion detection model from Hugging Face."""
    try:
        return pipeline("text-classification", model=CONFIG["hugging_face_model"], return_all_scores=True)
    except Exception as e:
        st.error(f"Could not load the emotion model from Hugging Face. Please check the internet connection. Error: {e}")
        return None

# =========================================================
# 7.5) Groq client + JSON parsing + LLM functions
# =========================================================
@st.cache_resource
def get_groq_client():
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)

def _safe_parse_json(text: str):
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _groq_call_json(client, prompt: str):
    """
    Tries models in order and returns:
    { "data": dict_or_None, "raw": raw_text, "model": model_used, "error": err_or_None }
    """
    if client is None:
        return {"data": None, "raw": "", "model": None, "error": "Missing GROQ_API_KEY"}

    last_err = None
    for model_name in CONFIG["groq_models_try_order"]:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=CONFIG["groq_temperature"],
            )
            raw = (completion.choices[0].message.content or "").strip()
            data = _safe_parse_json(raw)
            return {"data": data, "raw": raw, "model": model_name, "error": None}
        except Exception as e:
            last_err = f"{model_name}: {e}"
            continue

    return {"data": None, "raw": "", "model": None, "error": last_err}

def groq_review_audit(client, review_text: str):
    """
    Upgraded audit with:
    - languages detected (can be multiple)
    - slang examples + count + justification
    - electronic vs general vs other domain + guesses + justification
    - understandable + justification
    """
    if client is None:
        return {
            "review_in_english": "Unclear",
            "languages_detected": [],
            "language_note": "Groq client not configured.",
            "is_slang": "Unclear",
            "slang_terms_found": [],
            "slang_count": 0,
            "slang_justification": "Groq client not configured.",
            "electronic_product_review": "Unclear",
            "electronic_guess": "",
            "other_domain_guess": "",
            "product_justification": "Groq client not configured.",
            "understandable": "Unclear",
            "understandable_justification": "Groq client not configured.",
            "_raw": "Missing GROQ_API_KEY",
            "_model": None
        }

    prompt = f"""
You are a strict review auditor.

You MUST respond in JSON only.
NO markdown.
NO extra text.
NO code fences.

Return EXACTLY these keys:
- review_in_english
- languages_detected
- language_note
- is_slang
- slang_terms_found
- slang_count
- slang_justification
- electronic_product_review
- electronic_guess
- other_domain_guess
- product_justification
- understandable
- understandable_justification

Rules:
1) review_in_english must be one of: "Yes", "No", "Mixed", "Unclear"
2) languages_detected is a list of language names (e.g., ["English"], or ["English","Malay"]).
   - If multiple languages are used, include all major ones you can detect (max 3).
3) language_note:
   - If any non-English exists, include warning:
     "Non-English text may reduce the accuracy of the sentiment models."
   - Otherwise, brief note like "Appears fully English."
4) is_slang must be one of: "Yes", "No", "Some", "Unclear"
5) slang_terms_found:
   - list the slang words/phrases you saw in the review (lowercase).
   - If none, return [].
6) slang_count:
   - integer count of unique slang terms found (length of slang_terms_found).
7) slang_justification:
   - short reason (max 25 words), mention why those terms are slang.
8) electronic_product_review must be one of:
   - "Yes"       (clearly electronic product review)
   - "General"   (no specific product OR doesn't sound like any product review)
   - "OtherDomain" (sounds like a review but for a non-electronic domain)
   - "Unclear"
9) If electronic_product_review == "Yes":
   - electronic_guess: "This may refer to product like ____" (be specific)
   - other_domain_guess: ""
10) If electronic_product_review == "OtherDomain":
   - other_domain_guess: "Domain: <domain>, Possible product/service: <thing>"
   - electronic_guess: ""
11) If electronic_product_review == "General":
   - both guesses can be "" (or explain very briefly in product_justification).
12) product_justification:
   - short reason (max 25 words) explaining why "Yes"/"General"/"OtherDomain"
13) understandable must be one of: "Yes", "Partly", "No", "Unclear"
14) understandable_justification:
   - short reason (max 25 words). If sarcasm/fragmented/ambiguous, say so.

Review:
\"\"\"{review_text}\"\"\"
""".strip()

    res = _groq_call_json(client, prompt)
    data = res["data"]

    if not isinstance(data, dict):
        return {
            "review_in_english": "Unclear",
            "languages_detected": [],
            "language_note": "LLM failed to output valid JSON.",
            "is_slang": "Unclear",
            "slang_terms_found": [],
            "slang_count": 0,
            "slang_justification": "LLM failed to output valid JSON.",
            "electronic_product_review": "Unclear",
            "electronic_guess": "",
            "other_domain_guess": "",
            "product_justification": "LLM failed to output valid JSON.",
            "understandable": "Unclear",
            "understandable_justification": "LLM failed to output valid JSON.",
            "_raw": res["raw"] or f"Groq error: {res['error']}",
            "_model": res["model"]
        }

    languages = data.get("languages_detected", [])
    if not isinstance(languages, list):
        languages = []

    slang_terms = data.get("slang_terms_found", [])
    if not isinstance(slang_terms, list):
        slang_terms = []

    try:
        slang_count = int(data.get("slang_count", len(slang_terms)))
    except Exception:
        slang_count = len(slang_terms)

    return {
        "review_in_english": data.get("review_in_english", "Unclear"),
        "languages_detected": languages[:3],
        "language_note": data.get("language_note", ""),
        "is_slang": data.get("is_slang", "Unclear"),
        "slang_terms_found": [str(x).strip().lower() for x in slang_terms if str(x).strip()][:15],
        "slang_count": slang_count,
        "slang_justification": data.get("slang_justification", ""),
        "electronic_product_review": data.get("electronic_product_review", "Unclear"),
        "electronic_guess": data.get("electronic_guess", ""),
        "other_domain_guess": data.get("other_domain_guess", ""),
        "product_justification": data.get("product_justification", ""),
        "understandable": data.get("understandable", "Unclear"),
        "understandable_justification": data.get("understandable_justification", ""),
        "_raw": res["raw"],
        "_model": res["model"]
    }

def groq_sentiment_predict(client, review_text: str):
    """
    LLM sentiment prediction (independent from XGBoost models).
    Returns:
      - sentiment: Positive | Neutral | Negative | Unclear
      - confidence: 0..1 (float) or None
      - reason: short text
    """
    if client is None:
        return {
            "sentiment": "Unclear",
            "confidence": None,
            "reason": "Groq client not configured.",
            "_raw": "Missing GROQ_API_KEY",
            "_model": None
        }

    prompt = f"""
You are a sentiment classifier for product reviews.

You MUST respond in JSON only.
NO explanations outside JSON.
NO markdown.

Return exactly these keys:
- sentiment   (one of: "Positive", "Neutral", "Negative")
- confidence  (number between 0 and 1)
- reason      (max 20 words)

Review:
\"\"\"{review_text}\"\"\"
""".strip()

    res = _groq_call_json(client, prompt)
    data = res["data"]

    if not isinstance(data, dict):
        return {
            "sentiment": "Unclear",
            "confidence": None,
            "reason": "LLM failed to respond in JSON.",
            "_raw": res["raw"] or f"Groq error: {res['error']}",
            "_model": res["model"]
        }

    sentiment = data.get("sentiment", "Unclear")
    conf = data.get("confidence", None)
    reason = data.get("reason", "")

    if isinstance(sentiment, str):
        sentiment = sentiment.strip().capitalize()
        if sentiment not in ["Positive", "Neutral", "Negative"]:
            sentiment = "Unclear"
    else:
        sentiment = "Unclear"

    try:
        conf = float(conf)
        if conf < 0 or conf > 1:
            conf = None
    except Exception:
        conf = None

    if not isinstance(reason, str):
        reason = ""

    return {
        "sentiment": sentiment,
        "confidence": conf,
        "reason": reason.strip(),
        "_raw": res["raw"],
        "_model": res["model"]
    }

# =========================================================
# 8) Preprocessing Function
# =========================================================
@st.cache_data
def preprocess_text(text):
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_DIR)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(processed_tokens)

# =========================================================
# 9) Analysis Logic
# =========================================================
def analyze_sentiment(user_text, models, emotion_classifier):
    processed_text = preprocess_text(user_text)

    # --- Model 1: Without Emotion ---
    pipeline_cond1 = models["without_emotion"]
    prediction_proba = pipeline_cond1.predict_proba([processed_text])
    predicted_index = np.argmax(prediction_proba)
    predicted_label = CONFIG["sentiment_order"][predicted_index]

    # --- Model 2: With Emotion ---
    pipeline_cond2 = models["with_emotion"]
    truncated_text = user_text[:512]
    emotion_scores_raw = emotion_classifier(truncated_text)[0]

    scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
    emotion_features = np.array([scores_dict[l] for l in CONFIG["emotion_labels"]]).reshape(1, -1)

    emotion_data = {f"prob_{label}": score for label, score in zip(CONFIG["emotion_labels"], emotion_features[0])}
    data_dict = {'final_preprocessed_text': [processed_text], **emotion_data}
    input_df = pd.DataFrame(data_dict)

    prediction_proba_emo = pipeline_cond2.predict_proba(input_df)
    predicted_index_emo = np.argmax(prediction_proba_emo)
    predicted_label_emo = CONFIG["sentiment_order"][predicted_index_emo]

    # --- DataFrames for Plotting ---
    df_proba = pd.DataFrame({'Sentiment': CONFIG["sentiment_order"], 'Probability': prediction_proba[0] * 100})
    df_proba = df_proba.set_index('Sentiment').reindex(CONFIG["sentiment_order"]).reset_index()

    df_proba_emo = pd.DataFrame({'Sentiment': CONFIG["sentiment_order"], 'Probability': prediction_proba_emo[0] * 100})
    df_proba_emo = df_proba_emo.set_index('Sentiment').reindex(CONFIG["sentiment_order"]).reset_index()

    df_scores = pd.DataFrame(emotion_scores_raw)
    df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
    df_scores['Score'] = df_scores['Score'] * 100
    top_emotion = df_scores.loc[df_scores['Score'].idxmax()]['Emotion']

    # --- Interpretation ---
    confidence = np.max(prediction_proba)
    confidence_emo = np.max(prediction_proba_emo)
    is_uncertain1 = np.isclose(confidence, 1/3, atol=0.05)
    is_uncertain2 = np.isclose(confidence_emo, 1/3, atol=0.05)

    confidence_from_model1 = prediction_proba[0][predicted_index_emo]
    confidence_delta = confidence_emo - confidence_from_model1

    if is_uncertain1 or is_uncertain2:
        interpretation_text = "The model is **uncertain** because the input text is too short or contains words not in its vocabulary."
    elif predicted_label.lower() != predicted_label_emo.lower():
        interpretation_text = f"These models **disagree**. Model 1 predicts **{predicted_label.capitalize()}**, while Model 2 predicts **{predicted_label_emo.capitalize()}**. "
    else:
        interpretation_text = f"Both models **agree** that the sentiment is **{predicted_label.capitalize()}**. "

    if not (is_uncertain1 or is_uncertain2):
        if top_emotion != 'neutral':
            interpretation_text += f"The detection of strong **{top_emotion.capitalize()}** emotion likely influenced Model 2, leading to a more nuanced prediction."
        else:
            interpretation_text += f"This text was detected as emotionally **Neutral**, helping Model 2 produce a balanced sentiment prediction."

    return {
        "model1": {"prediction": predicted_label, "confidence": confidence, "is_uncertain": is_uncertain1, "df": df_proba},
        "model2": {"prediction": predicted_label_emo, "confidence": confidence_emo, "is_uncertain": is_uncertain2, "df": df_proba_emo},
        "emotion": {"df": df_scores, "top": top_emotion},
        "comparison": {"delta": confidence_delta, "text": interpretation_text},
        "processed_text": processed_text
    }

# =========================================================
# 10) UI Helper Functions
# =========================================================
def display_sentiment_result(prediction, confidence, is_uncertain, **kwargs):
    if is_uncertain:
        st.warning("Model is uncertain due to unrecognized input.")
    elif str(prediction).lower() == 'positive':
        st.success(f"**Positive** (Confidence: {confidence:.2%})")
    elif str(prediction).lower() == 'negative':
        st.error(f"**Negative** (Confidence: {confidence:.2%})")
    else:
        st.info(f"**Neutral** (Confidence: {confidence:.2%})")

def create_bar_chart(df, y_col, x_col, color_map, height, show_x_title=False):
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            y=[row[y_col].capitalize()],
            x=[row[x_col]],
            name=row[y_col].capitalize(),
            orientation='h',
            marker_color=color_map.get(row[y_col], '#888')
        ))

    xaxis_config = dict(range=[0, 100], showgrid=False)
    if show_x_title:
        xaxis_config['title'] = "Score (%)"

    fig.update_layout(
        showlegend=False,
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=xaxis_config,
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#fff")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def set_video_background():
    video_url = "https://raw.githubusercontent.com/eryansh/product_review_sentiment_with_emotion_features/main/background.mp4"
    st.markdown(f"""
        <style>
        .stApp {{ background: transparent; }}
        #bg-video {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; object-fit: cover; z-index: -1; }}
        </style>
        <video id="bg-video" autoplay loop muted><source src="{video_url}" type="video/mp4"></video>
        """, unsafe_allow_html=True)

# =========================================================
# 11) Main App Execution
# =========================================================
conn = get_conn()

# Visitor count (once per session)
increment_visitor_count_once_per_session(conn)
visitor_count = get_visitor_count(conn)

set_video_background()

st.markdown("""
    <style>
    @import url('https.googleapis.com/css2?family=Poppins:wght@700&display=swap');
    .main-title {
        font-family: 'tahoma', sans-serif;
        font-size: clamp(2.5rem, 8vw, 7rem);
        font-weight: 700;
        text-align: center;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        padding-top: 1rem;
        padding-bottom: 1rem;
        text-transform: uppercase;
    }
    </style>
    <p class="main-title">Sentiment Classification with Emotion Features</p>
    """, unsafe_allow_html=True)

# Show visitor count (always)
st.markdown(f"**👥 Visitors:** `{visitor_count}`")

# -----------------------------
# Username Gate (no login)
# -----------------------------
if not st.session_state.username.strip():
    st.markdown("## 👋 Welcome")
    st.markdown("Before using the app, please enter your name (this will be shown in shared history).")

    with st.form("name_gate"):
        name_in = st.text_input("Your name:", placeholder="e.g., Ali / Siti / John")
        ok = st.form_submit_button("Continue")

    if ok:
        st.session_state.username = name_in.strip()
        st.rerun()

    st.stop()

# Load models after gate
models = load_all_models()
emotion_classifier = load_emotion_model()
groq_client = get_groq_client()

# Sidebar options
with st.sidebar:
    st.markdown("### ⚙️ Options")
    st.session_state.llm_debug = st.toggle("Show LLM raw output (debug)", value=st.session_state.llm_debug)
    st.caption("Groq models try order:")
    st.code("\n".join(CONFIG["groq_models_try_order"]), language="text")

if models and emotion_classifier:
    st.markdown("""
        <style> textarea[aria-label="Enter review text here:"] { resize: none; overflow-y: hidden; } </style>
        <script>
            function setupAutoExpand() {
                const textarea = document.querySelector('textarea[aria-label="Enter review text here:"]');
                if (textarea && !textarea.hasAttribute('data-auto-expand-setup')) {
                    const adjustHeight = () => { textarea.style.height = 'auto'; textarea.style.height = (textarea.scrollHeight) + 'px'; };
                    textarea.addEventListener('input', adjustHeight);
                    textarea.setAttribute('data-auto-expand-setup', 'true');
                    setTimeout(adjustHeight, 100);
                }
            }
            setTimeout(setupAutoExpand, 200);
        </script>
    """, unsafe_allow_html=True)

    # --- Demo Selector ---
    def update_text_area():
        selected_example = st.session_state.example_selector
        if selected_example and demo_options[selected_example]:
            st.session_state.user_input = demo_options[selected_example]

    st.markdown("### 🧪 Test Scenarios")
    st.selectbox(
        "Choose a pre-defined review to test:",
        options=list(demo_options.keys()),
        key="example_selector",
        on_change=update_text_area,
        index=0
    )

    with st.form("sentiment_form"):
        user_text = st.text_area("Enter review text here:", key="user_input")
        submitted = st.form_submit_button("Predict Sentiment")

    if submitted and user_text.strip():

        # --- Language Detection (local quick warning) ---
        try:
            detected_lang = detect(user_text)
            if detected_lang != 'en':
                st.warning(
                    f"⚠️ **Warning:** The detected language is **'{detected_lang}'**. "
                    f"This model is trained on English data and may produce inaccurate results for non-English reviews."
                )
        except LangDetectException:
            st.warning("⚠️ **Warning:** Could not detect the language. Results may be inaccurate.")

        with st.spinner("Analyzing text..."):
            results = analyze_sentiment(user_text, models, emotion_classifier)

        # --- Store Session History (local, current user session only) ---
        st.session_state.history.insert(0, {
            "text": user_text,
            "model1_pred": results["model1"]["prediction"],
            "model2_pred": results["model2"]["prediction"],
            "top_emotion": results["emotion"]["top"]
        })

        # --- Store Shared History (visible to everyone) ---
        add_history_entry(conn, {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": st.session_state.username,
            "text": user_text,
            "model1_pred": results["model1"]["prediction"],
            "model2_pred": results["model2"]["prediction"],
            "top_emotion": results["emotion"]["top"]
        })

        st.divider()

        # --- Preprocessed Text Debugger ---
        with st.expander("Show Preprocessed Text (for XGBoost models)"):
            st.markdown("**Original Text:**")
            st.info(user_text)
            st.markdown("**Processed Text (Input for Model 1 & 2):**")
            if results["processed_text"].strip():
                st.success(results["processed_text"])
            else:
                st.warning("Text was empty after preprocessing.")

        # =========================================================
        # LLM CACHE (audit + sentiment)
        # =========================================================
        text_hash = _hash_text(user_text)
        cached = read_llm_cache(conn, text_hash)

        # =========================================================
        # LLM Review Audit (Groq) - upgraded outputs
        # =========================================================
        st.markdown("### 🧠 LLM Review Audit (Online - Groq)")

        if cached and cached.get("audit"):
            o = cached["audit"]
            o["_cached"] = True
        else:
            with st.spinner("LLM is checking the review..."):
                o = groq_review_audit(groq_client, user_text)
            write_llm_cache(conn, text_hash, audit_obj=o)
            o["_cached"] = False

        langs = o.get("languages_detected", [])
        langs_text = ", ".join(langs) if langs else "Unclear"

        # Language / English
        st.markdown(
            f"""
**Review in English?** {o.get('review_in_english','Unclear')}  
**Languages detected:** {langs_text}  
**Note:** {o.get('language_note','')}
            """.strip()
        )

        # Slang details
        slang_terms = o.get("slang_terms_found", [])
        slang_list = ", ".join(slang_terms) if slang_terms else "(none)"
        st.markdown(
            f"""
**Is this Slang?** {o.get('is_slang','Unclear')}  
**Slang words/phrases found:** {slang_list}  
**Number of slang terms:** {o.get('slang_count', 0)}  
**Justification:** {o.get('slang_justification','')}
            """.strip()
        )

        # Electronic / domain details
        st.markdown(
            f"""
**Electronic Product Review?** {o.get('electronic_product_review','Unclear')}  
**Electronic guess:** {o.get('electronic_guess','')}  
**Other domain guess:** {o.get('other_domain_guess','')}  
**Justification:** {o.get('product_justification','')}
            """.strip()
        )

        # Understandable details
        st.markdown(
            f"""
**Review is understandable?** {o.get('understandable','Unclear')}  
**Justification:** {o.get('understandable_justification','')}
            """.strip()
        )

        st.caption("✅ Cached result" if o.get("_cached") else "🌐 Live API result")

        # =========================================================
        # LLM Sentiment Prediction (Groq)
        # =========================================================
        st.markdown("### 🤖 LLM Sentiment Prediction (Groq)")

        if cached and cached.get("sentiment"):
            s = cached["sentiment"]
            s["_cached"] = True
        else:
            with st.spinner("LLM is predicting sentiment..."):
                s = groq_sentiment_predict(groq_client, user_text)
            write_llm_cache(conn, text_hash, sentiment_obj=s)
            s["_cached"] = False

        llm_sent = s.get("sentiment", "Unclear")
        llm_conf = s.get("confidence", None)
        llm_reason = s.get("reason", "")

        if llm_sent == "Positive":
            st.success(f"**LLM Sentiment:** Positive" + (f"  (Confidence: {llm_conf:.2%})" if isinstance(llm_conf, float) else ""))
        elif llm_sent == "Negative":
            st.error(f"**LLM Sentiment:** Negative" + (f"  (Confidence: {llm_conf:.2%})" if isinstance(llm_conf, float) else ""))
        elif llm_sent == "Neutral":
            st.info(f"**LLM Sentiment:** Neutral" + (f"  (Confidence: {llm_conf:.2%})" if isinstance(llm_conf, float) else ""))
        else:
            st.warning("**LLM Sentiment:** Unclear")

        if llm_reason:
            st.caption(f"Reason: {llm_reason}")

        st.caption("✅ Cached result" if s.get("_cached") else "🌐 Live API result")

        # Debug raw outputs
        if st.session_state.llm_debug:
            with st.expander("Show LLM raw outputs (debug)"):
                st.markdown("**Audit raw:**")
                st.caption(f"Model used: {o.get('_model')}")
                st.code(o.get("_raw", ""), language="text")
                st.markdown("**Sentiment raw:**")
                st.caption(f"Model used: {s.get('_model')}")
                st.code(s.get("_raw", ""), language="text")

        st.divider()

        # --- Results Columns ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model 1: Textual Features Only")
            display_sentiment_result(**results["model1"])
            st.markdown("###### Sentiment Probability Comparison")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.markdown("<p style='text-align: center;'>Without Emotion</p>", unsafe_allow_html=True)
                create_bar_chart(results["model1"]["df"], 'Sentiment', 'Probability', CONFIG["sentiment_color_map"], 180)
            with prob_col2:
                st.markdown("<p style='text-align: center;'>With Emotion</p>", unsafe_allow_html=True)
                create_bar_chart(results["model2"]["df"], 'Sentiment', 'Probability', CONFIG["sentiment_color_map"], 180)
            st.markdown("###### Interpretation of Results")
            st.info(results["comparison"]["text"])

        with col2:
            st.markdown("#### Model 2: Textual Features + Emotion Probabilistic Scores")
            display_sentiment_result(**results["model2"])
            if not results["model2"]["is_uncertain"]:
                st.metric(
                    label=f"Confidence Shift for '{results['model2']['prediction'].capitalize()}'",
                    value=f"{results['comparison']['delta']:+.2%}",
                    help="How much the confidence changed for this sentiment class after adding emotion features."
                )
            st.markdown("###### Emotion Analysis (Input Feature)")
            emotion_map = {'sadness': '😢', 'joy': '😂', 'anger': '😠', 'fear': '😨', 'surprise': '😮', 'disgust': '🤢', 'neutral': '😐'}
            top_emotion = results["emotion"]["top"]
            sub_col1, sub_col2 = st.columns([1, 3])
            with sub_col1:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<p style='font-size: 3rem; margin-bottom: 0;'>{emotion_map.get(top_emotion,'❓')}</p>"
                    f"<p style='font-weight: bold;'>{top_emotion.capitalize()}</p></div>",
                    unsafe_allow_html=True
                )
            with sub_col2:
                sorted_emotions = results["emotion"]["df"].sort_values('Score', ascending=True)
                create_bar_chart(sorted_emotions, 'Emotion', 'Score', CONFIG["emotion_color_map"], 220, show_x_title=True)

    elif submitted:
        st.warning("Please enter some text to analyze.")

    # =========================================================
    # 12) Shared History Section (Visible to everyone)
    # =========================================================
    st.divider()
    st.markdown("## Analysis History (Shared)")

    # Secret number box + clear button (no login)
    st.markdown("### 🧹 Clear History (Secret Number)")
    secret_input = st.text_input(
        "Enter secret number to delete shared history:",
        type="password",
        placeholder="(Only owner knows this)"
    )
    if st.button("Delete Shared History"):
        if secret_input == CLEAR_HISTORY_SECRET:
            clear_shared_history(conn)
            st.success("✅ Shared history deleted.")
            st.rerun()
        else:
            st.error("❌ Wrong secret number.")

    shared_history = read_shared_history(conn, limit=200)

    if not shared_history:
        st.info("No shared history yet. Run a prediction to create entries.")
    else:
        for i, entry in enumerate(shared_history, start=1):
            username = entry.get("username", "Unknown")
            ts = entry.get("timestamp", "")
            text = entry.get("text", "")
            preview = (text[:70] + "...") if len(text) > 70 else text

            with st.expander(f"**{i}.** {preview}  —  👤 {username}  |  🕒 {ts}"):
                st.markdown(f"**User:** `{username}`")
                if ts:
                    st.markdown(f"**Time:** `{ts}`")
                st.markdown(f"**Input Text:** _{text}_")
                st.markdown(f"**Model 1 (Text Only Prediction):** `{entry.get('model1_pred','')}`")
                st.markdown(f"**Model 2 (Text + Emotion Prediction):** `{entry.get('model2_pred','')}`")
                top_emo = entry.get("top_emotion", "")
                st.markdown(f"**Detected Top Emotion:** `{top_emo.capitalize() if isinstance(top_emo, str) else top_emo}`")

else:
    st.error("Application could not start. Please check the model files and internet connection.")

# --- Footer ---
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
    </style>
    <div class="footer">
        Model deployed by Heryanshah Bin Suhimi | This web application is for FYP research purposes only.
    </div>
""", unsafe_allow_html=True)
