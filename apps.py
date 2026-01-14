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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException

# ===========================
# NEW: Groq + requests imports
# ===========================
import requests
from groq import Groq
import json

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

    # ===========================
    # NEW: Groq model config
    # ===========================
    "groq_model": "llama3-8b-8192",
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

# NEW: username gate
if "username" not in st.session_state:
    st.session_state.username = ""

# NEW: visitor count session guard
if "visitor_counted" not in st.session_state:
    st.session_state.visitor_counted = False

# NEW: LLM debug toggle
if "llm_debug" not in st.session_state:
    st.session_state.llm_debug = False

# =========================================================
# 6) Persistent Storage: SQLite (History + Visitor Count)
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
# 7.5) NEW: Groq client + robust JSON parsing + audit function
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

def groq_review_audit(client, review_text: str):
    """
    Returns dict:
    - review_in_english
    - is_slang
    - electronic_product_review
    - understandable
    - _raw (debug)
    """
    if client is None:
        return {
            "review_in_english": "Unclear",
            "is_slang": "Unclear",
            "electronic_product_review": "Unclear",
            "understandable": "Unclear",
            "_raw": "Groq client not configured (missing GROQ_API_KEY)."
        }

    prompt = f"""
You are a strict classifier.

You MUST respond in JSON only.
NO explanations.
NO markdown.
NO extra text.

Return exactly these keys:
- review_in_english
- is_slang
- electronic_product_review
- understandable

Allowed values:
- review_in_english: "Yes" | "No" | "Unclear"
- is_slang: "Yes" | "No" | "Some"
- electronic_product_review:
    - "General"
    - "Yes"
    - "Specific: <device/type>"
- understandable: "Yes" | "No" | "Partly"

Review:
\"\"\"{review_text}\"\"\"
""".strip()

    try:
        completion = client.chat.completions.create(
            model=CONFIG["groq_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["groq_temperature"],
        )

        raw = (completion.choices[0].message.content or "").strip()
        data = _safe_parse_json(raw)

        if not isinstance(data, dict):
            return {
                "review_in_english": "Unclear",
                "is_slang": "Unclear",
                "electronic_product_review": "Unclear",
                "understandable": "Unclear",
                "_raw": raw
            }

        return {
            "review_in_english": data.get("review_in_english", "Unclear"),
            "is_slang": data.get("is_slang", "Unclear"),
            "electronic_product_review": data.get("electronic_product_review", "Unclear"),
            "understandable": data.get("understandable", "Unclear"),
            "_raw": raw
        }

    except Exception as e:
        return {
            "review_in_english": "Unclear",
            "is_slang": "Unclear",
            "electronic_product_review": "Unclear",
            "understandable": "Unclear",
            "_raw": f"Groq error: {e}"
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

# Load models after gate (saves compute for drive-by visitors)
models = load_all_models()
emotion_classifier = load_emotion_model()

# NEW: Groq client once
groq_client = get_groq_client()

# Sidebar options
with st.sidebar:
    st.markdown("### ⚙️ Options")
    st.session_state.llm_debug = st.toggle("Show LLM raw output (debug)", value=st.session_state.llm_debug)

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

    # --- FEATURE: Demo Selector ---
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

        # --- FEATURE: Language Detection ---
        try:
            detected_lang = detect(user_text)
            if detected_lang != 'en':
                st.warning(f"⚠️ **Warning:** The detected language is **'{detected_lang}'**. This model is trained on English data and may produce inaccurate results for non-English reviews.")
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
        # NEW: Online LLM Review Audit (Groq)
        # =========================================================
        st.markdown("### 🧠 LLM Review Audit (Online - Groq)")

        if groq_client is None:
            st.warning("Groq not configured. Add GROQ_API_KEY to Streamlit secrets (or env var).")
            o = {
                "review_in_english": "Unclear",
                "is_slang": "Unclear",
                "electronic_product_review": "Unclear",
                "understandable": "Unclear",
                "_raw": "Missing GROQ_API_KEY"
            }
        else:
            with st.spinner("LLM is checking the review..."):
                o = groq_review_audit(groq_client, user_text)

        st.markdown(
            f"""
Review in English? **{o.get('review_in_english','Unclear')}**  
Is this Slang? **{o.get('is_slang','Unclear')}**  
Electronic Product Review? **{o.get('electronic_product_review','Unclear')}**  
Review is understandable? **{o.get('understandable','Unclear')}**
            """.strip()
        )

        if st.session_state.llm_debug:
            with st.expander("Show LLM raw output (debug)"):
                st.code(o.get("_raw", ""), language="text")

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
