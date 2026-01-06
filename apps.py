import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import plotly.graph_objects as go
import os

# --- CONFIGURATION ---
CONFIG = {
    "model_names": {
        "without_emotion": "heryanshah/deberta-electronic-sentiment-text",
        "with_emotion": "heryanshah/deberta-emotion-sentiment",
        "emotion_detector": "j-hartmann/emotion-english-distilroberta-base"
    },
    "emotion_labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "sentiment_order": ['Negative', 'Neutral', 'Positive'], 
    "sentiment_color_map": {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#a1a1aa'},
    "emotion_color_map": {'sadness': '#3b82f6', 'joy': '#facc15', 'anger': '#ef4444', 'fear': '#a855f7', 'surprise': '#22d3ee', 'disgust': '#84cc16', 'neutral': '#a1a1aa'}
}

# --- Page Configuration ---
st.set_page_config(
    page_title="DeBERTa Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
)

# --- Initialize Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Asset Loading ---
@st.cache_resource
def load_models():
    """Loads models explicitly using TensorFlow classes to avoid path/config errors."""
    try:
        models = {}
        
        # 1. Load Text-Only Model (TensorFlow)
        st.toast("Loading Model 1 (Text-Only)...", icon="⏳")
        name_1 = CONFIG["model_names"]["without_emotion"]
        tokenizer_1 = AutoTokenizer.from_pretrained(name_1)
        model_1 = TFAutoModelForSequenceClassification.from_pretrained(name_1)
        
        models["without_emotion"] = pipeline(
            "text-classification", 
            model=model_1, 
            tokenizer=tokenizer_1,
            return_all_scores=True
        )
        
        # 2. Load Text + Emotion Model (TensorFlow)
        # We assume this is also a TF model. If this fails, it might be PyTorch.
        st.toast("Loading Model 2 (With Emotion)...", icon="⏳")
        name_2 = CONFIG["model_names"]["with_emotion"]
        tokenizer_2 = AutoTokenizer.from_pretrained(name_2)
        model_2 = TFAutoModelForSequenceClassification.from_pretrained(name_2)
        
        models["with_emotion"] = pipeline(
            "text-classification", 
            model=model_2, 
            tokenizer=tokenizer_2,
            return_all_scores=True 
        )
        
        # 3. Load Emotion Detector (Standard/PyTorch)
        # This is a public model which is usually PyTorch-based, so we let pipeline handle it automatically.
        st.toast("Loading Emotion Detector...", icon="⏳")
        models["emotion_classifier"] = pipeline(
            "text-classification", 
            model=CONFIG["model_names"]["emotion_detector"], 
            return_all_scores=True
        )
        
        st.toast("All models loaded successfully!", icon="✅")
        return models
        
    except Exception as e:
        st.error(f"Error loading models. Please check if 'tf-keras' is installed. Details: {e}")
        return None

# --- Analysis Logic ---
def analyze_sentiment(user_text, models):
    """
    Performs sentiment analysis using the two DeBERTa models and emotion detection.
    """
    
    # --- 1. Detect Emotion (Feature Extraction) ---
    truncated_text = user_text[:512] 
    emotion_scores_raw = models["emotion_classifier"](truncated_text)[0]
    
    # Process emotion scores for visualization
    df_scores = pd.DataFrame(emotion_scores_raw)
    df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
    df_scores['Score'] = df_scores['Score'] * 100
    top_emotion = df_scores.loc[df_scores['Score'].idxmax()]['Emotion']
    
    # --- 2. Model 1 Prediction (Text Only) ---
    pred_raw_1 = models["without_emotion"](user_text[:512])[0]
    
    # --- 3. Model 2 Prediction (Text + Emotion) ---
    input_text_model_2 = user_text 
    pred_raw_2 = models["with_emotion"](input_text_model_2[:512])[0]
    
    # --- Helper to process HF Pipeline Output into standard format ---
    def process_hf_output(raw_output):
        # Convert to dictionary {Label: Score}
        scores = {item['label']: item['score'] for item in raw_output}
        
        # Handle LABEL_0, LABEL_1 mappings if necessary
        if 'LABEL_0' in scores:
             mapped_scores = {}
             for i, label in enumerate(CONFIG["sentiment_order"]):
                 key = f"LABEL_{i}"
                 mapped_scores[label] = scores.get(key, 0.0)
             scores = mapped_scores
        
        # Find predicted label
        predicted_label = max(scores, key=scores.get)
        confidence = scores[predicted_label]
        
        # Create DataFrame
        df = pd.DataFrame(list(scores.items()), columns=['Sentiment', 'Probability'])
        df['Probability'] = df['Probability'] * 100
        # Sort based on fixed order
        df = df.set_index('Sentiment').reindex(CONFIG["sentiment_order"]).reset_index()
        
        return predicted_label, confidence, df

    # Process both outputs
    pred_label_1, conf_1, df_1 = process_hf_output(pred_raw_1)
    pred_label_2, conf_2, df_2 = process_hf_output(pred_raw_2)

    # --- Interpretation & Comparison ---
    is_uncertain1 = conf_1 < 0.40
    is_uncertain2 = conf_2 < 0.40
    
    # Calculate Delta
    row_match = df_1[df_1['Sentiment'] == pred_label_2]
    score_from_model1 = row_match['Probability'].values[0] / 100 if not row_match.empty else 0
    confidence_delta = conf_2 - score_from_model1

    # Generate text
    if is_uncertain1 or is_uncertain2:
        interpretation_text = "The model is **uncertain** (low confidence)."
    elif pred_label_1.lower() != pred_label_2.lower():
        interpretation_text = f"The models **disagree**. The Text-Only model sees **{pred_label_1}**, but with Emotion features, it shifts to **{pred_label_2}**."
    else:
        interpretation_text = f"Both models **agree** on **{pred_label_1}**."

    return {
        "model1": {"prediction": pred_label_1, "confidence": conf_1, "is_uncertain": is_uncertain1, "df": df_1},
        "model2": {"prediction": pred_label_2, "confidence": conf_2, "is_uncertain": is_uncertain2, "df": df_2},
        "emotion": {"df": df_scores, "top": top_emotion},
        "comparison": {"delta": confidence_delta, "text": interpretation_text}
    }

# --- UI Helper Functions ---
def display_sentiment_result(prediction, confidence, is_uncertain, **kwargs):
    if is_uncertain: st.warning("Uncertain")
    elif str(prediction).lower() == 'positive': st.success(f"**Positive** ({confidence:.2%})")
    elif str(prediction).lower() == 'negative': st.error(f"**Negative** ({confidence:.2%})")
    else: st.info(f"**Neutral** ({confidence:.2%})")

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

# --- Main App ---
set_video_background()

st.markdown("""
    <style>
    @import url('https.googleapis.com/css2?family=Poppins:wght@700&display=swap');
    .main-title {
        font-family: 'tahoma', sans-serif;
        font-size: clamp(2.5rem, 8vw, 6rem);
        font-weight: 700;
        text-align: center;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        color: white;
        padding-top: 1rem;
        text-transform: uppercase;
    }
    </style>
    <p class="main-title">DeBERTa Sentiment Comparison</p>
    """, unsafe_allow_html=True)

models = load_models()

if models:
    # Auto-expanding text area
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

    with st.form("sentiment_form"):
        user_text = st.text_area("Enter review text here:", "The battery life of this phone is amazing, I'm so happy with my purchase!")
        submitted = st.form_submit_button("Predict Sentiment")

    if submitted and user_text.strip():
        with st.spinner("Processing with DeBERTa models..."):
            results = analyze_sentiment(user_text, models)
        
        st.session_state.history.insert(0, {
            "text": user_text,
            "model1_pred": results["model1"]["prediction"],
            "model2_pred": results["model2"]["prediction"],
            "top_emotion": results["emotion"]["top"]
        })

        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model 1: DeBERTa (Text Only)")
            st.caption(f"Source: {CONFIG['model_names']['without_emotion']}")
            display_sentiment_result(**results["model1"])
            
            st.markdown("###### Probability Distribution")
            create_bar_chart(results["model1"]["df"], 'Sentiment', 'Probability', CONFIG["sentiment_color_map"], 150)
            
            st.markdown("###### Comparative Interpretation")
            st.info(results["comparison"]["text"])

        with col2:
            st.markdown("#### Model 2: DeBERTa (Text + Emotion)")
            st.caption(f"Source: {CONFIG['model_names']['with_emotion']}")
            display_sentiment_result(**results["model2"])
            
            if not results["model2"]["is_uncertain"]:
                st.metric(
                    label=f"Confidence Shift for '{results['model2']['prediction']}'",
                    value=f"{results['comparison']['delta']:+.2%}"
                )
            
            st.markdown("###### Emotion Analysis (Context)")
            emotion_map = {'sadness': '😢', 'joy': '😂', 'anger': '😠', 'fear': '😨', 'surprise': '😮', 'disgust': '🤢', 'neutral': '😐'}
            top_emo = results["emotion"]["top"]
            
            e_col1, e_col2 = st.columns([1, 4])
            with e_col1:
                st.markdown(f"<div style='text-align: center; font-size: 2.5rem;'>{emotion_map.get(top_emo,'❓')}</div>", unsafe_allow_html=True)
                st.caption(top_emo.capitalize())
            with e_col2:
                top_3_emotions = results["emotion"]["df"].sort_values('Score', ascending=False).head(3)
                create_bar_chart(top_3_emotions, 'Emotion', 'Score', CONFIG["emotion_color_map"], 120, show_x_title=True)
            
    elif submitted:
        st.warning("Please enter some text to analyze.")
    
    # --- HISTORY SECTION ---
    st.divider()
    st.markdown("## Analysis History")

    if not st.session_state.history:
        st.info("Your previous analyses in this session will appear here.")
    else:
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"**{len(st.session_state.history) - i}.** {entry['text'][:70]}..."):
                st.markdown(f"**Input:** _{entry['text']}_")
                st.markdown(f"**Text-Only:** `{entry['model1_pred']}` | **Text+Emotion:** `{entry['model2_pred']}`")
                st.markdown(f"**Emotion:** `{entry['top_emotion']}`")

else:
    st.error("Application could not start. Please check the model files and internet connection.")

# --- CREDIT SECTION ---
st.markdown("""
    <style>
        .footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: rgba(0, 0, 0, 0.5); color: white;
            text-align: center; padding: 10px; font-size: 14px;
        }
    </style>
    <div class="footer">
        DeBERTa Models by Heryanshah Bin Suhimi | FYP Research
    </div>
""", unsafe_allow_html=True)
