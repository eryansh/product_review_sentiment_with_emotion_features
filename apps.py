import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, pipeline, TFAutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DeBERTa Sentiment & Emotion Fusion",
    page_icon="🤖",
    layout="wide",
)

# --- CONFIGURATION ---
CONFIG = {
    "repo_id": "heryanshah/deberta-electronic-sentiment-text",
    "files": {
        "model_text_only": "ryan_text_deberta_model.h5",
        "model_text_emotion": "ryan_text_emotion_deberta_model.h5"
    },
    # Tokenizer for DeBERTa (must match what you trained with)
    "tokenizer_name": "microsoft/deberta-v3-base", 
    
    # Emotion Model (Using j-hartmann)
    "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
    
    "sentiment_labels": ["Negative", "Neutral", "Positive"],
    "emotion_labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "colors": {
        "Positive": "#22c55e", 
        "Negative": "#ef4444", 
        "Neutral": "#a1a1aa"
    }
}

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def load_components():
    """
    Downloads models from Hugging Face and loads the Tokenizer.
    """
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_name"])
    except:
        # Fallback if v3 fails, try standard base
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    # 2. Load Emotion Classifier (Pipeline)
    # --- CRITICAL FIX: Added framework="tf" to use TensorFlow instead of PyTorch ---
    emotion_pipe = pipeline(
        "text-classification", 
        model=CONFIG["emotion_model"], 
        return_all_scores=True, 
        framework="tf" # <--- Forces TensorFlow mode
    )

    # 3. Download & Load Keras Models (.h5)
    models = {}
    
    # Define custom objects (Often needed for Transformer layers in Keras)
    custom_objects = {"TFAutoModelForSequenceClassification": TFAutoModelForSequenceClassification}

    with st.spinner("Downloading your custom DeBERTa models from Hugging Face... (This happens once)"):
        try:
            # --- Model 1: Text Only ---
            path1 = hf_hub_download(repo_id=CONFIG["repo_id"], filename=CONFIG["files"]["model_text_only"])
            models["text_only"] = tf.keras.models.load_model(path1, custom_objects=custom_objects)
            
            # --- Model 2: Text + Emotion ---
            path2 = hf_hub_download(repo_id=CONFIG["repo_id"], filename=CONFIG["files"]["model_text_emotion"])
            models["text_emotion"] = tf.keras.models.load_model(path2, custom_objects=custom_objects)
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None

    return tokenizer, emotion_pipe, models

# --- PREPROCESSING FOR DEBERTA ---
def prepare_inputs(text, tokenizer, max_len=512):
    """
    Tokenizes text for DeBERTa. 
    Returns dictionary suitable for model.predict() or direct calling.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf' # Return TensorFlow tensors
    )
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

# --- MAIN ANALYSIS ENGINE ---
def analyze(text, tokenizer, emotion_pipe, models):
    # 1. Prepare Text Inputs (Tokens)
    inputs = prepare_inputs(text, tokenizer)
    
    # 2. Extract Emotion Features (for Model 2)
    # Truncate text to 512 for the emotion model
    emo_results = emotion_pipe(text[:512])[0]
    
    # Map scores to the fixed order defined in CONFIG
    scores_dict = {item['label']: item['score'] for item in emo_results}
    emotion_vector = np.array([scores_dict[l] for l in CONFIG["emotion_labels"]]).reshape(1, -1)
    
    # --- PREDICTION: MODEL 1 (Text Only) ---
    try:
        # Try passing dictionary
        pred_probs_1 = models["text_only"].predict(inputs)
    except:
        # Fallback: Try passing list [ids, mask]
        pred_probs_1 = models["text_only"].predict([inputs['input_ids'], inputs['attention_mask']])
        
    # Handle output format
    if hasattr(pred_probs_1, 'logits'):
        pred_probs_1 = tf.nn.softmax(pred_probs_1.logits, axis=1).numpy()[0]
    else:
        pred_probs_1 = pred_probs_1[0] 

    # --- PREDICTION: MODEL 2 (Text + Emotion) ---
    # Assumes input order: [input_ids, attention_mask, emotion_features]
    try:
        pred_probs_2 = models["text_emotion"].predict([
            inputs['input_ids'], 
            inputs['attention_mask'], 
            emotion_vector
        ])
    except:
        st.error("Input Shape Mismatch on Model 2. Ensure it accepts [input_ids, attention_mask, emotion_features].")
        pred_probs_2 = np.zeros((1, 3)) 

    if hasattr(pred_probs_2, 'logits'):
        pred_probs_2 = tf.nn.softmax(pred_probs_2.logits, axis=1).numpy()[0]
    else:
        pred_probs_2 = pred_probs_2[0]

    return {
        "text_only": {
            "label": CONFIG["sentiment_labels"][np.argmax(pred_probs_1)],
            "conf": np.max(pred_probs_1),
            "probs": pred_probs_1
        },
        "text_emotion": {
            "label": CONFIG["sentiment_labels"][np.argmax(pred_probs_2)],
            "conf": np.max(pred_probs_2),
            "probs": pred_probs_2
        },
        "emotion_data": {
            "top_emotion": max(scores_dict, key=scores_dict.get),
            "all_scores": scores_dict
        }
    }

# --- UI HELPERS ---
def draw_bar_chart(probs, title):
    df = pd.DataFrame({'Sentiment': CONFIG["sentiment_labels"], 'Probability': probs * 100})
    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Sentiment'],
        orientation='h',
        marker_color=[CONFIG["colors"][s] for s in CONFIG["sentiment_labels"]]
    ))
    fig.update_layout(
        title=title, 
        xaxis_range=[0, 100], 
        height=200, 
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- APP LAYOUT ---
st.title("⚡ DeBERTa Electronics Sentiment Analysis")
st.markdown("Comparing **Standard DeBERTa** vs **Emotion-Enhanced DeBERTa**")

# Load Resources
tokenizer, emotion_pipe, models = load_components()

if models:
    with st.form("analyze_form"):
        text_input = st.text_area("Enter Electronic Product Review:", height=100, 
                                  value="The battery life is amazing, but the camera quality is really disappointing for this price.")
        submitted = st.form_submit_button("Analyze Sentiment")

    if submitted and text_input:
        with st.spinner("Running Inference on Neural Networks..."):
            results = analyze(text_input, tokenizer, emotion_pipe, models)
        
        # --- RESULTS DISPLAY ---
        col1, col2 = st.columns(2)
        
        # Model 1
        with col1:
            st.subheader("Model 1: Text Only")
            st.metric("Prediction", results["text_only"]["label"], f"{results['text_only']['conf']:.2%}")
            draw_bar_chart(results["text_only"]["probs"], "Confidence Scores")
            
        # Model 2
        with col2:
            st.subheader("Model 2: Text + Emotion")
            st.metric("Prediction", results["text_emotion"]["label"], f"{results['text_emotion']['conf']:.2%}")
            draw_bar_chart(results["text_emotion"]["probs"], "Confidence Scores")
            
            # Show Emotion Info
            st.divider()
            st.caption(f"Detected Emotion: **{results['emotion_data']['top_emotion'].upper()}**")
            
        # Comparison logic
        if results["text_only"]["label"] != results["text_emotion"]["label"]:
            st.info(f"💡 **Insight:** The models disagree! The inclusion of **{results['emotion_data']['top_emotion']}** emotion shifted the prediction.")
        else:
            st.success("✅ Both models agree on the sentiment.")

else:
    st.error("Failed to load models. Please check your Hugging Face Repo ID and filenames.")
