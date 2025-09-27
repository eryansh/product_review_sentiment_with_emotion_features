import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis Comparison",
    page_icon="🤖",
    layout="wide",
)

# --- Asset Loading ---
@st.cache_resource
def load_all_models():
    """Loads all joblib model files."""
    try:
        models = {
            "without_emotion": (
                joblib.load('tfidf_vectorizer.joblib'),
                joblib.load('chi2_selector.joblib'),
                joblib.load('naive_bayes_model.joblib')
            ),
            "with_emotion": (
                joblib.load('tfidf_vectorizer_emo.joblib'),
                joblib.load('chi2_selector_emo.joblib'),
                joblib.load('naive_bayes_model_emo.joblib')
            )
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Error: A model file was not found. Please ensure all 6 .joblib files are present. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None

@st.cache_resource
def load_emotion_model():
    """Loads the emotion detection model from Hugging Face."""
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    except Exception as e:
        st.error(f"Could not load the emotion model from Hugging Face. Please check the internet connection. Error: {e}")
        return None

# --- Analysis Logic ---
def analyze_sentiment(user_text, models, emotion_classifier):
    """
    Performs sentiment and emotion analysis and returns all calculated results.
    This function separates the calculation logic from the display logic.
    """
    # --- Model 1: Without Emotion ---
    tfidf, selector, nb_model = models["without_emotion"]
    text_tfidf = tfidf.transform([user_text])
    text_chi2 = selector.transform(text_tfidf)
    prediction_proba = nb_model.predict_proba(text_chi2)
    predicted_label = nb_model.classes_[np.argmax(prediction_proba)]
    
    # --- Model 2: With Emotion ---
    tfidf_emo, selector_emo, nb_model_emo = models["with_emotion"]
    truncated_text = user_text[:512]  # Truncate for RoBERTa model limit
    emotion_scores_raw = emotion_classifier(truncated_text)[0]
    
    labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
    emotion_features = np.array([scores_dict[l] for l in labels]).reshape(1, -1)
    
    text_tfidf_emo = tfidf_emo.transform([user_text])
    text_chi2_emo = selector_emo.transform(text_tfidf_emo)
    final_features = hstack([text_chi2_emo, emotion_features])
    prediction_proba_emo = nb_model_emo.predict_proba(final_features)
    predicted_label_emo = nb_model_emo.classes_[np.argmax(prediction_proba_emo)]
    
    # --- DataFrames for Plotting ---
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    df_proba = pd.DataFrame({'Sentiment': nb_model.classes_, 'Probability': prediction_proba[0] * 100})
    df_proba = df_proba.set_index('Sentiment').reindex(sentiment_order).reset_index()

    df_proba_emo = pd.DataFrame({'Sentiment': nb_model_emo.classes_, 'Probability': prediction_proba_emo[0] * 100})
    df_proba_emo = df_proba_emo.set_index('Sentiment').reindex(sentiment_order).reset_index()

    df_scores = pd.DataFrame(emotion_scores_raw)
    df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
    df_scores['Score'] = df_scores['Score'] * 100
    top_emotion = df_scores.loc[df_scores['Score'].idxmax()]['Emotion']

    # --- Interpretation & Comparison ---
    confidence = np.max(prediction_proba)
    confidence_emo = np.max(prediction_proba_emo)
    is_uncertain1 = np.isclose(confidence, 1/3)
    is_uncertain2 = np.isclose(confidence_emo, 1/3)
    
    predicted_class_index_emo = np.argmax(prediction_proba_emo)
    confidence_from_model1 = prediction_proba[0][predicted_class_index_emo]
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
        "comparison": {"delta": confidence_delta, "text": interpretation_text}
    }

# --- UI Helper Functions ---
def display_sentiment_result(prediction, confidence, is_uncertain, **kwargs):
    """Displays the formatted sentiment result."""
    if is_uncertain: st.warning("Model is uncertain due to unrecognized input.")
    elif str(prediction).lower() == 'positive': st.success(f"**Positive** (Confidence: {confidence:.2%})")
    elif str(prediction).lower() == 'negative': st.error(f"**Negative** (Confidence: {confidence:.2%})")
    else: st.info(f"**Neutral** (Confidence: {confidence:.2%})")

def create_sentiment_chart(df):
    """Creates a sentiment probability bar chart."""
    sentiment_color_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#a1a1aa'}
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(y=[row['Sentiment'].capitalize()], x=[row['Probability']], name=row['Sentiment'].capitalize(), orientation='h', marker_color=sentiment_color_map.get(row['Sentiment'], '#888')))
    fig.update_layout(showlegend=False, height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showgrid=False), yaxis=dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#fff"))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def set_video_background():
    """Injects HTML for a video background."""
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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        text-align: center;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        padding-top: 2rem;
    }
    .sub-title {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.1rem;
        padding-bottom: 2rem;
    }
    </style>
    <p class="main-title">🤖 Sentiment Analysis Comparison</p>
    <p class="sub-title">Compare sentiment predictions from two models: one using text only, and another enriched with emotion features.</p>
    """, unsafe_allow_html=True)

models = load_all_models()
emotion_classifier = load_emotion_model()

if models and emotion_classifier:
    st.markdown("""
        <style> textarea[aria-label="Enter review text here:"] {{ resize: none; overflow-y: hidden; }} </style>
        <script>
            function setupAutoExpand() {{
                const textarea = document.querySelector('textarea[aria-label="Enter review text here:"]');
                if (textarea && !textarea.hasAttribute('data-auto-expand-setup')) {{
                    const adjustHeight = () => {{ textarea.style.height = 'auto'; textarea.style.height = (textarea.scrollHeight) + 'px'; }};
                    textarea.addEventListener('input', adjustHeight);
                    textarea.setAttribute('data-auto-expand-setup', 'true');
                    setTimeout(adjustHeight, 100);
                }}
            }}
            setTimeout(setupAutoExpand, 200);
        </script>
    """, unsafe_allow_html=True)

    with st.form("sentiment_form"):
        user_text = st.text_area("Enter review text here:", "The battery life of this phone is amazing, I'm so happy with my purchase!")
        submitted = st.form_submit_button("Compare Analysis")

    if submitted and user_text.strip():
        with st.spinner("Analyzing text..."):
            results = analyze_sentiment(user_text, models, emotion_classifier)

        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model 1: Textual Features")
            display_sentiment_result(**results["model1"])
            st.markdown("###### Sentiment Probability Comparison")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.markdown("<p style='text-align: center;'>Without Emotion</p>", unsafe_allow_html=True)
                create_sentiment_chart(results["model1"]["df"])
            with prob_col2:
                st.markdown("<p style='text-align: center;'>With Emotion</p>", unsafe_allow_html=True)
                create_sentiment_chart(results["model2"]["df"])
            st.markdown("###### Interpretation of Results")
            st.info(results["comparison"]["text"])

        with col2:
            st.markdown("#### Model 2: Textual + Emotion Features")
            display_sentiment_result(**results["model2"])
            if not results["model2"]["is_uncertain"]:
                st.metric(
                    label=f"Confidence Shift for '{results['model2']['prediction'].capitalize()}'",
                    value=f"{results['comparison']['delta']:+.2%}",
                    help="How much the confidence changed after adding emotion features."
                )
            st.markdown("###### Emotion Analysis (Input Feature)")
            emotion_map = {'sadness': '😢', 'joy': '😂', 'anger': '😠', 'fear': '😨', 'surprise': '😮', 'disgust': '🤢', 'neutral': '😐'}
            color_map = {'sadness': '#3b82f6', 'joy': '#facc15', 'anger': '#ef4444', 'fear': '#a855f7', 'surprise': '#22d3ee', 'disgust': '#84cc16', 'neutral': '#a1a1aa'}
            top_emotion = results["emotion"]["top"]
            sub_col1, sub_col2 = st.columns([1, 3])
            with sub_col1:
                st.markdown(f"<div style='text-align: center;'><p style='font-size: 3rem; margin-bottom: 0;'>{emotion_map.get(top_emotion,'❓')}</p><p style='font-weight: bold;'>{top_emotion.capitalize()}</p></div>", unsafe_allow_html=True)
            with sub_col2:
                fig_emotion = go.Figure()
                for _, row in results["emotion"]["df"].sort_values('Score', ascending=True).iterrows():
                    fig_emotion.add_trace(go.Bar(y=[row['Emotion'].capitalize()], x=[row['Score']], name=row['Emotion'].capitalize(), orientation='h', marker_color=color_map.get(row['Emotion'], '#888')))
                fig_emotion.update_layout(showlegend=False, height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showgrid=False, title="Score (%)"), yaxis=dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#fff"))
                st.plotly_chart(fig_emotion, use_container_width=True, config={'displayModeBar': False})
    elif submitted:
        st.warning("Please enter some text to analyze.")

else:
    st.error("Application could not start. Please check the model files and internet connection.")

