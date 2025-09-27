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
    layout="wide", # Changed to 'wide' for a better comparison view
)

# --- Loading Your Pipeline Assets ---
@st.cache_resource
def load_all_models():
    """Loads both sets of pipelines: with and without emotion features."""
    try:
        # Load model WITHOUT emotion
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        chi2_selector = joblib.load('chi2_selector.joblib')
        naive_bayes_model = joblib.load('naive_bayes_model.joblib')

        # Load model WITH emotion
        tfidf_vectorizer_emo = joblib.load('tfidf_vectorizer_emo.joblib')
        chi2_selector_emo = joblib.load('chi2_selector_emo.joblib')
        naive_bayes_model_emo = joblib.load('naive_bayes_model_emo.joblib')
        
        models = {
            "without_emotion": (tfidf_vectorizer, chi2_selector, naive_bayes_model),
            "with_emotion": (tfidf_vectorizer_emo, chi2_selector_emo, naive_bayes_model_emo)
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Error: Failed to find one of the model files. Please ensure all 6 .joblib files are in the main directory. Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error while loading your models: {e}")
        return None

# --- Loading Emotion Model (for Feature Generation) ---
@st.cache_resource
def load_emotion_model():
    """Loads the emotion detection model from Hugging Face."""
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        return emotion_classifier
    except Exception as e:
        st.error(f"Error loading the emotion model: {e}")
        return None

# --- UI and Logic ---

# --- Video Background ---
# The video URL is now hardcoded and the sidebar option is removed.
video_url = "https://raw.githubusercontent.com/eryansh/product_review_sentiment_with_emotion_features/main/background.mp4"


if video_url:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        #bg-video {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            z-index: -1;
        }}
        </style>
        <video id="bg-video" autoplay loop muted>
            <source src="{video_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )


st.title("🤖 Sentiment Analysis Comparison")
st.markdown("Compare sentiment predictions from two models: one using text only, and another enriched with emotion features.")

# Loading all necessary models
with st.spinner("Loading AI models, please wait..."):
    models = load_all_models()
    emotion_classifier = load_emotion_model()

if models and emotion_classifier:
    # Initialize session state to manage the app's flow
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
        st.session_state.user_text = "The battery life of this phone is amazing, I'm so happy with my purchase!"

    # Callback functions to update the state
    def handle_submit():
        if st.session_state.text_input:
            st.session_state.user_text = st.session_state.text_input
            st.session_state.submitted = True
        else:
            st.warning("Please enter some text to analyze.")

    def handle_reset():
        st.session_state.submitted = False
        st.session_state.text_input = st.session_state.user_text # Keep last text

    # Add custom JS and CSS for the auto-expanding textarea
    st.markdown("""
        <style>
            textarea[aria-label="Enter review text here:"] {
                resize: none;
                overflow-y: hidden;
            }
        </style>
        <script>
            function setupAutoExpand() {
                const textarea = document.querySelector('textarea[aria-label="Enter review text here:"]');
                if (textarea && !textarea.hasAttribute('data-auto-expand-setup')) {
                    const adjustHeight = () => {
                        textarea.style.height = 'auto';
                        textarea.style.height = (textarea.scrollHeight) + 'px';
                    };
                    textarea.addEventListener('input', adjustHeight);
                    textarea.setAttribute('data-auto-expand-setup', 'true');
                    setTimeout(adjustHeight, 100);
                }
            }
            setTimeout(setupAutoExpand, 200);
        </script>
    """, unsafe_allow_html=True)

    if not st.session_state.submitted:
        # Show the input form
        with st.form("sentiment_form"):
            st.text_area("Enter review text here:", key="text_input", value=st.session_state.user_text)
            st.form_submit_button("Compare Analysis", on_click=handle_submit)
    else:
        # Show the results
        user_text = st.session_state.user_text
        with st.spinner("Analyzing text..."):
            # --- PERFORM ALL CALCULATIONS FIRST ---
            # Model 1
            tfidf, selector, nb_model = models["without_emotion"]
            text_tfidf = tfidf.transform([user_text])
            text_chi2 = selector.transform(text_tfidf)
            prediction_proba = nb_model.predict_proba(text_chi2)
            confidence = np.max(prediction_proba)
            predicted_label = nb_model.classes_[np.argmax(prediction_proba)]
            is_uncertain1 = np.isclose(confidence, 1/3)

            # Model 2
            tfidf_emo, selector_emo, nb_model_emo = models["with_emotion"]
            truncated_text = user_text[:512]
            emotion_scores_raw = emotion_classifier(truncated_text)[0]
            labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
            emotion_features = np.array([scores_dict[l] for l in labels]).reshape(1, -1)
            text_tfidf_emo = tfidf_emo.transform([user_text])
            text_chi2_emo = selector_emo.transform(text_tfidf_emo)
            final_features = hstack([text_chi2_emo, emotion_features])
            prediction_proba_emo = nb_model_emo.predict_proba(final_features)
            confidence_emo = np.max(prediction_proba_emo)
            predicted_class_index_emo = np.argmax(prediction_proba_emo)
            predicted_label_emo = nb_model_emo.classes_[predicted_class_index_emo]
            is_uncertain2 = np.isclose(confidence_emo, 1/3)
            
            confidence_from_model1 = prediction_proba[0][predicted_class_index_emo]
            confidence_delta = confidence_emo - confidence_from_model1
            df_scores = pd.DataFrame(emotion_scores_raw)
            df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
            df_scores['Score'] = df_scores['Score'] * 100
            top_emotion_data = df_scores.loc[df_scores['Score'].idxmax()]
            top_emotion = top_emotion_data['Emotion']
            
            # Build interpretation text
            interpretation_text = ""
            if is_uncertain1 or is_uncertain2:
                interpretation_text = "The model is **uncertain** because the input text is too short or contains words not in its vocabulary. Please provide a more complete sentence."
            elif predicted_label.lower() != predicted_label_emo.lower():
                interpretation_text += f"These models **disagree**. Model 1 predicts **{predicted_label.capitalize()}**, while Model 2 predicts **{predicted_label_emo.capitalize()}**. "
            else:
                interpretation_text += f"Both models **agree** that the sentiment is **{predicted_label.capitalize()}**. "
            
            if not (is_uncertain1 or is_uncertain2):
                if top_emotion not in ['neutral']:
                    interpretation_text += f"The detection of strong **{top_emotion.capitalize()}** emotion likely influenced Model 2, leading to higher confidence and a more nuanced prediction."
                else:
                    interpretation_text += f"This text was detected as emotionally **Neutral**. This helps Model 2 reduce any bias and produce a more balanced sentiment prediction."

            # --- DISPLAY RESULTS IN COLUMNS ---
            st.markdown(f"> **Original Text:** *{user_text}*")
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Model 1: Without Emotion Features")
                if is_uncertain1: st.warning("Model is uncertain due to unrecognized input.")
                elif str(predicted_label).lower() == 'positive': st.success(f"**Positive** (Confidence: {confidence:.2%})")
                elif str(predicted_label).lower() == 'negative': st.error(f"**Negative** (Confidence: {confidence:.2%})")
                else: st.info(f"**Neutral** (Confidence: {confidence:.2%})")
                st.markdown("###### Sentiment Probability Comparison")
                prob_col1, prob_col2 = st.columns(2)
                sentiment_color_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#a1a1aa'}
                with prob_col1:
                    st.markdown("<p style='text-align: center;'>Without Emotion</p>", unsafe_allow_html=True)
                    df_proba = pd.DataFrame({'Sentiment': nb_model.classes_, 'Probability': prediction_proba[0] * 100})
                    sentiment_order = ['Negative', 'Neutral', 'Positive']
                    df_proba = df_proba.set_index('Sentiment').reindex(sentiment_order).reset_index()
                    fig_sentiment1 = go.Figure()
                    for _, row in df_proba.iterrows():
                        fig_sentiment1.add_trace(go.Bar(y=[row['Sentiment'].capitalize()], x=[row['Probability']], name=row['Sentiment'].capitalize(), orientation='h', marker_color=sentiment_color_map.get(row['Sentiment'], '#888')))
                    fig_sentiment1.update_layout(showlegend=False, height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showgrid=False), yaxis=dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#fff"))
                    st.plotly_chart(fig_sentiment1, use_container_width=True, config={'displayModeBar': False})
                with prob_col2:
                    st.markdown("<p style='text-align: center;'>With Emotion</p>", unsafe_allow_html=True)
                    df_proba_emo = pd.DataFrame({'Sentiment': nb_model_emo.classes_, 'Probability': prediction_proba_emo[0] * 100})
                    df_proba_emo = df_proba_emo.set_index('Sentiment').reindex(sentiment_order).reset_index()
                    fig_sentiment2 = go.Figure()
                    for _, row in df_proba_emo.iterrows():
                        fig_sentiment2.add_trace(go.Bar(y=[row['Sentiment'].capitalize()], x=[row['Probability']], name=row['Sentiment'].capitalize(), orientation='h', marker_color=sentiment_color_map.get(row['Sentiment'], '#888')))
                    fig_sentiment2.update_layout(showlegend=False, height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showgrid=False), yaxis=dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#fff"))
                    st.plotly_chart(fig_sentiment2, use_container_width=True, config={'displayModeBar': False})
                st.markdown("###### Interpretation of Results")
                st.info(interpretation_text)

            with col2:
                st.markdown("#### Model 2: With Emotion Features")
                if is_uncertain2: st.warning("Model is uncertain due to unrecognized input.")
                elif str(predicted_label_emo).lower() == 'positive': st.success(f"**Positive** (Confidence: {confidence_emo:.2%})")
                elif str(predicted_label_emo).lower() == 'negative': st.error(f"**Negative** (Confidence: {confidence_emo:.2%})")
                else: st.info(f"**Neutral** (Confidence: {confidence_emo:.2%})")
                if not is_uncertain2:
                    st.metric(label=f"Confidence Shift for '{predicted_label_emo.capitalize()}'", value=f"{confidence_delta:+.2%}", help=f"The change in confidence for the '{predicted_label_emo.capitalize()}' sentiment after adding emotion features.")
                st.markdown("###### Emotion Analysis (Input Feature)")
                emotion_map = {'sadness': {'emoji': '😢', 'color': '#3b82f6'}, 'joy': {'emoji': '😂', 'color': '#facc15'}, 'anger': {'emoji': '😠', 'color': '#ef4444'}, 'fear': {'emoji': '😨', 'color': '#a855f7'}, 'surprise': {'emoji': '😮', 'color': '#22d3ee'}, 'disgust': {'emoji': '🤢', 'color': '#84cc16'}, 'neutral': {'emoji': '😐', 'color': '#a1a1aa'}}
                sub_col1, sub_col2 = st.columns([1, 3])
                with sub_col1:
                    st.markdown(f"<div style='text-align: center;'><p style='font-size: 3rem; margin-bottom: 0;'>{emotion_map.get(top_emotion,{}).get('emoji','❓')}</p><p style='font-weight: bold;'>{top_emotion.capitalize()}</p></div>", unsafe_allow_html=True)
                with sub_col2:
                    fig_emotion = go.Figure()
                    for _, row in df_scores.sort_values('Score', ascending=True).iterrows():
                        fig_emotion.add_trace(go.Bar(y=[row['Emotion'].capitalize()], x=[row['Score']], name=row['Emotion'].capitalize(), orientation='h', marker_color=emotion_map.get(row['Emotion'], {}).get('color', '#888')))
                    fig_emotion.update_layout(showlegend=False, height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showgrid=False, title="Score (%)"), yaxis=dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#fff"))
                    st.plotly_chart(fig_emotion, use_container_width=True, config={'displayModeBar': False})
        
        st.button("Analyze Another Review", on_click=handle_reset)

else:
    st.error("The application could not start because the models failed to load. Please check your model files and internet connection.")

