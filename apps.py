import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
import re  # <--- NEW IMPORT for text cleansing
import nltk # <--- NEW IMPORT for text processing
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Resource Downloads (Robust Version) ---
# This creates a local 'nltk_data' directory and forces NLTK to use it.
# This is the most reliable method for Streamlit Cloud.

# Get the directory of the current script
# Use a fallback for environments where __file__ isn't defined (like some notebooks)
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()

# Define the path for NLTK data
NLTK_DATA_DIR = os.path.join(APP_DIR, "nltk_data")

# Create the directory if it doesn't exist
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

# Add this path to NLTK's data path list
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Now, download the necessary packages to that specific directory
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR) # <--- ADD THIS LINE
# --- END NEW SECTION ---

# --- CONFIGURATION ---
CONFIG = {
    "model_paths": {
        "label_encoder": 'label_encoder.joblib', # <-- ADDED
        "without_emotion": {
            # Point to the single XGBoost pipeline
            "pipeline": 'xgb_model_condition1.joblib' 
        },
        "with_emotion": {
            # Point to the single XGBoost pipeline
            "pipeline": 'xgb_model_condition2.joblib'
        }
    },
    "emotion_labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "sentiment_order": ['Negative', 'Neutral', 'Positive'],
    "hugging_face_model": "j-hartmann/emotion-english-distilroberta-base",
    "sentiment_color_map": {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#a1a1aa'},
    "emotion_color_map": {'sadness': '#3b82f6', 'joy': '#facc15', 'anger': '#ef4444', 'fear': '#a855f7', 'surprise': '#22d3ee', 'disgust': '#84cc16', 'neutral': '#a1a1aa'}
}


# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Classification with Emotion Features",
    page_icon="🤖",
    layout="wide",
)

# --- Initialize Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []


# --- Asset Loading ---
@st.cache_resource
def load_all_models():
    """Loads all joblib model files."""
    try:
        # Load the new XGBoost pipelines and the LabelEncoder
        models = {
            "label_encoder": joblib.load(CONFIG["model_paths"]["label_encoder"]),
            "without_emotion": (
                joblib.load(CONFIG["model_paths"]["without_emotion"]["pipeline"])
            ),
            "with_emotion": (
                joblib.load(CONFIG["model_paths"]["with_emotion"]["pipeline"])
            )
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

# --- NEW PREPROCESSING FUNCTION ---
@st.cache_data  # Cache this computation
def preprocess_text(text):
    """
    Applies the full preprocessing pipeline:
    a. Cleansing (Lowercase, numbers, punctuation, HTML)
    b. Tokenization
    c. Stopword Removal
    d. Lemmatization
    """
    # --- ADD THIS BLOCK ---
    # This re-appends the path *inside* the cached function
    # to ensure NLTK can find the data.
    # NLTK_DATA_DIR is defined at the top of your script.
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.append(NLTK_DATA_DIR)
    # --- END NEW BLOCK ---

    # Initialize components
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # a. Data Cleansing
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.lower()                 # Lowercasing
    
    # b. Tokenization
    tokens = word_tokenize(text)
    
    processed_tokens = []
    for word in tokens:
        # c. Stopword Removal
        if word not in stop_words:
            # d. Lemmatization
            processed_tokens.append(lemmatizer.lemmatize(word))
            
    # Return a single string, as TfidfVectorizer expects this
    return ' '.join(processed_tokens)
# --- END NEW FUNCTION ---


# --- Analysis Logic ---
def analyze_sentiment(user_text, models, emotion_classifier):
    """
    Performs sentiment and emotion analysis and returns all calculated results.
    This function separates the calculation logic from the display logic.
    """
    
    # --- NEW: Get LabelEncoder ---
    le = models["label_encoder"]
    
    # --- Preprocessing Step ---
    # We preprocess the text first, as the pipelines were trained on preprocessed text
    processed_text = preprocess_text(user_text)
    
    # --- Model 1: Without Emotion (XGBoost Pipeline) ---
    # This single object contains TFIDF, Chi2, SMOTE, and XGB
    pipeline_cond1 = models["without_emotion"]
    
    # We call predict_proba on the single processed text string.
    # The pipeline handles all vectorization and selection internally.
    prediction_proba = pipeline_cond1.predict_proba([processed_text])
    
    # Get the predicted class *index* (e.g., 0, 1, or 2)
    predicted_index = np.argmax(prediction_proba)
    # Convert the index back to the string label (e.g., 'Positive')
    predicted_label = le.inverse_transform([predicted_index])[0]
    
    # --- Model 2: With Emotion (XGBoost Pipeline) ---
    # This single object contains ColumnTransformer, Chi2, SMOTE, and XGB
    pipeline_cond2 = models["with_emotion"]
    
    # --- Get emotion features (same as before) ---
    truncated_text = user_text[:512] # Truncate for RoBERTa model limit
    emotion_scores_raw = emotion_classifier(truncated_text)[0]
    
    scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
    emotion_features = np.array([scores_dict[l] for l in CONFIG["emotion_labels"]]).reshape(1, -1)
    
    # --- Create the DataFrame for the pipeline ---
    # This pipeline was trained on a DataFrame, so we must build one
    # that matches the training data structure.
    
    # Create a dictionary for the emotion features with the correct column names
    emotion_data = {f"prob_{label}": score for label, score in zip(CONFIG["emotion_labels"], emotion_features[0])}
    
    data_dict = {
        'final_preprocessed_text': [processed_text], 
        **emotion_data
    }
    input_df = pd.DataFrame(data_dict)

    # We call predict_proba on the DataFrame.
    # The pipeline handles all feature extraction and selection.
    prediction_proba_emo = pipeline_cond2.predict_proba(input_df)
    
    # Convert the index back to the string label
    predicted_index_emo = np.argmax(prediction_proba_emo)
    predicted_label_emo = le.inverse_transform([predicted_index_emo])[0]
    
    # --- DataFrames for Plotting ---
    # Use le.classes_ to get the correct order ('Negative', 'Neutral', 'Positive')
    df_proba = pd.DataFrame({'Sentiment': le.classes_, 'Probability': prediction_proba[0] * 100})
    df_proba = df_proba.set_index('Sentiment').reindex(CONFIG["sentiment_order"]).reset_index()

    df_proba_emo = pd.DataFrame({'Sentiment': le.classes_, 'Probability': prediction_proba_emo[0] * 100})
    df_proba_emo = df_proba_emo.set_index('Sentiment').reindex(CONFIG["sentiment_order"]).reset_index()

    df_scores = pd.DataFrame(emotion_scores_raw)
    df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
    df_scores['Score'] = df_scores['Score'] * 100
    top_emotion = df_scores.loc[df_scores['Score'].idxmax()]['Emotion']

    # --- Interpretation & Comparison ---
    confidence = np.max(prediction_proba)
    confidence_emo = np.max(prediction_proba_emo)
    is_uncertain1 = np.isclose(confidence, 1/3, atol=0.05)
    is_uncertain2 = np.isclose(confidence_emo, 1/3, atol=0.05)
    
    # Get the probability for the *same class* from the other model
    # Note: Use predicted_index_emo (the index)
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
        "processed_text": processed_text # <--- This is now the processed text
    }

# --- UI Helper Functions ---
def display_sentiment_result(prediction, confidence, is_uncertain, **kwargs):
    """Displays the formatted sentiment result."""
    if is_uncertain: st.warning("Model is uncertain due to unrecognized input.")
    elif str(prediction).lower() == 'positive': st.success(f"**Positive** (Confidence: {confidence:.2%})")
    elif str(prediction).lower() == 'negative': st.error(f"**Negative** (Confidence: {confidence:.2%})")
    else: st.info(f"**Neutral** (Confidence: {confidence:.2%})")

def create_bar_chart(df, y_col, x_col, color_map, height, show_x_title=False):
    """Creates a generic horizontal bar chart for sentiment or emotion."""
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
    @import url('httpsAF://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
    .main-title {
        font-family: 'tahoma', sans-serif;
        font-size: clamp(2.5rem, 8vw, 7rem); /* Responsive font size */
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

models = load_all_models()
emotion_classifier = load_emotion_model()

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

    with st.form("sentiment_form"):
        user_text = st.text_area("Enter review text here:", "The battery life of this phone is amazing, I'm so happy with my purchase!")
        submitted = st.form_submit_button("Predict Sentiment")

    if submitted and user_text.strip():
        with st.spinner("Analyzing text..."):
            results = analyze_sentiment(user_text, models, emotion_classifier)
        
        # --- Store results in history (most recent first) ---
        st.session_state.history.insert(0, {
            "text": user_text,
            "model1_pred": results["model1"]["prediction"],
            "model2_pred": results["model2"]["prediction"],
            "top_emotion": results["emotion"]["top"]
        })

        st.divider()
        
        # --- NEW SECTION TO DISPLAY PROCESSED TEXT ---
        with st.expander("Show Preprocessed Text (for XGBoost models)"):
            st.markdown("**Original Text:**")
            st.info(user_text)
            st.markdown("**Processed Text (Input for Model 1 & 2):**")
            # Display processed text, or a note if it's empty after processing
            if results["processed_text"].strip():
                st.success(results["processed_text"])
            else:
                st.warning("Text was empty after preprocessing (e.g., only contained stopwords or numbers).")
        # --- END NEW SECTION ---
        
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
            st.markdown("#### Model 2: Textual Features Enriched with Emotion Features")
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
                st.markdown(f"<div style='text-align: center;'><p style='font-size: 3rem; margin-bottom: 0;'>{emotion_map.get(top_emotion,'❓')}</p><p style='font-weight: bold;'>{top_emotion.capitalize()}</p></div>", unsafe_allow_html=True)
            with sub_col2:
                sorted_emotions = results["emotion"]["df"].sort_values('Score', ascending=True)
                create_bar_chart(sorted_emotions, 'Emotion', 'Score', CONFIG["emotion_color_map"], 220, show_x_title=True)
            
    elif submitted:
        st.warning("Please enter some text to analyze.")
    
    # --- ADDED HISTORY SECTION ---
    st.divider()
    st.markdown("## Analysis History")

    if not st.session_state.history:
        st.info("Your previous analyses in this session will appear here.")
    else:
        for i, entry in enumerate(st.session_state.history):
            # Use a unique key for each expander
            with st.expander(f"**{len(st.session_state.history) - i}.** {entry['text'][:70]}..."):
                st.markdown(f"**Input Text:** _{entry['text']}_")
                st.markdown(f"**Model 1 (Text Only Prediction):** `{entry['model1_pred']}`")
                st.markdown(f"**Model 2 (Text + Emotion Prediction):** `{entry['model2_pred']}`")
                st.markdown(f"**Detected Top Emotion:** `{entry['top_emotion'].capitalize()}`")


else:
    st.error("Application could not start. Please check the model files and internet connection.")

# --- CREDIT SECTION ---
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
