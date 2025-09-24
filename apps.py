import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from transformers import pipeline

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sentiment Analysis Comparison",
    page_icon="🤖",
    layout="wide", # Diubah kepada 'wide' untuk paparan perbandingan yang lebih baik
)

# --- Memuatkan Aset Pipeline Anda ---
@st.cache_resource
def load_all_models():
    """Memuatkan kedua-dua set pipeline: dengan dan tanpa ciri emosi."""
    try:
        # Muatkan model TANPA emosi
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        chi2_selector = joblib.load('chi2_selector.joblib')
        naive_bayes_model = joblib.load('naive_bayes_model.joblib')

        # Muatkan model DENGAN emosi
        tfidf_vectorizer_emo = joblib.load('tfidf_vectorizer_emo.joblib')
        chi2_selector_emo = joblib.load('chi2_selector_emo.joblib')
        naive_bayes_model_emo = joblib.load('naive_bayes_model_emo.joblib')
        
        models = {
            "without_emotion": (tfidf_vectorizer, chi2_selector, naive_bayes_model),
            "with_emotion": (tfidf_vectorizer_emo, chi2_selector_emo, naive_bayes_model_emo)
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Ralat: Gagal mencari salah satu fail model. Pastikan semua 6 fail .joblib berada di direktori utama. Ralat: {e}")
        return None
    except Exception as e:
        st.error(f"Ralat semasa memuatkan model anda: {e}")
        return None

# --- Memuatkan Model Emosi (untuk Penjanaan Ciri) ---
@st.cache_resource
def load_emotion_model():
    """Memuatkan model pengesan emosi dari Hugging Face."""
    try:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        return emotion_classifier
    except Exception as e:
        st.error(f"Ralat memuatkan model emosi: {e}")
        return None

# --- UI dan Logik ---
st.title("🤖 Perbandingan Analisis Sentimen")
st.markdown("Bandingkan ramalan sentimen daripada dua model: satu menggunakan teks sahaja, dan satu lagi diperkaya dengan ciri-ciri emosi.")

# Memuatkan semua model yang diperlukan
with st.spinner("Memuatkan model AI, sila tunggu..."):
    models = load_all_models()
    emotion_classifier = load_emotion_model()

if models and emotion_classifier:
    with st.form("sentiment_form"):
        user_text = st.text_area("Masukkan teks ulasan di sini:", "The battery life of this phone is amazing, I'm so happy with my purchase!", height=100)
        submitted = st.form_submit_button("Bandingkan Analisis")

    if submitted and user_text:
        with st.spinner("Menganalisis teks..."):
            
            col1, col2 = st.columns(2)

            # --- PREDIKSI TANPA EMOSI (KOLUM 1) ---
            with col1:
                st.markdown("#### Model 1: Tanpa Ciri Emosi")
                tfidf, selector, nb_model = models["without_emotion"]
                
                text_tfidf = tfidf.transform([user_text])
                text_chi2 = selector.transform(text_tfidf)
                
                prediction_proba = nb_model.predict_proba(text_chi2)
                confidence = np.max(prediction_proba)
                predicted_label = nb_model.classes_[np.argmax(prediction_proba)]
                
                if str(predicted_label).lower() == 'positive':
                    st.success(f"**Positif** (Keyakinan: {confidence:.2%})")
                elif str(predicted_label).lower() == 'negative':
                    st.error(f"**Negatif** (Keyakinan: {confidence:.2%})")
                else:
                    st.info(f"**Neutral** (Keyakinan: {confidence:.2%})")
                
                st.markdown("###### Pecahan Sentimen")
                df_proba = pd.DataFrame({'Sentimen': nb_model.classes_, 'Kebarangkalian': prediction_proba[0]})
                sentiment_map = {'Positive': 'Positif', 'Negative': 'Negatif', 'Neutral': 'Neutral'}
                df_proba['Sentimen'] = df_proba['Sentimen'].map(sentiment_map).fillna(df_proba['Sentimen'])
                st.bar_chart(df_proba.set_index('Sentimen'), height=200)

            # --- PREDIKSI DENGAN EMOSI (KOLUM 2) ---
            with col2:
                st.markdown("#### Model 2: Dengan Ciri Emosi")
                tfidf_emo, selector_emo, nb_model_emo = models["with_emotion"]
                
                emotion_scores_raw = emotion_classifier(user_text)[0]
                labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
                scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
                emotion_features = np.array([scores_dict[l] for l in labels]).reshape(1, -1)

                text_tfidf_emo = tfidf_emo.transform([user_text])
                text_chi2_emo = selector_emo.transform(text_tfidf_emo)
                final_features = hstack([text_chi2_emo, emotion_features])

                prediction_proba_emo = nb_model_emo.predict_proba(final_features)
                confidence_emo = np.max(prediction_proba_emo)
                predicted_label_emo = nb_model_emo.classes_[np.argmax(prediction_proba_emo)]

                if str(predicted_label_emo).lower() == 'positive':
                    st.success(f"**Positif** (Keyakinan: {confidence_emo:.2%})")
                elif str(predicted_label_emo).lower() == 'negative':
                    st.error(f"**Negatif** (Keyakinan: {confidence_emo:.2%})")
                else:
                    st.info(f"**Neutral** (Keyakinan: {confidence_emo:.2%})")

                # DIUBAH SUAI: Cipta kolum bersarang untuk carta
                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
                    st.markdown("###### Pecahan Sentimen")
                    df_proba_emo = pd.DataFrame({'Sentimen': nb_model_emo.classes_, 'Kebarangkalian': prediction_proba_emo[0]})
                    df_proba_emo['Sentimen'] = df_proba_emo['Sentimen'].map(sentiment_map).fillna(df_proba_emo['Sentimen'])
                    st.bar_chart(df_proba_emo.set_index('Sentimen'), height=200)
                
                with sub_col2:
                    st.markdown("###### Analisis Emosi (Input)")
                    df_scores = pd.DataFrame(emotion_scores_raw)
                    df_scores.rename(columns={'label': 'Emosi', 'score': 'Skor'}, inplace=True)
                    st.bar_chart(df_scores.set_index('Emosi'), height=200)


    elif submitted and not user_text:
        st.warning("Sila masukkan teks untuk dianalisis.")
else:
    st.error("Aplikasi tidak dapat dimulakan kerana model gagal dimuatkan. Sila semak fail model anda dan sambungan internet.")

