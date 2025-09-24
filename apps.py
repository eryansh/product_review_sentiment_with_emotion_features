import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from transformers import pipeline

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="centered",
)

# --- Memuatkan Aset Pipeline Anda ---
# Guna st.cache_resource untuk memuatkan model sekali sahaja
@st.cache_resource
def load_your_pipeline():
    """Memuatkan komponen pipeline Naive Bayes yang telah anda latih."""
    try:
        # Memuatkan set model yang betul (_emo)
        tfidf_vectorizer = joblib.load('tfidf_vectorizer_emo.joblib')
        chi2_selector = joblib.load('chi2_selector_emo.joblib')
        naive_bayes_model = joblib.load('naive_bayes_model_emo.joblib')
        return tfidf_vectorizer, chi2_selector, naive_bayes_model
    except FileNotFoundError:
        # Mesej ralat dikemaskini
        st.error("Ralat: Pastikan fail 'tfidf_vectorizer_emo.joblib', 'chi2_selector_emo.joblib', dan 'naive_bayes_model_emo.joblib' berada di direktori utama repositori anda.")
        return None, None, None
    except Exception as e:
        st.error(f"Ralat semasa memuatkan model anda: {e}")
        return None, None, None

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
st.title("🤖 Analisis Sentimen Ulasan Produk")
st.markdown("Masukkan teks ulasan produk, dan aplikasi ini akan menganalisis emosi serta meramalkan sentimen (Positif/Negatif/Neutral) menggunakan model Naive Bayes yang telah dilatih.")

# Memuatkan semua model yang diperlukan
with st.spinner("Memuatkan model AI, sila tunggu..."):
    tfidf, selector, nb_model = load_your_pipeline()
    emotion_classifier = load_emotion_model()

if tfidf and selector and nb_model and emotion_classifier:
    with st.form("sentiment_form"):
        user_text = st.text_area("Masukkan teks ulasan di sini:", "The battery life of this phone is amazing, I'm so happy with my purchase!", height=150)
        submitted = st.form_submit_button("Analisis Teks")

    if submitted and user_text:
        st.divider()
        st.subheader("Keputusan Analisis")

        with st.spinner("Menganalisis teks..."):
            # === LANGKAH 1: Dapatkan Ciri Emosi dari model Hugging Face ===
            emotion_scores_raw = emotion_classifier(user_text)[0]
            
            # Susun skor mengikut urutan yang sama seperti semasa latihan
            emotion_labels_ordered = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
            emotion_features = np.array([scores_dict[label] for label in emotion_labels_ordered]).reshape(1, -1)

            # === LANGKAH 2: Proses Ciri Teks menggunakan pipeline anda ===
            text_tfidf = tfidf.transform([user_text])
            text_chi2 = selector.transform(text_tfidf)

            # === LANGKAH 3: Gabungkan Ciri Teks dan Emosi ===
            final_features = hstack([text_chi2, emotion_features])

            # === LANGKAH 4: Buat Ramalan Sentimen ===
            prediction_proba = nb_model.predict_proba(final_features)
            
            # Kira keyakinan dengan cara yang lebih selamat
            confidence = np.max(prediction_proba)
            # Dapatkan label sebenar yang diramalkan dari model
            predicted_class_index = np.argmax(prediction_proba)
            predicted_label = nb_model.classes_[predicted_class_index]

            # DIBETULKAN: Gunakan label string sebenar dari model untuk logik paparan
            # Model anda mengembalikan label seperti "Positive", "Negative", "Neutral"
            predicted_label_str = str(predicted_label)

            if predicted_label_str.lower() == 'positive':
                sentiment_label_display = "Positif"
                st.success(f"**Sentimen Diramalkan:** {sentiment_label_display} (Keyakinan: {confidence:.2%})")
            elif predicted_label_str.lower() == 'negative':
                sentiment_label_display = "Negatif"
                st.error(f"**Sentimen Diramalkan:** {sentiment_label_display} (Keyakinan: {confidence:.2%})")
            elif predicted_label_str.lower() == 'neutral':
                sentiment_label_display = "Neutral"
                st.info(f"**Sentimen Diramalkan:** {sentiment_label_display} (Keyakinan: {confidence:.2%})")
            else:
                st.warning(f"Gagal meramal sentimen. Label tidak dikenali: {predicted_label_str}")


            # Paparkan analisis emosi yang digunakan sebagai ciri
            with st.expander("Lihat Analisis Emosi Terperinci"):
                df_scores = pd.DataFrame(emotion_scores_raw)
                df_scores.rename(columns={'label': 'Emosi', 'score': 'Skor Keyakinan'}, inplace=True)
                st.bar_chart(df_scores.set_index('Emosi'))
                st.dataframe(df_scores, use_container_width=True)

    elif submitted and not user_text:
        st.warning("Sila masukkan teks untuk dianalisis.")
else:
    st.error("Aplikasi tidak dapat dimulakan kerana model gagal dimuatkan. Sila semak fail model anda dan sambungan internet.")

