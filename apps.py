import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go

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
            
            # --- LAKUKAN SEMUA PENGIRAAN DAHULU ---
            # Model 1
            tfidf, selector, nb_model = models["without_emotion"]
            text_tfidf = tfidf.transform([user_text])
            text_chi2 = selector.transform(text_tfidf)
            prediction_proba = nb_model.predict_proba(text_chi2)
            confidence = np.max(prediction_proba)
            predicted_label = nb_model.classes_[np.argmax(prediction_proba)]

            # Model 2
            tfidf_emo, selector_emo, nb_model_emo = models["with_emotion"]
            emotion_scores_raw = emotion_classifier(user_text)[0]
            labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            scores_dict = {item['label']: item['score'] for item in emotion_scores_raw}
            emotion_features = np.array([scores_dict[l] for l in labels]).reshape(1, --1)
            text_tfidf_emo = tfidf_emo.transform([user_text])
            text_chi2_emo = selector_emo.transform(text_tfidf_emo)
            final_features = hstack([text_chi2_emo, emotion_features])
            prediction_proba_emo = nb_model_emo.predict_proba(final_features)
            confidence_emo = np.max(prediction_proba_emo)
            predicted_class_index_emo = np.argmax(prediction_proba_emo)
            predicted_label_emo = nb_model_emo.classes_[predicted_class_index_emo]
            confidence_from_model1 = prediction_proba[0][predicted_class_index_emo]
            confidence_delta = confidence_emo - confidence_from_model1
            df_scores = pd.DataFrame(emotion_scores_raw)
            df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
            df_scores['Score'] = df_scores['Score'] * 100
            top_emotion_data = df_scores.loc[df_scores['Score'].idxmax()]
            top_emotion = top_emotion_data['Emotion']
            
            # Bina teks interpretasi
            interpretation_text = ""
            if predicted_label.lower() != predicted_label_emo.lower():
                interpretation_text += f"Model-model ini **tidak bersetuju**. Model 1 meramalkan **{predicted_label.capitalize()}**, manakala Model 2 meramalkan **{predicted_label_emo.capitalize()}**. "
            else:
                interpretation_text += f"Kedua-dua model **bersetuju** bahawa sentimennya adalah **{predicted_label.capitalize()}**. "
            if top_emotion not in ['neutral']:
                interpretation_text += f"Pengesanan emosi **{top_emotion.capitalize()}** yang kuat berkemungkinan besar mempengaruhi Model 2, memberikannya keyakinan yang lebih tinggi dan ramalan yang lebih jitu."
            else:
                interpretation_text += f"Teks ini dikesan sebagai **Neutral** dari segi emosi. Ini membantu Model 2 untuk mengurangkan sebarang kecenderungan (bias) dan menghasilkan ramalan sentimen yang lebih seimbang."

            # --- PAPARKAN KEPUTUSAN DALAM KOLUM ---
            col1, col2 = st.columns(2)

            # --- KOLUM 1: Model Tanpa Emosi + Interpretasi ---
            with col1:
                st.markdown("#### Model 1: Tanpa Ciri Emosi")
                
                if str(predicted_label).lower() == 'positive':
                    st.success(f"**Positif** (Keyakinan: {confidence:.2%})")
                elif str(predicted_label).lower() == 'negative':
                    st.error(f"**Negatif** (Keyakinan: {confidence:.2%})")
                else:
                    st.info(f"**Neutral** (Keyakinan: {confidence:.2%})")
                
                # DIUBAH SUAI: Gantikan carta dengan interpretasi
                st.markdown("###### Interpretasi Keputusan")
                st.info(interpretation_text)

            # --- KOLUM 2: Model Dengan Emosi ---
            with col2:
                st.markdown("#### Model 2: Dengan Ciri Emosi")
                
                if str(predicted_label_emo).lower() == 'positive':
                    st.success(f"**Positif** (Keyakinan: {confidence_emo:.2%})")
                elif str(predicted_label_emo).lower() == 'negative':
                    st.error(f"**Negatif** (Keyakinan: {confidence_emo:.2%})")
                else:
                    st.info(f"**Neutral** (Keyakinan: {confidence_emo:.2%})")
                
                st.metric(
                    label="Peningkatan Keyakinan",
                    value=f"{confidence_emo:.2%}",
                    delta=f"{confidence_delta:.2%}",
                    help="Perbezaan keyakinan untuk sentimen ini berbanding model tanpa ciri emosi."
                )
                
                st.markdown("###### Analisis Emosi (Ciri Input)")
                
                emotion_map = {
                    'sadness': {'emoji': '😢', 'color': '#3b82f6'},
                    'joy': {'emoji': '😂', 'color': '#facc15'},
                    'anger': {'emoji': '😠', 'color': '#ef4444'},
                    'fear': {'emoji': '😨', 'color': '#a855f7'},
                    'surprise': {'emoji': '😮', 'color': '#22d3ee'},
                    'disgust': {'emoji': '🤢', 'color': '#84cc16'},
                    'neutral': {'emoji': '😐', 'color': '#a1a1aa'}
                }

                sub_col1, sub_col2 = st.columns([1, 3])

                with sub_col1:
                    st.markdown(f"<div style='text-align: center;'><p style='font-size: 3rem; margin-bottom: 0;'>{emotion_map.get(top_emotion,{}).get('emoji','❓')}</p><p style='font-weight: bold;'>{top_emotion.capitalize()}</p></div>", unsafe_allow_html=True)

                with sub_col2:
                    fig_emotion = go.Figure()
                    for index, row in df_scores.sort_values('Score', ascending=True).iterrows():
                        emotion = row['Emotion']
                        fig_emotion.add_trace(go.Bar(
                            y=[emotion.capitalize()],
                            x=[row['Score']],
                            name=emotion.capitalize(),
                            orientation='h',
                            marker_color=emotion_map.get(emotion, {}).get('color', '#888')
                        ))
                    
                    fig_emotion.update_layout(
                        showlegend=False,
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(range=[0, 100], showgrid=False, title="Skor (%)"),
                        yaxis=dict(showgrid=False),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#fff")
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)

    elif submitted and not user_text:
        st.warning("Sila masukkan teks untuk dianalisis.")
else:
    st.error("Aplikasi tidak dapat dimulakan kerana model gagal dimuatkan. Sila semak fail model anda dan sambungan internet.")

