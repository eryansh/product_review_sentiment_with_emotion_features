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
                # DIUBAH SUAI: Menggunakan Plotly untuk carta sentimen
                df_proba = pd.DataFrame({'Sentimen': nb_model.classes_, 'Kebarangkalian': prediction_proba[0]})
                df_proba['Kebarangkalian'] = df_proba['Kebarangkalian'] * 100 # Tukar kepada peratusan

                sentiment_color_map = {
                    'Positive': '#22c55e', # Hijau
                    'Negative': '#ef4444', # Merah
                    'Neutral': '#a1a1aa'  # Kelabu
                }

                fig_sentiment = go.Figure()
                for index, row in df_proba.sort_values('Kebarangkalian', ascending=True).iterrows():
                    sentiment = row['Sentimen']
                    fig_sentiment.add_trace(go.Bar(
                        y=[sentiment.capitalize()],
                        x=[row['Kebarangkalian']],
                        name=sentiment.capitalize(),
                        orientation='h',
                        marker_color=sentiment_color_map.get(sentiment, '#888')
                    ))

                fig_sentiment.update_layout(
                    showlegend=False,
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(range=[0, 100], showgrid=False, title="Kebarangkalian (%)"),
                    yaxis=dict(showgrid=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#fff")
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)


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
                
                # DIUBAH SUAI: Visualisasi Emosi Gaya Baharu
                st.markdown("###### Analisis Emosi (Ciri Input)")
                
                # Peta Emosi ke Emoji dan Warna
                emotion_map = {
                    'sadness': {'emoji': '😢', 'color': '#3b82f6'},
                    'joy': {'emoji': '😂', 'color': '#facc15'},
                    'anger': {'emoji': '😠', 'color': '#ef4444'},
                    'fear': {'emoji': '😨', 'color': '#a855f7'},
                    'surprise': {'emoji': '😮', 'color': '#22d3ee'},
                    'disgust': {'emoji': '🤢', 'color': '#84cc16'},
                    'neutral': {'emoji': '😐', 'color': '#a1a1aa'}
                }

                df_scores = pd.DataFrame(emotion_scores_raw)
                df_scores.rename(columns={'label': 'Emotion', 'score': 'Score'}, inplace=True)
                df_scores['Score'] = df_scores['Score'] * 100 # Tukar kepada peratusan
                top_emotion_data = df_scores.loc[df_scores['Score'].idxmax()]
                top_emotion = top_emotion_data['Emotion']

                sub_col1, sub_col2 = st.columns([1, 2])

                with sub_col1:
                    st.markdown(f"<div style='text-align: center;'><p style='font-size: 4rem; margin-bottom: 0;'>{emotion_map.get(top_emotion,{}).get('emoji','❓')}</p><p style='font-weight: bold;'>{top_emotion.capitalize()}</p></div>", unsafe_allow_html=True)

                with sub_col2:
                    # Cipta carta bar mendatar dengan Plotly
                    fig = go.Figure()
                    for index, row in df_scores.sort_values('Score', ascending=True).iterrows():
                        emotion = row['Emotion']
                        fig.add_trace(go.Bar(
                            y=[emotion.capitalize()],
                            x=[row['Score']],
                            name=emotion.capitalize(),
                            orientation='h',
                            marker_color=emotion_map.get(emotion, {}).get('color', '#888')
                        ))
                    
                    fig.update_layout(
                        barmode='stack',
                        showlegend=False,
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(range=[0, 100], showgrid=False, title="Skor (%)"),
                        yaxis=dict(showgrid=False),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#fff")
                    )
                    st.plotly_chart(fig, use_container_width=True)


    elif submitted and not user_text:
        st.warning("Sila masukkan teks untuk dianalisis.")
else:
    st.error("Aplikasi tidak dapat dimulakan kerana model gagal dimuatkan. Sila semak fail model anda dan sambungan internet.")

