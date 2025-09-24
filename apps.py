import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Emotion Detector App",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once and cache it for subsequent runs.
# This hugely improves performance.
@st.cache_resource
def load_emotion_model():
    """Loads the emotion detection model from Hugging Face."""
    try:
        # Using the specified model for emotion classification
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True # Set to True to get scores for all emotions
        )
        return emotion_classifier
    except Exception as e:
        # If the model fails to load, display an error message.
        st.error(f"Error loading the model: {e}")
        return None

# --- UI Layout and Logic ---

# Title and Subheader
st.title("😊 Emotion Detector")
st.markdown("Enter any text below, and the app will analyze the underlying emotion using a pre-trained AI model.")

# Load the model and show a spinner while it's loading.
with st.spinner("Loading AI model, please wait..."):
    classifier = load_emotion_model()

if classifier:
    # Create a form for user input to prevent the app from rerunning on every key press
    with st.form("emotion_form"):
        # Text area for user input
        user_text = st.text_area("Enter your text here:", "I am feeling incredibly happy and excited about this new project!", height=150)
        # Submit button for the form
        submitted = st.form_submit_button("Analyze Emotion")

    # --- Processing and Displaying Results ---
    if submitted and user_text:
        st.divider()
        st.subheader("Analysis Results")

        # Show a spinner while the model is processing the text
        with st.spinner("Analyzing text..."):
            # Get predictions from the model
            # The model returns a list containing a dictionary of emotions and their scores
            prediction = classifier(user_text)

            if prediction:
                # Extract the list of scores from the nested structure
                scores = prediction[0]

                # Convert the list of dictionaries to a pandas DataFrame for easier handling
                df_scores = pd.DataFrame(scores)
                # Rename columns for clarity
                df_scores.rename(columns={'label': 'Emotion', 'score': 'Confidence'}, inplace=True)
                # Find the emotion with the highest confidence score
                top_emotion = df_scores.loc[df_scores['Confidence'].idxmax()]

                # Display the top emotion with a success message
                st.success(f"**Top Emotion Detected:** {top_emotion['Emotion'].capitalize()} (Confidence: {top_emotion['Confidence']:.2%})")

                # Display a bar chart of all emotion scores
                st.write("Full Emotion Analysis:")

                # Create a bar chart from the dataframe, setting the emotion as the index
                chart = st.bar_chart(df_scores.set_index('Emotion'))

                # Display the scores in a table as well
                st.write("Confidence Scores:")
                st.dataframe(df_scores, use_container_width=True)
            else:
                st.error("Could not analyze the text. Please try again.")

    elif submitted and not user_text:
        # Show a warning if the user clicks submit with no text
        st.warning("Please enter some text to analyze.")

else:
    # Message if the model could not be loaded
    st.error("The application could not start because the AI model failed to load. Please check your internet connection or try again later.")
