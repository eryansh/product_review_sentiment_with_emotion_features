# Import the streamlit library
import streamlit as st

# Add a title to your app
st.title("My First Streamlit App")

# Add some text
st.write("Welcome! This is a simple app.")

# Create a text input box for the user's name
name = st.text_input("What's your name?")

# Check if the user has entered a name
if name:
    # Display a personalized greeting
    st.write(f"Hello, {name}! Nice to meet you.")