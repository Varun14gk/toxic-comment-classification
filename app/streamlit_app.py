#importing the libraries
import streamlit as st
import pickle
from text_classification_pipeline import TextPreprocessor  # Make sure to have the TextPreprocessor class in a separate file or include it in this script

# Load the models and vectorizer from the pickle file
with open('text_classification_models.pkl', 'rb') as file:
    data = pickle.load(file)
    models = data['models']
    vectorizer = data['vectorizer']

# Initialize the text preprocessor
preprocessor = TextPreprocessor()

# Streamlit app
st.title('Toxic Comment Classification')

# Text input
user_input = st.text_area("Enter a comment to classify:", value='', height=None, max_chars=None, key=None)

# Predict button
if st.button('Predict'):
    if user_input:
        # Preprocess the input
        preprocessed_input = preprocessor.preprocess(user_input)
        # Vectorize the input
        vectorized_input = vectorizer.transform([preprocessed_input])
        # Generate predictions
        predictions = {}
        for label, model in models.items():
            predictions[label] = model.predict(vectorized_input)[0]
        
        # Display predictions in boxes with the correct colors
        st.subheader('Predictions:')
        for label, prediction in predictions.items():
            if prediction == 1:
                st.markdown(f"<div style='color: white; background-color: red; padding: 10px; border-radius: 5px;'>{label.replace('_', ' ').capitalize()}: Yes</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color: white; background-color: green; padding: 10px; border-radius: 5px;'>{label.replace('_', ' ').capitalize()}: No</div>", unsafe_allow_html=True)
    else:
        st.write("Please enter a comment to get predictions.")

# Run the Streamlit app by saving this script and running the following command:
# streamlit run streamlit_app.py
