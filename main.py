import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Ensure 'punkt' resource is available
nltk.download('punkt')  # Download 'punkt' resource
nltk.download('stopwords')  # Download 'stopwords' resource

# If running in a deployed environment, set the path to the nltk data folder
# Example path (adjust according to where your nltk_data is located):
nltk.data.path.append('/home/appuser/nltk_data')


# Function to transform the text (preprocessing)
def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = word_tokenize(text)  # Tokenize the text using NLTK's word_tokenize
    y = []

    # Remove non-alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    ps = nltk.PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)


# Load the trained model and vectorizer
model = joblib.load('rfc_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Streamlit app
def main():
    st.title("Text Classification with RandomForest")
    st.write("This app classifies text messages as spam or ham.")

    # User input for prediction
    user_input = st.text_area("Enter text for prediction", "")

    if st.button("Predict"):
        if user_input:
            transformed_text = transform_text(user_input)
            text_vector = vectorizer.transform([transformed_text]).toarray()  # Transform user input text
            prediction = model.predict(text_vector)  # Make prediction using the trained model

            if prediction == 1:
                st.write("Prediction: Spam")
            else:
                st.write("Prediction: Ham")
        else:
            st.write("Please enter some text to classify.")


# Run the app
if __name__ == "__main__":
    main()
