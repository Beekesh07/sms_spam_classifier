import streamlit as st
import joblib
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string



# Load the saved model and vectorizer
rfc = joblib.load('rfc_model.pkl')
tf = joblib.load('vectorizer.pkl')

# Initialize the stemmer
ps = PorterStemmer()

# Preprocessing function (same as before)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

# Streamlit app
def main():
    st.title('Text Classification with RandomForest')

    st.write("This app classifies text messages as spam or ham.")

    # Text input from the user
    user_input = st.text_area("Enter text for prediction", "")

    if st.button("Classify"):
        if user_input:
            # Preprocess the input text
            transformed_text = transform_text(user_input)

            # Transform the text using the loaded vectorizer
            x_input = tf.transform([transformed_text]).toarray()

            # Predict using the loaded model
            prediction = rfc.predict(x_input)

            # Display the prediction result
            if prediction[0] == 1:
                st.success("This message is classified as: Spam")
            else:
                st.success("This message is classified as: Ham")
        else:
            st.error("Please enter a text to classify.")

if __name__ == "__main__":
    main()
