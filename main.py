import nltk
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
st.title('SMS Spam Classifier')

tf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('rfc_model.pkl','rb'))

sms=st.text_input('Enter message')

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

if st.button('predict'):
    transformed_sms=transform_text(sms)
    vector_input=tf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    if result==0:
        st.header('not spam')
    else:
        st.header('spam')
