import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import streamlit as st
import numpy as np
nltk.download('wordnet')
nltk.download('omw-1.4')
tokenizer = Tokenizer()
ps = WordNetLemmatizer()

def preprocess_context(para):
    sentences = para.split('.' or '\n')
    corpus = []
    for i in range(0, len(sentences)):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower()
        review = review.split()
        
        review = [ps.lemmatize(word) for word in review]# if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    tokenizer.fit_on_texts(corpus)
    st.session_state.tokenizer = tokenizer
    sequence_data = tokenizer.texts_to_sequences(corpus)

    sent_length=10
    embedded_docs=pad_sequences(sequence_data,padding='pre',maxlen=sent_length)

    dict = tokenizer.index_word

    y_train = []
    x_train = []


    for s in embedded_docs:
        for i in range(4):
            x_train.append(s[i:i+5])
            y_train.append(s[i+5])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    y_train = y_train.reshape(1,-1)
    y_train = y_train[0]

    y_cat = to_categorical(y_train, num_classes=len(dict)+1)

    
    return x_train,y_cat,len(tokenizer.index_word)+1


def get_next_word(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.lemmatize(word) for word in review]# if not word in stopwords.words('english')]
    review = ' '.join(review)
    z = tokenizer.texts_to_sequences([review])
    z = pad_sequences(z,padding="pre",maxlen=10)
    z = np.array(z)
    z = z[0][-3:]
    z = np.expand_dims(z,axis=0)
    return z




