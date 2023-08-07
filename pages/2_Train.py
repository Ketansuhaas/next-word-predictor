import streamlit as st
import streamlit as st
from model import build_model
from preprocess import preprocess_context, get_next_word, tokenizer
import numpy as np
from tensorflow import keras
st.set_page_config(
    layout="wide",
    page_title="Train"
)
st.title("Train")
# print model information

file = st.file_uploader("Choose your .txt file")
if (st.button("Train")):

    x_train, y_train, vocab_size = preprocess_context(raw)
    model = build_model(5,vocab_size)
    model.fit(x_train,y_train, epochs=1000, batch_size = 32, verbose=1)
    model.save("first_model.h5")
    st.session_state.model = model


if file is not None:
    raw = str(file.read(),"utf-8")
    st.write(raw)

