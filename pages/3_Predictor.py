import streamlit as st
import streamlit as st
import streamlit_scrollable_textbox as stx
from model import build_model
from preprocess import preprocess_context, get_next_word, tokenizer
import numpy as np
from tensorflow import keras
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Predictor"
)
st.title("Predictor")
c1,c2 = st.columns(2)
c1.title("Input")
in_text = c1.text_input("Enter your context")      

if(c1.button("Submit")):  
    predict_this = get_next_word(in_text)
    #model = keras.models.load_model("first_model.h5")
    k = st.session_state.model.predict(predict_this)
    #st.write(k.shape)
    x = np.argsort(k[0])[::-1][:]
    prob = []
    word_order = []
    for i in x:
        if i!=0:
            word_order.append(st.session_state.tokenizer.index_word[i])
        else:
            word_order.append("<OOV>")
        prob.append(k[0][i])
    df = pd.DataFrame({
        "Words": word_order,
        "Probabilities": prob
    }
) 
    c2.title("Next words")
    c2.dataframe(data = df, width=700, height=768)
