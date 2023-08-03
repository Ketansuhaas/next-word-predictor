import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Next word predictor"
)
st.title("Next word predictor")
st.write('''I have created a stacked LSTM model for the prediction of the next word in a word sequence.
         Use the side bar for navigation.''')
st.sidebar.success("Select a page above")
