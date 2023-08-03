from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.layers import Dense
def build_model(lookback,vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size,100,input_length=lookback))
    model.add(LSTM(1000,return_sequences=True))
    model.add(LSTM(1000))
    model.add(Dense(1000,activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer='adam')
    return model

