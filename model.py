import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import *
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import re

data = pd.read_csv(
    "/Users/arunkaul/Desktop/Training Data/ReviewsFinal.csv",
    names=["Score", "text"],
)


data["text"] = data["text"].apply(lambda x: str(x).lower())
data["text"] = data["text"].apply((lambda x: re.sub("[^a-zA-z0-9\s]", "", x)))


vocab = 20000
tokenizer = Tokenizer(num_words=vocab, split=" ")
tokenizer.fit_on_texts(data["text"].values)
X = tokenizer.texts_to_sequences(data["text"].values)
X = pad_sequences(X)


Y = data["Score"].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


model = Sequential()
model.add(Embedding(vocab, 32, input_length=X.shape[1]))
model.add(Conv1D(32, 3, padding="same", activation="relu"))
model.add(Dropout(0.65))
model.add(Conv1D(16, 3, padding="same", activation="relu"))
model.add(
    Bidirectional(LSTM(16, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
)
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.65))
model.add(Dense(2, activation="softmax"))
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.005),
    metrics=["accuracy"],
)
print(model.summary())


batch_size = 64
info = model.fit(
    X_train,
    Y_train,
    epochs=30,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_test, Y_test),
)

plt.title("Loss Analysis")
plt.plot(info.history["loss"], label="Loss")
plt.plot(info.history["accuracy"], label="Accuracy")
plt.legend()
model.save("model2.h5")
plt.show()
