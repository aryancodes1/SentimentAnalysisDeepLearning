import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import *
import pandas as pd
import re
from tkinter import *
import matplotlib.pyplot as plt

data = pd.read_csv(
    "/Users/arunkaul/Desktop/Training Data/ReviewsFinal.csv",
    names=["Score", "text"],
)


d = {1: "Positive", 0: "Negative"}
L = np.array(["Negative", "Positive"])
data["text"] = data["text"].apply(lambda x: str(x).lower())
data["text"] = data["text"].apply((lambda x: re.sub("[^a-zA-z0-9\s]", "", x)))


features = 20000
tokenizer = Tokenizer(num_words=features, split=" ")
tokenizer.fit_on_texts(data["text"].values)
X = tokenizer.texts_to_sequences(data["text"].values)
X = pad_sequences(X)

model = load_model("model2.h5")
model.compile(
    optimizer=Nadam(0.005),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
)
model.summary()
win = Tk()
win.title("Review Analysis")
var = StringVar(win)


def action():
    Text = []
    Text.append(var.get())
    Text = tokenizer.texts_to_sequences(Text)
    Text = pad_sequences(Text, maxlen=29, dtype="int32", value=0)
    sentiment = model.predict(Text)[0]
    sentiment = np.array(sentiment)
    plt.title("Review Analysis")
    plt.bar(L, sentiment, width=0.2, color="maroon")
    plt.show()
    print(d[np.argmax(sentiment)])


Label1 = Label(win, text="Enter Review ").pack(side="top",fill = "x",pady=20)
Entry1 = Entry(win, textvariable=var).pack(side="top",fill = "x",pady=20)
Button1 = Button(win, command=action, text="Submit").pack(side="top",fill = "x",pady=20)
win.mainloop()
