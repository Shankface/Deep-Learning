# -*- coding: utf-8 -*-
"""Deep_Learning_HW5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RfQdFpSZ16-39u8UB1RXubaoxic_64eC

For this project, I decided to go with an LSTM model. I started with 2 Bidir LSTM layers, max pooled them, then had 3 additional layers of decreasing width each with dropout.
  I played with the max sentence length and found that when I decreased it from the max sentence in the training data to just the average sentence length plus the std, the training time decreased significantly and the accuracy remained the same.
"""

#####
# Ayden Shankman
# ECE-472
# Assigment 5
# Got help from https://www.tensorflow.org/text/tutorials/text_classification_rnn
###

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

# Formatting data 
def format_data(data):
    data.columns = ['ClassIndex', 'Title', 'Description']
    x = data['Title'] + " " + data['Description'] # Concat title to rest of article
    y = data['ClassIndex'].apply(lambda x: x-1).values # Convert labels from 1-4 to 0-3
    return x, y

# Tokenizing sentences and padding them 
def tokenize(x_train, x_test, max_words, max_sent_len):
    tokenizer = Tokenizer(num_words = max_words, oov_token = 'oov')
    tokenizer.fit_on_texts(x_train)

    # Tokenize data
    seq_train = tokenizer.texts_to_sequences(x_train)
    seq_test = tokenizer.texts_to_sequences(x_test)

    # Pad data
    padded_seq_train = pad_sequences(seq_train, maxlen = max_sent_len)
    padded_seq_test = pad_sequences(seq_test, maxlen = max_sent_len)

    return padded_seq_train, padded_seq_test


# Training Model
def train_model(x_train, y_train, max_words, max_sent_len):
    
    checkpoint_callback = [
        ModelCheckpoint(
            filepath='weights.h5',
            monitor='val_accuracy', 
            mode='max', 
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    #----MODEL----
    model = keras.Sequential()

    model.add(Embedding(input_dim = max_words, 
                        output_dim = 32,
                        input_length = max_sent_len))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True))) 
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation = 'softmax'))

    adam = Adam(learning_rate = 0.001)
    model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer = adam,
                    metrics = ['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, validation_split = (7600/120000), batch_size = 256, epochs = 2, verbose = 1, callbacks = checkpoint_callback)

    model.save('model_1.h5')

    return model,history



if __name__ == "__main__":
    train_data = pd.read_csv('test.csv')
    test_data = pd.read_csv('test.csv')

    x_train, y_train = format_data(train_data)
    x_test, y_test = format_data(test_data)

    max_sent_len = int(x_train.map(lambda x: len(x.split())).mean()) + int(x_train.map(lambda x: len(x.split())).std()) # max sentence length is average + std
    max_words = 20000 # limit of how many unique words to train
    
    x_train, x_test = tokenize(x_train, x_test, max_words = max_words, max_sent_len = max_sent_len) # tokenizing training and testing data 
    
    train = False
    if(train):
        model,history = train_model(x_train, y_train, max_words, max_sent_len) # training model
        print("----------ACCURACY----------")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2) # displaying accuracy

    else:
        model = keras.models.load_model('model_1.h5')
        print("----------ACCURACY----------")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)