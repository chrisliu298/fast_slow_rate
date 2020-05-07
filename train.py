# Models
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Dataset
import tensorflow_datasets as tfds

# Others
from statistics import mean
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def plot(history, string):
    """
    Plot training acc/loss and validation acc/loss
    """
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(["train_" + string, "valid_" + string])
    plt.show()


class GridSearch:
    def __init__(
        self,
        train_size,
        valid_size,
        test_size,
        lstm_search_space,
        dense_search_space,
        dropout_search_space,
        batch_size,
        learning_rate,
    ):
        # Parameters
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.lstm_search_space = lstm_search_space
        self.dense_search_space = dense_search_space
        self.dropout_search_space = dropout_search_space
        self.vocab_size = 20000
        self.embedding_dim = 50
        self.max_len = 512
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = 50
        self.patience = 10
        self.seed = 42
        self.trunc_type = "post"
        self.padding = "pre"
        self.oov_token = "<OOV>"
        self.callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
                restore_best_weights=True,
            )
        ]
        # Load the dataset
        imdb = tfds.load("imdb_reviews", as_supervised=True, shuffle_files=True)
        self.train_dataset = imdb["train"]
        self.test_dataset = imdb["test"]

    def prepare_data(self):
        train_text = []
        train_labels = []
        test_text = []
        test_labels = []

        # Put sentences and labels in lists
        for s, l in self.train_dataset:
            train_text.append(str(s.numpy()))
            train_labels.append(l.numpy())
        for s, l in self.test_dataset:
            test_text.append(str(s.numpy()))
            test_labels.append(l.numpy())

        # Convert them into numpy arrays
        train_text, train_labels, test_text, test_labels = (
            np.array(train_text),
            np.array(train_labels),
            np.array(test_text),
            np.array(test_labels),
        )

        # Shuffle the train/test set
        train_rand = np.arange(len(train_text))
        np.random.shuffle(train_rand)
        train_text = train_text[train_rand]
        train_labels = train_labels[train_rand]
        test_rand = np.arange(len(test_text))
        np.random.shuffle(test_rand)
        test_text = test_text[test_rand]
        test_labels = test_labels[test_rand]

        # Take the subset of the data
        train_reviews, valid_reviews = (
            train_text[: self.train_size],
            train_text[-self.valid_size :],
        )
        train_sentiments, valid_sentiments = (
            np.array(train_labels[: self.train_size]),
            np.array(train_labels[-self.valid_size :]),
        )
        test_reviews = test_text[: self.test_size]
        test_sentiments = np.array(test_labels[: self.test_size])

        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        tokenizer.fit_on_texts(train_reviews)

        train_seq = tokenizer.texts_to_sequences(train_reviews)
        train_padded = pad_sequences(
            train_seq,
            maxlen=self.max_len,
            truncating=self.trunc_type,
            padding=self.padding,
        )
        valid_seq = tokenizer.texts_to_sequences(valid_reviews)
        valid_padded = pad_sequences(
            valid_seq,
            maxlen=self.max_len,
            truncating=self.trunc_type,
            padding=self.padding,
        )
        test_seq = tokenizer.texts_to_sequences(test_reviews)
        test_padded = pad_sequences(
            test_seq,
            maxlen=self.max_len,
            truncating=self.trunc_type,
            padding=self.padding,
        )

        print("Training sentences count:", len(train_padded))
        print("Training labels count:", len(train_sentiments))
        print("Validation sentences count:", len(valid_padded))
        print("Validation labels count:", len(valid_sentiments))
        print("Test sentences count:", len(test_padded))
        print("Testing labels count:", len(test_sentiments))
        return [
            (train_padded, train_sentiments),
            (valid_padded, valid_sentiments),
            (test_padded, test_sentiments),
        ]

    def build_model(self, lstm_hidden_size, dense_hidden_size, dropout_rate):
        model = Sequential()
        model.add(
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len)
        )
        model.add(Bidirectional(LSTM(lstm_hidden_size, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(dense_hidden_size, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=1e-3),
            metrics=["accuracy"],
        )
        return model

    def train(self, data):
        (
            (train_padded, train_sentiments),
            (valid_padded, valid_sentiments),
            (test_padded, test_sentiments),
        ) = data
        histories = []
        counter = 0
        for lstm_size in self.lstm_search_space:
            for dense_size in self.dense_search_space:
                for dropout_rate in self.dropout_search_space:
                    model = self.build_model(lstm_size, dense_size, dropout_rate)
                    counter += 1
                    print(f"========== Trial {counter} Summary ==========")
                    print(f"   lstm_search_space = {lstm_size}")
                    print(f"  dense_search_space = {dense_size}")
                    print(f"dropout_search_space = {dropout_rate}")
                    history = model.fit(
                        train_padded,
                        train_sentiments,
                        validation_data=(valid_padded, valid_sentiments),
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        shuffle=True,
                        callbacks=self.callbacks,
                    )
                    log = {
                        "lstm_hidden_size": lstm_size,
                        "dense_size": dense_size,
                        "dropout_rate": dropout_rate,
                        "train_acc_converge": history.history["accuracy"][-11],
                        "train_acc_final": history.history["accuracy"][-1],
                        "valid_acc_converge": history.history["val_accuracy"][-11],
                        "valid_acc_final": history.history["val_accuracy"][-1],
                    }
                    histories.append(log)
                    print(log)
                    print()
        return histories
