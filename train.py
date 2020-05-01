import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


# Parameters
# TRAIN_SIZES = list(range(2500, 25000, 2500))
# BATCH_SIZE = 32
# LEARNING_RATE = 5e-5
# PATIENCE = 2
# EPOCHS = 15
# DROPOUT = 0.1
# HIDDEN_DIM = 128
# VOCAB_SIZE = 10000
# EMBEDDING_DIM = 16
# MAX_LEN = 512
# TRUNC_TYPE = "post"
# PADDING = "pre"
# OOV_TOKEN = "<OOV>"
# CALLBACK = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=PATIENCE, restore_best_weights=True
# )


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


class BiLSTM:
    def __init__(
        self,
        vocab_size=20000,
        embedding_dim=50,
        max_seq_len=128,
        learning_rate=5e-5,
        dropout=0.1,
        hidden_dim=64,
        loss="binary_crossentropy",
    ):
        self.vocab_size = 20000
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.training_sizes = list(range(1000, 23500, 1000))

    def build_model(self):
        model = Sequential(
            [
                Embedding(
                    self.vocab_size, self.embedding_dim, input_length=self.max_seq_len
                ),
                Bidirectional(
                    LSTM(
                        self.hidden_dim,
                        return_sequences=True,
                        dropout=self.dropout,
                        recurrent_dropout=self.dropout,
                    )
                ),
                GlobalMaxPool1D(),
                Dense(64, activation="relu"),
                Dropout(self.dropout),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            loss=self.loss,
            optimizer="adam",  # Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        model.summary()
        return model


class IMDB:
    def __init__(self, train_size, valid_size, test_size, vocab_size=20000, max_len=512):
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dataset = tfds.load("imdb_reviews", as_supervised=True, shuffle_files=True)
        self.train_dataset = self.dataset["train"]
        self.test_dataset = self.dataset["test"]

    def preprocess_imdb(self):
        # Group train/test sentences/labels
        train_text = []
        train_labels = []
        test_text = []
        test_labels = []
        for s, l in self.train_dataset:
            train_text.append(str(s.numpy()))
            train_labels.append(l.numpy())
        for s, l in self.test_dataset:
            test_text.append(str(s.numpy()))
            test_labels.append(l.numpy())

        train_reviews, valid_reviews = train_text[: self.train_size], train_text[22500:]
        train_sentiments, valid_sentiments = (
            np.array(train_labels[: self.train_size]),
            np.array(train_labels[22500:]),
        )
        test_reviews = test_text[: self.test_size]
        test_sentiments = np.array(test_labels[: self.test_size])

        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_reviews)

        train_seq = tokenizer.texts_to_sequences(train_reviews)
        train_padded = pad_sequences(
            train_seq, maxlen=self.max_len, truncating="post", padding="pre"
        )
        valid_seq = tokenizer.texts_to_sequences(valid_reviews)
        valid_padded = pad_sequences(
            valid_seq, maxlen=self.max_len, truncating="post", padding="pre"
        )
        test_seq = tokenizer.texts_to_sequences(test_reviews)
        test_padded = pad_sequences(
            test_seq, maxlen=self.max_len, truncating="post", padding="pre"
        )

        print("Training sentences count:", len(train_padded))
        print("Training labels count:", len(train_sentiments))
        print("Validation sentences count:", len(valid_padded))
        print("Validation labels count:", len(valid_sentiments))
        print("Test sentences count:", len(test_padded))
        print("Testing labels count:", len(test_sentiments))

        return (
            (train_padded, valid_padded, test_padded),
            (train_sentiments, valid_sentiments, test_sentiments),
        )
