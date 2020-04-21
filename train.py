import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


# Parameters
TRAIN_SIZES = list(range(2500, 25000, 2500))
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
PATIENCE = 2
EPOCHS = 15
DROPOUT = 0.1
HIDDEN_DIM = 128
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LEN = 512
TRUNC_TYPE = "post"
PADDING = "pre"
OOV_TOKEN = "<OOV>"
CALLBACK = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)


def plot(history, string, count):
    """
    Plot training acc/loss and validation acc/loss
    """
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(["train_" + string, "valid_" + string])
    plt.savefig(f"trial{count}_{string}.png")


def build_lstm():
    """
    Build a lstm layer model
    """
    model_lstm = Sequential()
    model_lstm.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model_lstm.add(Dropout(DROPOUT))
    model_lstm.add(LSTM(HIDDEN_DIM))
    model_lstm.add(Dropout(DROPOUT))
    model_lstm.add(Dense(1, activation="sigmoid"))
    # Compile the model
    model_lstm.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    model_lstm.summary()
    return model_lstm


def train(train_size, trial_count):
    imdb = tfds.load("imdb_reviews", as_supervised=True, shuffle_files=True)

    train_data, test_data = imdb["train"], imdb["test"]

    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []

    for sentence, label in train_data:
        training_sentences.append(str(sentence.numpy()))
        training_labels.append(label.numpy())

    for sentence, label in test_data:
        testing_sentences.append(str(sentence.numpy()))
        testing_labels.append(label.numpy())

    train_s = training_sentences[:train_size]
    train_l = np.array(training_labels[:train_size])

    valid_s = training_sentences[22500:]
    valid_l = np.array(training_labels[22500:])

    test_s = testing_sentences[:12500]
    test_l = np.array(testing_labels[:12500])

    print("Training sentences count:", len(train_s))
    print("Training labels count:", len(train_l))
    print("Validation sentences count:", len(valid_s))
    print("Validation labels count:", len(valid_l))
    print("Test sentences count:", len(test_s))
    print("Testing labels count:", len(test_l))

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_s)

    train_sequences = tokenizer.texts_to_sequences(train_s)
    train_padded = pad_sequences(
        train_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE, padding=PADDING
    )
    valid_sequences = tokenizer.texts_to_sequences(valid_s)
    valid_padded = pad_sequences(
        valid_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE, padding=PADDING
    )
    testing_sequences = tokenizer.texts_to_sequences(test_s)
    testing_padded = pad_sequences(
        testing_sequences, maxlen=MAX_LEN, truncating=TRUNC_TYPE, padding=PADDING
    )

    model_lstm = build_lstm()

    history = model_lstm.fit(
        train_padded,
        train_l,
        validation_data=(valid_padded, valid_l),
        batch_size=BATCH_SIZE,
        shuffle=True,
        epochs=EPOCHS,
        callbacks=[CALLBACK],
    )

    # loss, acc = model_lstm.evaluate(testing_padded, test_l, batch_size=32)
    plot(history, "accuracy", trial_count)
    plot(history, "loss", trial_count)


if __name__ == "__main__":
    trial_counter = 0

    for size in TRAIN_SIZES:
        trial_counter += 1
        print("=" * 10 + f" Trial {trial_counter}: {size} training examples " + "=" * 10)
        train(size, trial_counter)
