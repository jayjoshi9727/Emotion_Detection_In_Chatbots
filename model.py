"""Define and train deep learning sequential models for emotion detection."""
import argparse as ap
import pickle


import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    GRU,
    LSTM,
    Dense,
    Embedding,
)
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def parse_script_parameters():
    """Script level parameters."""

    parser = ap.ArgumentParser("Script level default variables")

    parser.add_argument(
        "--embedding_matrix_path", type=str, default="./embedding_matrix.pkl"
    )

    parser.add_argument(
        "--clear_train_label_path", type=str, default="./clear_train_label.pkl"
    )

    parser.add_argument("--lines_pad_path", type=str, default="./lines_pad.pkl")

    parser.add_argument("--word_index_path", type=str, default="./word_index.pkl")

    parser.add_argument("--metrics", type=str, default="accuracy")

    parser.add_argument("--loss", type=str, default="categorical_crossentropy")

    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--no_classes", type=int, default=4)

    parser.add_argument("--activation", type=str, default="tanh")

    parser.add_argument("--output_activation", type=str, default="softmax")

    parser.add_argument("--output_unit_1", type=int, default=256)
    parser.add_argument("--output_unit_2", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=80)

    args = parser.parse_args()

    return args


def define_lstm_stacked_sequential_model(
    word_index: dict,
    embedding_dim: int,
    embedding_matrix: np.array,
    input_length: int,
    output_unit_1: int,
    output_unit_2: int,
    metrics: str,
    loss: str,
    optimizer: str,
    no_classes: int,
    activation: str,
    output_activation: str,
):
    """Define LSTM Stack models.

    Args:
        word_index:
        embedding_dim:
        embedding_matrix:
        input_length:
        output_unit_1:
        output_unit_2:
        metrics:
        loss:
        optimizer:
        no_classes:
        activation:
        output_activation:

    Returns:
        stacked deep learning lstm model.
    """
    model = Sequential()
    embedding_layer = Embedding(
        len(word_index) + 1,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
    )
    model.add(embedding_layer)
    model.add(LSTM(units=output_unit_1, return_sequences=True, activation=activation))
    model.add(LSTM(units=output_unit_2, return_sequences=True, activation=activation))
    model.add(Dense(no_classes, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    print(model.summary())

    return model


def load_pickle_objects(file_path):
    """Load pickle objects.

    Args:
        file_path: path to load pickle object.

    Returns:
            file object.
    """
    file_obj = pickle.load(open(file_path, "rb"))

    return file_obj


if __name__ == "__main__":
    print("program started...")

    print("reading script level parameters..")
    kwargs = parse_script_parameters()

    word_index = load_pickle_objects(kwargs.word_index_path)

    embedding_matrix = load_pickle_objects(kwargs.embedding_matrix_path)

    clear_train_label = load_pickle_objects(kwargs.clear_train_label_path)

    lines_pad = load_pickle_objects(kwargs.lines_pad_path)

    print("splitting the dataset into train and test...")
    train_X, test_X, train_Y, test_Y = train_test_split(
        lines_pad,
        clear_train_label,
        test_size=0.25,
        shuffle=True,
        stratify=clear_train_label,
    )

    X_test, X_valid, Y_test, Y_valid = train_test_split(
        test_X, test_Y, test_size=0.4, shuffle=True, stratify=test_Y
    )

    print("defining model and checkpoint..")
    filepath = (
        "isear_kaggle_emotion_model/weights.best_{epoch:03d}-{" "val_accuracy:.3f}.hdf5"
    )
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode="auto")

    callbacks_list = [checkpoint]

    model = define_lstm_stacked_sequential_model(
        word_index,
        kwargs.embedding_dim,
        embedding_matrix,
        kwargs.input_length,
        kwargs.output_unit_1,
        kwargs.output_unit_2,
        kwargs.metrics,
        kwargs.loss,
        kwargs.optimizer,
        kwargs.no_classes,
        kwargs.activation,
        kwargs.output_activation,
    )

    print("train model....")
    # Train model
    history = model.fit(
        train_X,
        train_Y,
        validation_data=(X_valid, Y_valid),
        epochs=kwargs.epochs,
        callbacks=[callbacks_list],
    )

    loss_train = history.history["accuracy"]
    loss_validate = history.history["val_accuracy"]
    epochs = range(0, kwargs.epochs)

    print("plot training curve...")
    plt.plot(epochs, loss_train, "g", label="Training accuracy")
    plt.plot(epochs, loss_validate, "b", label="validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("./training_curve.png")

    print("model is trained..")
