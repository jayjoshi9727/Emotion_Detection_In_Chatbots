"""Evaluate deep learning models on unseen data and save the final metrics."""

import argparse as ap
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


def parse_script_parameters():
    """Script level parameters."""

    parser = ap.ArgumentParser("Script level default variables")

    parser.add_argument("--model_path", type=str, default=".hdf5")

    parser.add_argument("--encoder_class_path", type=str, default="./le_classes.pkl")

    parser.add_argument("--x_test_path", type=str, default="./isear_kaggle_X_test.pkl")

    parser.add_argument("--y_test_path", type=str, default="./isear_kaggle_Y_test.pkl")

    parser.add_argument('--graph_name', type=str, default='./lstm_test_set_score.pdf')

    args = parser.parse_args()

    return args


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

    print('read script level parameters...')
    kwargs = parse_script_parameters()

    print('load deep learning model....')
    model = load_model(kwargs.model_path)

    X_test = load_pickle_objects(kwargs.x_test_path)

    Y_test = load_pickle_objects(kwargs.y_test_path)

    le = load_pickle_objects(kwargs.le_classes.pkl)

    print('predict on test data...')
    predY = model.predict(X_test)
    predYClasses = np.argmax(predY, axis=1)
    trueY = np.argmax(Y_test, axis=1)

    print('classification report....')
    confusionMTX = confusion_matrix(trueY, predYClasses)
    print(classification_report(trueY, predYClasses, target_names=le.classes_))

    print('plot and save the final metrics...')
    test_classification_dict = classification_report(
        trueY, predYClasses, target_names=le.classes_, output_dict=True
    )

    df = pd.DataFrame(test_classification_dict).transpose()

    del df["support"]

    df = df[0:4]

    font = {"family": "serif", "color": "darkblue", "weight": "normal", "size": 16}
    plt.savefig(kwargs.graph_name, bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.figsize"] = [7, 5]
    plt.rcParams["figure.dpi"] = 500
    df.plot(kind="bar")
    scale_factor = 0.5
    ymin, ymax = plt.ylim()
    plt.ylim(scale_factor, 1)
    plt.legend(loc="upper center", bbox_to_anchor=(1.2, 0.5))
    plt.xlabel("Labels", fontdict=font)
    plt.ylabel("Metrics", fontdict=font)
    plt.title("Metric: Classification Metrics Test Set", fontdict=font)
    plt.show()

    print('program ended success...')
