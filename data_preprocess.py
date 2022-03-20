"""Convert the natural language data into the format compatible with algorithms."""

import os
import argparse as ap
import pickle
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from autocorrect import Speller
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder


def parse_script_parameters():
    """Script level parameters."""

    parser = ap.ArgumentParser("Script level default variables")

    parser.add_argument("--")


def read_input_data(file_path) -> pd.DataFrame:
    """Read input data in pandas frame.

    Args:
        file_path: csv file path

    Returns:
        pandas frame
    """
    df = pd.read.csv(file_path)
    print(df.columns)
    return df


def clean_text(txt: str) -> str:
    """cleans the input text in the following steps
    1- replace contractions
    2- removing punctuation
    3- spliting into words
    4- removing stopwords
    5- removing leftover punctuations

    Args:
        txt: input raw text in natural language
    Returns:
        cleaned text without punctuations and stopwords.

    """
    contraction_dict = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "this's": "this is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "here's": "here is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }

    def _get_contractions(contraction_dict):
        contraction_re = re.compile("(%s)" % "|".join(contraction_dict.keys()))
        return contraction_dict, contraction_re

    def replace_contractions(text):
        contractions, contractions_re = _get_contractions(contraction_dict)

        def replace(match):
            return contractions[match.group(0)]

        return contractions_re.sub(replace, text)

    # replace contractions
    txt = replace_contractions(txt)

    # remove punctuations
    txt = "".join([char for char in txt if char not in string.punctuation])
    txt = re.sub("[0-9]+", "", txt)

    # split into words
    words = word_tokenize(txt)

    # remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if not w in stop_words]

    # removing leftover punctuations
    words = [word for word in words if word.isalpha()]

    cleaned_text = " ".join(words)
    return cleaned_text


def tokenize_sentences(lines):
    """Tokenize input sentences
    Args:
        lines: input text data
    Returns:
        tokenized cleaned text lines
    """
    spell = Speller(lang="en")
    text_lines = list()
    for line in lines:
        #     Autocorrect Incorrect Spellings
        spellings = spell(line)
        #     Create tokens
        tokens = word_tokenize(spellings)
        tokens = [w.lower() for w in tokens]
        #     Remove punctuations
        table = str.maketrans("", "", string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        #     Remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        text_lines.append(words)
    # Print first preprocessed statement
    print(text_lines[0])


def convert_words_into_integers(cleaned_text) -> tuple:
    """Tokenize words into integers.

    Args:
        cleaned_text: cleaned text input
    Returns:
        text into integers

    """

    max_length = 50  # Maximum number of words that can occur in one sentence.

    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(cleaned_text)
    sequences = tokenizer_obj.texts_to_sequences(cleaned_text)
    print(sequences[0])

    # print unique tokens
    word_index = tokenizer_obj.word_index
    print("Unique tokens " + str(len(word_index)))

    # print vocab size
    vocab_size = len(tokenizer_obj.word_index) + 1
    print("Vocab_size " + str(vocab_size))

    # extra padding for sentences having lengths greater than 50
    lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')
    print(lines_pad.shape)
    print(lines_pad[0])

    return word_index, lines_pad


def generate_embedding_matrix(glove_embedding: dict, word_index: dict) -> np.array:
    """Generate word vector matrix for each word in the text data.

    Args:
        glove_embedding: pretrained glove embeddings.
        word_index: dictionary of words with index no in the dataset.

    Returns:
        glove vector matrix for total words in the dataset.
    """
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in glove_embedding.keys():
            embedding_vector = glove_embedding[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Embedding Matrix Generated : ", embedding_matrix.shape)

    return embedding_matrix


def categorical_label_encoder(df: pd.DataFrame, categorical_column: str) -> tuple:
    """Encode categorical columns into algorithm acceptable format.

    Args:
        df: pandas dataframe
        categorical_column: name of categorical column in dataset.

    Returns:
        tuple of encoder object and formatted data columns.
    """
    le = LabelEncoder()
    le.fit(data['Emotion_value'])
    print("Classes: " + str(le.classes_))
    encode_Train_Labels = le.transform(df[categorical_column])

    # Make labels categorical
    clear_Train_Label = np_utils.to_categorical(encode_Train_Labels)
    num_classes = clear_Train_Label.shape[1]
    print("Number of classes: " + str(num_classes))

    return le, clear_Train_Label


def save_pickle_objects(obj, file_path):
    """Save pickle objects.

    Args:
        obj: object to save as .pkl
        file_path: path were to store the object
    """
    pickle.dump(obj, open(file_path, 'wb'))


def load_pickle_objects(file_path):
    """Load pickle objects.

    Args:
        file_path: path to load pickle object.

    Returns:
            file object.
    """
    file_obj = pickle.load(open(file_path, "rb"))

    return file_obj


def viterbi_segment(text):
    """Dynamic algorithm to find the best split combination of words.

    Args:
        text: text input
    Returns:
        tuple of word probability and occurence
    """
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max(
            (probs[j] * word_prob(text[j:i]), j)
            for j in range(max(0, i - max_word_length), i)
        )
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]: i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]


def word_prob(word: str) -> str:
    """Calculate word occurrence probability.
    Args:
        word:
    Returns:
        word count probability

    """
    return dictionary[word] / total


def words(text: str) -> list:
    """Regular expression to convert to lower case.

    Args:
        text
    Returns:
        lower case words in sentences
    """
    return re.findall("[a-z]+", text.lower())


dictionary = Counter(words(open("big.txt").read()))
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))

if __name__ == "__main__":
    print("Program started....")

    print('reading data set...')
    filepath = ""
    data = read_input_data(filepath)
    column_name = ''

    print('cleaning text data..')
    data[column_name] = data[column_name].apply(lambda txt: clean_text(txt))

    print('convert into machine readable format...')
    word_index, lines_pad = convert_words_into_integers(data[column_name])

    save_pickle_objects(word_index, './word_index.pkl')

    save_pickle_objects(lines_pad, './lines_pad.pkl')

    glove_emb = load_pickle_objects('./ibc_word_embeddings')

    print('generating embedding matrix...')
    embedding_matrix = generate_embedding_matrix(glove_emb, word_index)

    save_pickle_objects(embedding_matrix, './embedding_matrix.pkl')

    print('encode categorical variables...')
    le_classes, clear_train_label = categorical_label_encoder(data, "")

    save_pickle_objects(le_classes, './le_classes.pkl')

    save_pickle_objects(clear_train_label, './clear_train_label.pkl')

    print("program finished successfully...")
