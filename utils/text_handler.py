import re
import argparse
import numpy as np
import nltk
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

from utils.vocabulary import Vocabulary

nltk.download("punkt")


class TextHandler:
    def __init__(self, clean=False, use_sep=True):
        self.__clean = clean
        self.__start_token = "startsequence"
        self.__end_token = "endsequence"
        self.__seq_sep = None
        if use_sep:
            self.__seq_sep = " endofsequence "

    def get_basic_token(self):
        return self.__start_token, self.__end_token, self.__seq_sep

    def __preprocess_text(self, text):
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub("&", "and", text)
        text = re.sub("@", "at", text)
        text = re.sub("0", "zero", text)
        text = re.sub("1", "one", text)
        text = re.sub("2", "two", text)
        text = re.sub("3", "three", text)
        text = re.sub("4", "four", text)
        text = re.sub("5", "five", text)
        text = re.sub("6", "six", text)
        text = re.sub("7", "seven", text)
        text = re.sub("8", "eight", text)
        text = re.sub("9", "nine", text)
        text = re.sub("year old", "", text)
        text = re.sub("yearold", "", text)
        if self.__clean:
            text = self.__clean_text(text)

        text = text.strip().lower()
        text = " ".join(text.split())  # removes unwanted spaces
        if text == "":
            text = np.nan

        return text

    def __clean_text(self, text):
        regex = r"\d."
        text = re.sub(regex, "", text)

        regex = r"X+"
        text = re.sub(regex, "", text)

        regex = r"[^.a-zA-Z]"
        text = re.sub(regex, " ", text)

        regex = r"http\S+"
        text = re.sub(regex, "", text)
        # TODO check for period sign!
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|!?$%^&*'/+\[\]_]+", "", text)

        return text

    def separate_sequences(self, text):
        start, end, seq_sep = self.get_basic_token()
        if seq_sep is not None:
            sequences = nltk.tokenize.sent_tokenize(text)
            sequences = [s for s in sequences if len(s) > 5]
            text = self.seq_sep.join(sequences)
        return start + " " + text + " " + end

    def preprocess_all(self, texts):
        preprocessed_texts = [self.__preprocess_text(text) for text in texts]
        separated_texts = [self.separate_sequences(text) for text in preprocessed_texts]
        return separated_texts
