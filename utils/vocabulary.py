import pandas as pd
import re
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer


class Vocabulary:
    def __init__(self, texts: list, threshold: int = 3):
        self.texts = texts
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.threshold = threshold
        self.tokenizer = Tokenizer(oov_token="<unk>")
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self):
        self.tokenizer.fit_on_texts(self.texts)
        sorted_vocab = dict(
            sorted(self.tokenizer.word_counts.items(), key=lambda t: t[1], reverse=True)
        )

        word_index_threshold, index_word_threshold = {}, {}

        # add pad and unk token to tokenizer dictionary
        word_index_threshold[self.pad_token] = 0
        index_word_threshold[0] = self.pad_token
        word_index_threshold[self.unk_token] = 1
        index_word_threshold[1] = self.unk_token

        # begin from index=2
        idx = 2
        for k, v in sorted_vocab.items():
            if v >= self.threshold:
                word_index_threshold[k] = idx
                index_word_threshold[idx] = k
                idx += 1

        self.tokenizer.word_index = word_index_threshold
        self.tokenizer.index_word = index_word_threshold

        dictionary = self.tokenizer.word_index

        for k, v in dictionary.items():
            self.word2idx[k] = v
            self.idx2word[v] = k

        print(f"Made a vocabulary with {len(self.word2idx)} words!")
        
        return self.tokenizer, self.word2idx, self.idx2word
