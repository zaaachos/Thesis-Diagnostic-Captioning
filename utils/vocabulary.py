# tensorflow imports
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer


class Vocabulary:
    def __init__(self, texts:list, threshold:int = 3):
        """ Vocabulary class we utilised in CNN-RNN models. This class creates a vocabulary for our captions with a cut-off frequency.

        Args:
            texts (list): All training captions from which we extract our vocabulary
            threshold (int, optional): The cut-off frequency in our dictionary. Defaults to 3.
        """
        self.texts = texts
        # add <pad> abd <unk> tokens in vocabulary. <unk> refer to the OOV (out-of-vocabulary) tokens.
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.threshold = threshold
        # init the tokenizer
        self.tokenizer = Tokenizer(oov_token="<unk>")
        # dictionaries
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self) ->tuple[Tokenizer, dict, dict]:
        """ This function creates the Vocabulary we want to employ for our CNN-RNN model.

        Returns:
            tuple[Tokenizer, dict, dict]: The fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary. 
            The latters are mappers for words and index respectively
        """
        # fit in training captions
        self.tokenizer.fit_on_texts(self.texts)
        # create the vocabulary in sorted way
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
            # if the occurence of current word is less than the threshold, continue.
            if v >= self.threshold:
                word_index_threshold[k] = idx
                index_word_threshold[idx] = k
                idx += 1

        # get the dictionaries
        self.tokenizer.word_index = word_index_threshold
        self.tokenizer.index_word = index_word_threshold

        dictionary = self.tokenizer.word_index

        # append to the global ones
        for k, v in dictionary.items():
            self.word2idx[k] = v
            self.idx2word[v] = k

        print(f"Made a vocabulary with {len(self.word2idx)} words!")
        
        return self.tokenizer, self.word2idx, self.idx2word
