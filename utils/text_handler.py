# re imports
import re

# numpy imports 
import numpy as np

# tensorflow imports
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

# inflect imports
import inflect
# create the NUMBER2TEXT object which helps us to convert each numerical text to its textual represantation
NUMBER_TO_TEXT = inflect.engine()

# utils imports
from utils.vocabulary import Vocabulary
# nltk imports
import nltk
nltk.download("punkt")

class TextHandler:
    def __init__(self, clean:bool=False, use_sep:bool=True):
        """ Text Hanlder class we used to pre-process our captions.
        The steps are provided in my Thesis.

        Args:
            clean (bool, optional): If we want to clean our text from special words like x-XXXX. Defaults to False.
            use_sep (bool, optional): If we want to separate our sentences with a SEQ_SEP token. Defaults to True.
        """
        self.__clean = clean
        self.__start_token = "startsequence"
        self.__end_token = "endsequence"
        self.__seq_sep = None
        if use_sep:
            self.__seq_sep = " endofsequence "

    def get_basic_token(self) -> tuple[str, str, str]:
        """ Returns the start, end, and seq_sep special tokens

        Returns:
            tuple[str, str, str]: start, end, and seq_sep tokens
        """
        return self.__start_token, self.__end_token, self.__seq_sep
    
    def remove_punctuations(self, text:str) -> str:
        """ Removes punctuations from training captions as well as cpecial characters

        Args:
            text (str): Text to pre-process

        Returns:
            str: Pre-processed text, without punctuation
        """
        return re.sub(r"[-()\"#/@;:<>{}`+=~|!.?$%^&*'/+\[\]_]+", "", text)
    
    def num2words(self, text:str) -> str:
        """This function converts each numerical text to its textual represantation. Like 10 to ten, and not onezero.

        Args:
            text (str): Text to pre-process

        Returns:
            str: Pre-processed text, with textual numbers
        """
        sentences = text.split('.')
        new_seqs = list()
        # get all sequences
        for s in sentences:
            tokens = s.split()
            new_tokens = list()
            # for each seq, get all words
            for token in tokens:
                # find the number
                try:
                    number = int(token)
                    # convert to each textual represantion. This also converts 10 to ten, and not onezero
                    word = NUMBER_TO_TEXT.number_to_words(number)
                except:
                    word = token
                new_tokens.append(word)
            new_seqs.append(' '.join(new_tokens))

        # connect again whole sentence
        modified_text = '. '.join(new_seqs)
        return modified_text
            

    def __preprocess_text(self, text:str) -> str:
        """ Exetures the pre-processed steps. More details are provided in my Thesis

        Args:
            text (str): Text to pre-process

        Returns:
            str: Pre-processed text. 
        """
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
        text = re.sub("year old", "", text)
        text = re.sub("yearold", "", text)
        
        text = self.num2words(text)
        
        if self.__clean:
            text = self.__clean_text(text)

        text = text.strip().lower()
        text = " ".join(text.split())  # removes unwanted spaces
        if text == "":
            text = np.nan

        return text

    def __clean_text(self, text:str) -> str:
        """ This function cleans the text from special words.

        Args:
            text (str): Text to pre-process

        Returns:
            str: Pre-processed text, without special wortds.
        """
        regex = r"\d."
        text = re.sub(regex, "", text)

        regex = r"X+"
        text = re.sub(regex, "", text)

        regex = r"[^.a-zA-Z]"
        text = re.sub(regex, " ", text)

        regex = r"http\S+"
        text = re.sub(regex, "", text)

        return text

    def separate_sequences(self, text:str) -> str:
        """ This function reads a sequence of texts and appends a SEQ_SEP token between sentences, for better training.
        More details are provided in my Thesis

        Args:
            text (str): Text to pre-process

        Returns:
            str: Pre-processed text, with SEQ_SEP special token.
        """
        start, end, seq_sep = self.get_basic_token()
        if seq_sep is not None:
            sequences = nltk.tokenize.sent_tokenize(text)
            sequences = [s for s in sequences if len(s) > 5]
            text = seq_sep.join(sequences)
            text = self.remove_punctuations(text)
        return start + " " + text + " " + end

    def preprocess_all(self, texts:list) -> list:
        """ Begins the pre-processing for a list of texts.

        Args:
            texts (list): All texts in which we want to apply the pre-process.

        Returns:
            list: Pre-processed texts
        """
        preprocessed_texts = [self.__preprocess_text(text) for text in texts]
        separated_texts = [self.separate_sequences(text) for text in preprocessed_texts]
        return separated_texts
