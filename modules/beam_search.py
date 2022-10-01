# numpy import
import numpy as np

# tensorflow imports
import tensorflow as tf
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model


class BeamSearch:
    def __init__(self, start_token:str, end_token:str, max_length:int, tokenizer:Tokenizer, idx_to_word:dict, word_to_idx:dict, beam_index:int):
        """ The Beam Search sampling method for generating captions. An illustration of the algorithm is provided in my Thesis paper.

        Args:
            start_token (str): The start-token used during pre-processing of the training captions
            end_token (str): The end-token used during pre-processing of the training captions
            max_length (int): The maximum length (limit) for the generated captions
            tokenizer (Tokenizer): The fitted tokenizer from the Vocabulary object
            idx_to_word (dict): Dictionary with keys to be the index number and values the words in the created vocabulary
            word_to_idx (dict): Dictionary with keys to be the words and values the index number in the created vocabulary  
            beam_index (int): The beam size for the Beam Seach algorithm.
        """
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.beam_index = beam_index

    def get_word(self, idx:int) -> str:
        """ Fetches the word from the index-to-word vocab, which was created after the pre-processing of the Training captions

        Args:
            idx (int): The index for the index-to-word vocab.

        Returns:
            str: The word for the given index if exist in the created index-to-word vocab, else None
        """
        return self.idx_to_word.get(idx, None)

    def get_idx(self, word:str)->int:
        """ Fetches the index number from the word-to-index vocab, which was created after the pre-processing of the Training captions

        Args:
            word (str): The word for which we want its index in the word-to-index dictionary.

        Returns:
            int: The index for the given word if exist in the created word-to-index vocab, else -1. The latter number refer to None
        """
        return self.word_to_idx.get(word, -1)

    def beam_search_predict(self, model:Model, image:np.array, tag:np.array, dataset:str='iuxray', multi_modal:bool=False)->str:
        """ Executes the beam search algorithm employing the pre-trained model along with the test instance's data.

        Args:
            model (Model): The model we want to evaluate on our employed dataset
            image (np.array): Current test image embedding
            tag (np.array): The tag embedding for the current test instance. This is used only for IU X-Ray dataset.
            dataset (str, optional): The dataset we employed for the model. Defaults to 'iuxray'.
            multi_modal (bool, optional): If we want to use the multi-modal version of model. This is used only for IU X-Ray dataset. Defaults to False.

        Returns:
            str: The generated description for the given image using the beam search.
        """
        start = [self.get_idx(self.start_token)]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < self.max_length:
            # store current word,probs pairs
            temp = []
            # for current sequence
            for s in start_word:
                # pad the sequence in order to fetch the next token
                par_caps = tf.keras.preprocessing.sequence.pad_sequences([s[0]], maxlen=self.max_length, padding="post")
                if multi_modal:
                    if dataset == 'iuxray':
                        preds = model.predict(
                            [image[0], image[1], tag, par_caps], verbose=0)
                    else:
                        preds = model.predict(
                            [image, tag, par_caps], verbose=0)
                else:
                    if dataset == 'iuxray':
                        preds = model.predict(
                            [image[0], image[1], par_caps], verbose=0)
                    else:
                        preds = model.predict([image, par_caps], verbose=0)
                # get the best paths
                word_preds = np.argsort(preds[0])[-self.beam_index:]

                # Getting the top <self.self.beam_index>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-self.beam_index:]

        # get the best path
        start_word = start_word[-1][0]
        intermediate_caption = [self.get_word(i) for i in start_word]
        final_caption = []

        for i in intermediate_caption:
            if i != self.end_token:
                final_caption.append(i)
            else:
                break

        final_caption = " ".join(final_caption[1:])
        return final_caption

    def ensemble_beam_search(self, models:list, images_list:list)->str:
        """ Executes the beam search algorithm employing the pre-trained models along with the test instances data.
        This utilises the beam search algorithm for each model in models list.

        Args:
            models (list): The models we want to evaluate on our employed dataset
            images_list (list): Current test images embeddings for each encoder we used.

        Returns:
            str: The generated description for the given image using the beam search.
        """
        start = [self.get_idx(self.start_token)]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < self.max_length:
            # for current seq
            for s in start_word:
                # pad current caption
                current_caption = tf.keras.preprocessing.sequence.pad_sequences([s[0]], maxlen=self.max_length, padding="post")
                # get all predictions from the pre-trained models
                ensemble_predictions = [ ensemble_member.predict([image, current_caption], verbose=0) for ensemble_member, image in zip(models, images_list) ]
                # get the best pairs
                ensemble_word_predictions = [ np.argsort(prediction[0])[-self.beam_index:] for prediction in ensemble_predictions ]

                # and store them with word,pairs for each ensemble member
                ensemble_current_probs = list()
                for member_prediction, member_word_predictions in zip(ensemble_predictions, ensemble_word_predictions):
                    temp_current_seq = list()

                    for word in member_word_predictions:
                        next_cap, prob = s[0][:], s[1]
                        next_cap.append(word)
                        prob += member_prediction[0][word]
                        temp_current_seq.append([next_cap, prob])

                    ensemble_current_probs.append(temp_current_seq)
            # get all the best candidates
            ensemble_starting_words = [
                sorted(current_probs, reverse=False, key=lambda l: l[1])[-self.beam_index:] for current_probs in ensemble_current_probs
            ]

            start_words = [member_starting_words[-1] for member_starting_words in ensemble_starting_words]
            start_word = sorted(start_words, reverse=False, key=lambda l: l[1])

        # create the caption
        start_word = start_word[-1][0]
        intermediate_caption = [self.get_word(i) for i in start_word]
        final_caption = []

        for i in intermediate_caption:
            if i != self.end_token:
                final_caption.append(i)
            else:
                break

        final_caption = " ".join(final_caption[1:])
        return final_caption
