import numpy as np
# tensorflow imports
import tensorflow as tf
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model

class GreedySearch:
    def __init__(self, start_token:str, end_token:str, max_length:int, tokenizer:Tokenizer, idx_to_word:dict, word_to_idx:dict):
        """ The Greedy Search sampling method for generating captions.

        Args:
            start_token (str): The start-token used during pre-processing of the training captions
            end_token (str): The end-token used during pre-processing of the training captions
            max_length (int): The maximum length (limit) for the generated captions
            tokenizer (Tokenizer): The fitted tokenizer from the Vocabulary object
            idx_to_word (dict): Dictionary with keys to be the index number and values the words in the created vocabulary
            word_to_idx (dict): Dictionary with keys to be the words and values the index number in the created vocabulary  
        """
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

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

    def greedy_search_predict(self, model:Model, photo:np.array, tag:np.array, dataset:str='iuxray', multi_modal:bool=False)->str:
        """ Executes the greedy search algorithm, employing the pre-trained model along with the test instance's data.

        Args:
            model (Model): The model we want to evaluate on our employed dataset
            photo (np.array): Current test image embedding
            tag (np.array): The tag embedding for the current test instance. This is used only for IU X-Ray dataset.
            dataset (str, optional): The dataset we employed for the model. Defaults to 'iuxray'.
            multi_modal (bool, optional): If we want to use the multi-modal version of model. This is used only for IU X-Ray dataset. Defaults to False.

        Returns:
            str: The generated description for the given image
        """
        # seed the generation process
        in_text = self.start_token
        # iterate over the whole length of the sequence
        for i in range(self.max_length):
            # integer encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence], maxlen=self.max_length
            )
            # predict next word
            if multi_modal:
                if dataset=='iuxray':
                    yhat = model.predict([photo[0], photo[1], tag, sequence], verbose=0)
            else:
                if dataset=='iuxray':
                    yhat = model.predict([photo[0], photo[1], sequence], verbose=0)
                else:
                    yhat = model.predict([photo, sequence], verbose=0)
            
                
            # convert probability to integer
            yhat = np.argmax(yhat)
            # map integer to word
            word = self.get_word(yhat)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += " " + word
            # stop if we predict the end of the sequence
            if word == self.end_token:
                break
        return in_text

    def greedy_search_ensembles_AP(self, models:list, photos:list, tags:list, dataset:str='iuxray', multi_modal:bool=False)->str:
        """ Executes the Average Probability Greedy Search algorithm employing the pre-trained models along with the test instances data.
        More details are provided in my Thesis. Acknowledgements: https://ieeexplore.ieee.org/document/9031513

        Args:
            models (list): The models we want to evaluate on our employed dataset
            photos (list): Current test images embeddings for each encoder we used.
            tags (list): Current test tags embeddings for each encoder we used.
            dataset (str, optional): The dataset we employed for the model. Defaults to 'iuxray'.
            multi_modal (bool, optional): If we want to use the multi-modal version of model. This is used only for IU X-Ray dataset. Defaults to False.

        Returns:
            str: The generated description for the given image ID.
        """
        
        # seed the generation process
        in_text = self.start_token
        # iterate over the whole length of the sequence
        for i in range(self.max_length):
            # integer encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence], maxlen=self.max_length
            )
            # predict next word
            
            if multi_modal:
                if dataset=='iuxray':
                    yhats = [model.predict([photo[0], photo[1], tags, sequence], verbose=0) for model, photo in zip(models, photos)]
            else:
                if dataset=='iuxray':
                    yhats = [model.predict([photo[0], photo[1], sequence], verbose=0) for model, photo in zip(models, photos)]
                else:
                    yhats = [model.predict([photo, sequence], verbose=0) for model, photo in zip(models, photos)]

            # yhats = [
            #     model.predict([photo, sequence], verbose=0)
            #     for model, photo in zip(models, photos)
            # ]
            summed = np.sum(yhats, axis=0)
            # convert probability to integer
            yhat = np.argmax(summed, axis=1)

            # map integer to word
            word = self.get_word(yhat[0])

            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += " " + word
            # stop if we predict the end of the sequence
            if word == self.end_token:
                break

        return in_text

    def greedy_search_ensembles_MVP(self, models:list, photos:list, tags:list, dataset:str='iuxray', multi_modal:bool=False)->str:
        """ Executes the Maximum Voting Probability Greedy Search algorithm employing the pre-trained models along with the test instances data.
        More details are provided in my Thesis. Acknowledgements: https://ieeexplore.ieee.org/document/9031513

        Args:
            models (list): The models we want to evaluate on our employed dataset
            photos (list): Current test images embeddings for each encoder we used.
            tags (list): Current test tags embeddings for each encoder we used.
            dataset (str, optional): The dataset we employed for the model. Defaults to 'iuxray'.
            multi_modal (bool, optional): If we want to use the multi-modal version of model. This is used only for IU X-Ray dataset. Defaults to False.

        Returns:
            str: The generated description for the given image ID.
        """
        # seed the generation process
        in_text = self.start_token
        # iterate over the whole length of the sequence
        for i in range(self.max_length):
            pred = []
            index = 0
            for each_model in models:

                # integer encode input sequence
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                # pad input
                sequence = tf.keras.preprocessing.sequence.pad_sequences(
                    [sequence], maxlen=self.max_length
                )
                if multi_modal:
                    if dataset == 'iuxray':
                        yhat = each_model.predict([photos[index][0], photos[index][1], tags, sequence], verbose=0)
                else:
                    if dataset == 'iuxray':
                        yhat = each_model.predict([photos[index][0], photos[index][1], sequence], verbose=0)
                    else:
                        yhat = each_model.predict([photos[index], sequence], verbose=0)
                pred.append(np.argmax(yhat))
                index += 1

            # predict next word
            yhats = max(pred, key=pred.count)

            # map integer to word
            word = self.get_word(yhats)

            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += " " + word
            # stop if we predict the end of the sequence
            if word == self.end_token:
                break

        return in_text

    def greedy_search_ensembles_MP(self, models:list, photos:list, tags:list, dataset:str='iuxray', multi_modal:bool=False)->str:
        """ Executes the Maximum Probability Greedy Search algorithm employing the pre-trained models along with the test instances data.
        More details are provided in my Thesis. 

        Args:
            models (list): The models we want to evaluate on our employed dataset
            photos (list): Current test images embeddings for each encoder we used.
            tags (list): Current test tags embeddings for each encoder we used.
            dataset (str, optional): The dataset we employed for the model. Defaults to 'iuxray'.
            multi_modal (bool, optional): If we want to use the multi-modal version of model. This is used only for IU X-Ray dataset. Defaults to False.

        Returns:
            str: The generated description for the given image ID.
        """
        # seed the generation process
        in_text = self.start_token
        # iterate over the whole length of the sequence
        for i in range(self.max_length):
            pred = []
            max_value = []
            index = 0
            for each_model in models:

                # integer encode input sequence
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                # pad input
                sequence = tf.keras.preprocessing.sequence.pad_sequences(
                    [sequence], maxlen=self.max_length
                )
                if multi_modal:
                    if dataset == 'iuxray':
                        yhat = each_model.predict([photos[index][0], photos[index][1], tags, sequence], verbose=0)
                else:
                    if dataset == 'iuxray':
                        yhat = each_model.predict([photos[index][0], photos[index][1], sequence], verbose=0)
                    else:
                        yhat = each_model.predict([photos[index], sequence], verbose=0)
                max_value.append(np.amax(yhat))
                pred.append(np.argmax(yhat))
                index += 1

            # predict next word
            yhats = max(max_value)
            max_index = max_value.index(yhats)
            yhats = pred[max_index]

            # map integer to word
            word = self.get_word(yhats)

            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += " " + word
            # stop if we predict the end of the sequence
            if word == self.end_token:
                break

        return in_text
