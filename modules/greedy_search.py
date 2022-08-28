import numpy as np
import tensorflow as tf


class GreedySearch:
    def __init__(
        self, start_token, end_token, max_length, tokenizer, idx_to_word, word_to_idx
    ):
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

    def get_word(self, idx):
        return self.idx_to_word.get(idx, None)

    def get_idx(self, word):
        return self.word_to_idx.get(word, -1)

    # generate a description for an image
    def greedy_search_predict(self, model, photo, tag, dataset:str='iuxray', multi_modal:bool=False):

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

    # generate a description for an image

    def greedy_search_ensembles_AP(self, models, photos, tags, dataset:str='iuxray', multi_modal:bool=False):
        
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

    def greedy_search_ensembles_MVP(self, models, photos, tags, dataset:str='iuxray', multi_modal:bool=False):
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

    def greedy_search_ensembles_MP(self, models, photos, tags, dataset:str='iuxray', multi_modal:bool=False):
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
