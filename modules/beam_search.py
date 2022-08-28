import numpy as np
import tensorflow as tf


class BeamSearch:
    def __init__(
        self,
        start_token,
        end_token,
        max_length,
        tokenizer,
        idx_to_word,
        word_to_idx,
        beam_index
    ):
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.beam_index = beam_index
        

    def get_word(self, idx):
        return self.idx_to_word.get(idx, None)
    
    def get_idx(self, word):
        return self.word_to_idx.get(word, -1)

    def beam_search_predict(self, model, image, tag, dataset='iuxray', multi_modal=False):
        start = [self.get_idx(self.start_token)]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < self.max_length:
            temp = []
            for s in start_word:
                par_caps = tf.keras.preprocessing.sequence.pad_sequences(
                    [s[0]], maxlen=self.max_length, padding="post"
                )
                if multi_modal:
                    if dataset=='iuxray':
                        preds = model.predict([image[0], image[1], tag, par_caps], verbose=0)
                    else:
                        preds = model.predict([image, tag, par_caps], verbose=0)
                else:
                    if dataset=='iuxray':
                        preds = model.predict([image[0], image[1], par_caps], verbose=0)
                    else:
                        preds = model.predict([image, par_caps], verbose=0)
                        
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

    def beam_search_predictions_ensemble(
        self, model1, model2, model3, model4, image1, image2, image3, image4
    ):
        start = [self.get_idx(self.start_token)]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < self.max_length:
            temp1, temp2, temp3, temp4 = [], [], [], []
            for s in start_word:
                par_caps = tf.keras.preprocessing.sequence.pad_sequences(
                    [s[0]], maxlen=self.max_length, padding="post"
                )
                preds1 = model1.predict([image1, par_caps], verbose=0)
                preds2 = model2.predict([image2, par_caps], verbose=0)
                preds3 = model3.predict([image3, par_caps], verbose=0)
                preds4 = model4.predict([image4, par_caps], verbose=0)
                word_preds1 = np.argsort(preds1[0])[-self.beam_index:]
                word_preds2 = np.argsort(preds2[0])[-self.beam_index:]
                word_preds3 = np.argsort(preds3[0])[-self.beam_index:]
                word_preds4 = np.argsort(preds4[0])[-self.beam_index:]
                # Getting the top <self.self.beam_index>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds1:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds1[0][w]
                    temp1.append([next_cap, prob])

                for w in word_preds2:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds2[0][w]
                    temp2.append([next_cap, prob])

                for w in word_preds3:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds3[0][w]
                    temp3.append([next_cap, prob])

                for w in word_preds4:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds4[0][w]
                    temp4.append([next_cap, prob])

            start_word1 = temp1
            # Sorting according to the probabilities
            start_word1 = sorted(
                start_word1, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word1 = start_word1[-self.beam_index:]

            start_word2 = temp2
            # Sorting according to the probabilities
            start_word2 = sorted(
                start_word2, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word2 = start_word2[-self.beam_index:]

            start_word3 = temp3
            # Sorting according to the probabilities
            start_word3 = sorted(
                start_word3, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word3 = start_word3[-self.beam_index:]

            start_word4 = temp4
            # Sorting according to the probabilities
            start_word4 = sorted(
                start_word4, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word4 = start_word4[-self.beam_index:]

            start_words = [
                start_word1[-1],
                start_word2[-1],
                start_word3[-1],
                start_word4[-1],
            ]
            start_word = sorted(start_words, reverse=False, key=lambda l: l[1])

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
