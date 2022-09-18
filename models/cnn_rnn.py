import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dropout,
    Dense,
    Conv2D,
    MaxPooling2D,
    RepeatVector,
    GlobalMaxPooling2D,
    TimeDistributed,
    Bidirectional,
    
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tqdm import tqdm
from utils.text_handler import TextHandler

from modules import BeamSearch, GreedySearch
from utils.dataset import Dataset

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle

MODELS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TAGS_EMBEDDINGS_PATH = os.path.join(os.path.dirname(MODELS_DIR_PATH), 'data')

# this class is based on the Show and Tell model.
class CNN_RNN:
    def __init__(
        self,
        tokenizer,
        word_to_idx,
        idx_to_word,
        max_length: int,
        embedding_dim: int,
        ling_model:str, 
        multi_modal: bool,
        loss="categorical_crossentropy"
    ):
        # self.dataset = dataset
        # self, self.tokenizer, self.word2idx, self.idx2word = self.dataset.vocab, self.dataset.tokenizer, self.dataset.word2idx, self.dataset.idx2word
        self.tokenizer, self.word2idx, self.idx2word = tokenizer, word_to_idx, idx_to_word
        self.max_length = max_length

        self.vocab_size = len(self.word2idx)

        self.multi_modal = multi_modal

        self.encoder = Encoder()
        self.embedding = EmbeddingLayer(embedding_dim, self.vocab_size, linguistic_model=ling_model)
        self.decoder = Decoder(max_length, self.vocab_size, linguistic_model=ling_model)
        self.loss = loss
        (
            self.start_token,
            self.end_token,
            self.seq_sep,
        ) = TextHandler().get_basic_token()
        
    
    def __load_embeddings(self):
        npy_file = os.path.join(TAGS_EMBEDDINGS_PATH, 'fasttext.npy')
        fastttext_voc_file = os.path.join(TAGS_EMBEDDINGS_PATH, 'fasttext_voc.pkl')
        fasttext_embed = np.load(npy_file)
        fasttext_word_to_index = pickle.load(open(fastttext_voc_file, 'rb'))
        return fasttext_embed, fasttext_word_to_index
    
    def __make_multimodal_weights(self, fasttext_embed, fasttext_word_to_index, max_tags, embedding_dim, word_index):
        embedding_matrix = np.zeros((max_tags+2, embedding_dim))  # +2 (pad, unknown)

        for word, i in word_index.items():
            if i > max_tags:
                    continue
            try:
                embedding_vector = fasttext_embed[fasttext_word_to_index[word],:]
                embedding_matrix[i] = embedding_vector
            except:
                pass
        return embedding_matrix
    
    def __tokenize_tags(self, tags, max_tags, max_sequence_length):
         # Init tokenizer
        tokenizer = Tokenizer(num_words=max_tags, oov_token='__UNK__')
        word_index = tokenizer.word_index

        # Fit tokenizer
        tokenizer.fit_on_texts(list(tags.values()))

        tag_patient_pair = {}
        for key,value in tags.items():
            tag_data = tokenizer.texts_to_sequences([value])
            tag_data = pad_sequences(sequences=tag_data, maxlen=max_sequence_length, padding='post')
            tag_patient_pair[key] = tag_data
        
        return tag_patient_pair, tokenizer, word_index
    
    def build_multimodal_encoder(self, tags):
        
        
        values = list(tags.values())
        print('build_multimodal_encoder', len(values))
        _tags = []
        max_sequence_tags = []
        for i in range(len(values)):
            max_sequence_tags.append(len(values[i]))
            for word in values[i]:
                _tags.append(word)
                
        max_tags = len(set(_tags))
        max_sequence_length = max(max_sequence_tags) 
        fasttext_embed, fasttext_word_to_index = self.__load_embeddings()
        
        tag_patient_pair, _, word_index = self.__tokenize_tags(tags, max_tags, max_sequence_length)
    
        embedding_dim = fasttext_embed.shape[1]
        
        embedding_matrix = self.__make_multimodal_weights(fasttext_embed, fasttext_word_to_index, max_tags, embedding_dim, word_index)
        
        self.tag_encoder = TagEncoder(
                pretrained=True,
                embedding_dim=embedding_dim,
                vocab_size=self.vocab_size,
                max_tags=max_tags,
                max_sequence=max_sequence_length,
                weights=[embedding_matrix], 
                max_length=self.max_length,
                dropout_rate=0.2,
            )
        
        return tag_patient_pair

    def __make_imageclef_model(self, input_shape, optimizer):
        # images
        input_image1 = Input(shape=input_shape, name='Input image')
        output_image1 = self.encoder.encode_(input_image1)
        output_image1 = RepeatVector(self.max_length)(output_image1)

        # embeddings
        input2 = Input(shape=(self.max_length,), name='Input caption words')
        output2 = self.embedding.get_embedding(input2)

        # decoder
        decoder_input = concatenate([output_image1, output2])
                

        outputs = self.decoder.decode_(decoder_input)

        # define model
        model = Model(inputs=[input_image1, input2], outputs=outputs)

        model.compile(loss=self.loss, optimizer=optimizer,
                      metrics=["accuracy"])
        return model
    
    def __make_iuxray_model(self, input_shape, optimizer):
        # images
        input_image1 = Input(shape=input_shape, name='Input image 1')
        output_image1 = self.encoder.encode_(input_image1)
        
        input_image2 = Input(shape=input_shape, name='Input image 2')
        output_image2 = self.encoder.encode_(input_image2)
        
        image_features = concatenate([output_image1, output_image2])
        image_features = RepeatVector(self.max_length)(image_features)
        
        # multimodal tags
        if self.multi_modal:
            inputTag = Input(shape=(self.tag_encoder.max_sequence,), name='Input tags')
            outputTag = self.tag_encoder.tag_encode_(inputTag)
            

        # embeddings
        input2 = Input(shape=(self.max_length,), name='Input caption words')
        output2 = self.embedding.get_embedding(input2)
        
        # decoder
        if self.multi_modal:
            decoder_input = concatenate([image_features, outputTag, output2])
        else:
            decoder_input = concatenate([image_features, output2])
                
        outputs = self.decoder.decode_(decoder_input)

        # define model
        if self.multi_modal:
            model = Model(inputs=[input_image1, input_image2, inputTag, input2], outputs=outputs)
        else:
            model = Model(inputs=[input_image1, input_image2, input2], outputs=outputs)

        model.compile(loss=self.loss, optimizer=optimizer,
                      metrics=["accuracy"])
        return model
    
    
    def __create_imageclef_sequences(self, caption, image):
        Ximages, XSeq, y = list(), list(), list()

        # integer encode the description
        seq = self.tokenizer.texts_to_sequences([caption])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # select
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = tensorflow.keras.preprocessing.sequence.pad_sequences(
                [in_seq], maxlen=self.max_length
            )[0]
            # encode output sequence
            out_seq = tensorflow.keras.utils.to_categorical(
                [out_seq], num_classes=self.vocab_size
            )[0]
            # store
            Ximages.append(image)
            XSeq.append(in_seq)
            y.append(out_seq)
        return [Ximages, XSeq, y]
    
    def __create_iuxray_sequences(self, caption, image1, image2, tag):
        Ximages1, Ximages2, Xtags, XSeq, y = list(), list(), list(), list(), list()

        # integer encode the description
        seq = self.tokenizer.texts_to_sequences([caption])[0]
        
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # select
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = tensorflow.keras.preprocessing.sequence.pad_sequences(
                [in_seq], maxlen=self.max_length
            )[0]
            
            # encode output sequence
            out_seq = tensorflow.keras.utils.to_categorical(
                [out_seq], num_classes=self.vocab_size
            )[0]
            # store
            Ximages1.append(image1)
            Ximages2.append(image2)
            Xtags.append(tag)
            XSeq.append(in_seq)
            y.append(out_seq)
            
        if self.multi_modal:
            return [Ximages1, Ximages2, Xtags, XSeq, y]
        
        return [Ximages1, Ximages2, XSeq, y]
        

    def __imageclef_data_generator(
        self, captions, image_tuples, n_step, validation, validation_num):
        while True:
            patients = list(captions.keys())
            if validation:
                assert validation_num > 0
                patients = patients[-validation_num:]
            elif not validation:
                if validation_num > 0:
                    patients = patients[:-validation_num]

            for i in range(0, len(patients), n_step):
                Ximages, XSeq, y = list(), list(), list()
                for j in range(i, min(len(patients), i + n_step)):
                    patient_id = patients[j]
                    # retrieve text input
                    caption = captions[patient_id]
                    # generate input-output pairs (many images in each batch)
                    img = image_tuples[patient_id][0]
                    in_img1, in_seq, out_word = self.__create_imageclef_sequences(
                        caption, img)
                    for k in range(len(in_img1)):
                        Ximages.append(in_img1[k])
                        XSeq.append(in_seq[k])
                        y.append(out_word[k])
                yield [np.array(Ximages), np.array(XSeq)], np.array(y)
    
    def __iuxray_data_generator(
        self, captions, image_tuples, tags, n_step, validation, validation_num):
        while True:
            patients = list(captions.keys())
            if validation:
                assert validation_num>0
                patients = patients[-validation_num:]
            elif not validation:
                if validation_num>0:
                    patients = patients[:-validation_num]
                
            for i in range(0, len(patients), n_step):
                Ximages1, Ximages2, Xtags, XSeq, y = list(), list(), list(), list(),list()
                for j in range(i, min(len(patients), i+n_step)):
                    patient_id = patients[j]
                    # retrieve text input
                    caption = captions[patient_id]
                    
                    # generate input-output pairs (many images in each batch)
                    img1 = image_tuples[patient_id][0][0]
                    img2 = image_tuples[patient_id][1][0]
                    tag = tags[patient_id][0]
                    if self.multi_modal:
                        in_img1, in_img2, in_tag, in_seq, out_word = self.__create_iuxray_sequences(
                            caption, img1, img2, tag)
                        for k in range(len(in_img1)):
                            Ximages1.append(in_img1[k])
                            Ximages2.append(in_img2[k])
                            Xtags.append(in_tag[k])
                            XSeq.append(in_seq[k])
                            y.append(out_word[k])
                    else:
                        in_img1, in_img2, in_seq, out_word = self.__create_iuxray_sequences(
                            caption, img1, img2, tag)
                        for k in range(len(in_img1)):
                            Ximages1.append(in_img1[k])
                            Ximages2.append(in_img2[k])
                            XSeq.append(in_seq[k])
                            y.append(out_word[k])
                    # yield this batch of samples to the model
                    #print (array(Ximages1).shape)
                if self.multi_modal:
                    yield [np.array(Ximages1), np.array(Ximages2), np.array(Xtags), np.array(XSeq)], np.array(y)
                else:
                    yield [np.array(Ximages1), np.array(Ximages2), np.array(XSeq)], np.array(y)

    def train_imageclef_model(self, train_data, input_shape, optimizer, model_name, n_epochs, batch_size):
        verbose = 1

        val_len = int(0.05 * len(train_data[1]))
        train_len = len(train_data[1])
        train_steps = int(train_len / batch_size)
        val_steps = int(val_len / batch_size)
        model = self.__make_imageclef_model(input_shape, optimizer)
        print('val_steps:', val_steps)
        print('steps_per_epoc:', val_steps)
        
        train_gen = self.__imageclef_data_generator(
            train_data[1],
            train_data[0],
            batch_size,
            validation=False,
            validation_num=val_len,
        )
        val_gen = self.__imageclef_data_generator(
            train_data[1],
            train_data[0],
            batch_size,
            validation=True,
            validation_num=val_len,
        )

        early = tensorflow.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )

        # reduce learning rate
        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=1, verbose=1, mode="min"
        )
        # callbacks
        cs = [early, reduce_lr]

        model.fit_generator(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            epochs=n_epochs,
            verbose=verbose,
            callbacks=cs,
        )

        saved_models_path = (
            "/home/cave-of-time/panthro/panthro/imageclef2022/saved_models/"
        )
        model.save(saved_models_path + model_name + ".h5")
        return model
    
    def train_iuxray_model(self, train_data, input_shape, optimizer, model_name, n_epochs, batch_size):
        verbose = 1

        val_len = int(0.05 * len(train_data[1]))
        train_len = len(train_data[1])
        train_steps = int(train_len / batch_size)
        val_steps = int(val_len / batch_size)
        print('val_steps:', val_steps)
        print('steps_per_epoc:', train_steps)
        model = self.__make_iuxray_model(input_shape, optimizer)
        train_gen = self.__iuxray_data_generator(
            train_data[1],
            train_data[0],
            train_data[2],
            batch_size,
            validation=False,
            validation_num=val_len,
        )
        val_gen = self.__iuxray_data_generator(
            train_data[1],
            train_data[0],
            train_data[2],
            batch_size,
            validation=True,
            validation_num=val_len,
        )

        early = tensorflow.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=5,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )

        # reduce learning rate
        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=1, verbose=1, mode="min"
        )
        # callbacks
        cs = [early, reduce_lr]

        model.fit(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            epochs=n_epochs,
            verbose=verbose,
            callbacks=cs,
        )
        
        model.save(model_name + ".h5")
        return model

    
    def evaluate_ensemble_model(self, models, test_captions, test_images, test_tags, dataset:str='iu_xray', ensemble_method: str = 'AP'):
        gold, predicted = {}, {}
        
        gs = GreedySearch(start_token=self.start_token, end_token=self.end_token, max_length=self.max_length,
                            tokenizer=self.tokenizer, idx_to_word=self.idx2word, word_to_idx=self.word2idx)
        for key in tqdm(test_images[0]):
            test_photos = [test_images[i][key] for i in range(len(test_images))]
            current_tags = test_tags[key]
            if ensemble_method == 'AP':
                caption = gs.greedy_search_ensembles_AP(models, test_photos, current_tags, dataset=dataset, multi_modal=self.multi_modal)
            elif ensemble_method == 'MVP':
                caption = gs.greedy_search_ensembles_MVP(models, test_photos, current_tags, dataset=dataset, multi_modal=self.multi_modal)
            elif ensemble_method == 'MP':
                caption = gs.greedy_search_ensembles_MP(models, test_photos, current_tags, dataset=dataset, multi_modal=self.multi_modal)
                
                

            predicted[key] = self.remove_basic_tokens(caption)
            gold[key] = test_captions[key]

        return gold, predicted

    def evaluate_model(self, model, test_captions, test_images, test_tags, eval_dataset:str='iu_xray', evaluator_choice: str = 'beam_10'):
        gold, predicted = {}, {}
        if evaluator_choice.split('_')[0] == 'beam':
            bs = BeamSearch(start_token=self.start_token, end_token=self.end_token, max_length=self.max_length,
                            tokenizer=self.tokenizer, idx_to_word=self.idx2word, word_to_idx=self.word2idx, beam_index= int(evaluator_choice.split('_')[1]) )
            for key in tqdm(test_images, desc='Evaluating model..'):
                if eval_dataset == 'iu_xray':
                    caption = bs.beam_search_predict(
                        model, test_images[key], test_tags[key], dataset=eval_dataset, multi_modal=self.multi_modal)
                else:
                    caption = bs.beam_search_predict(
                        model, test_images[key], None, dataset=eval_dataset, multi_modal=self.multi_modal)

                predicted[key] = self.remove_basic_tokens(caption)
                gold[key] = test_captions[key]
        else:
            gs = GreedySearch(start_token=self.start_token, end_token=self.end_token, max_length=self.max_length,
                            tokenizer=self.tokenizer, idx_to_word=self.idx2word, word_to_idx=self.word2idx)
            for key in tqdm(test_images, desc='Evaluating model..'):
                if eval_dataset == 'iu_xray':
                    caption = gs.greedy_search_predict(
                        model, test_images[key], tag=test_tags[key], dataset=eval_dataset, multi_modal=self.multi_modal)
                else:
                    caption = gs.greedy_search_predict(
                    model, test_images[key], tag=None, dataset=eval_dataset, multi_modal=self.multi_modal)

                predicted[key] = self.remove_basic_tokens(caption)
                gold[key] = test_captions[key]

        return gold, predicted

    def remove_basic_tokens(self, caption):
        caption = caption.replace(self.start_token, " ").replace(
            self.end_token, " ").replace("<pad>", " ").replace("<unk>", " ").replace(self.seq_sep, '. ')
        return caption


class Encoder:
    def __init__(
        self,
        dropout_rate: float = 0.2,
        dense_activation="relu"):
        self.activation_function = dense_activation
        self.fe = Dense(256, activation=self.activation_function)

        self.avg_pool = GlobalMaxPooling2D()
        self.dropout = Dropout(dropout_rate)

    def encode_(self, img):
        # img = self.avg_pool(img)        ## for CotNet and BEiT only
        img = self.dropout(img)
        img = self.fe(img)
        return img


class TagEncoder:
    def __init__(
        self,
        pretrained: bool,
        embedding_dim: int,
        vocab_size:int,
        max_tags: int,
        max_sequence:int, 
        weights:list, 
        max_length: int,
        dropout_rate: float,
        dense_activation="relu",
    ):
        self.max_length = max_length
        self.max_sequence = max_sequence
        self.activation_function = dense_activation
        if pretrained:
            self.embedding = Embedding(
                input_dim=max_tags + 2,
                output_dim=embedding_dim,
                weights=weights,
                mask_zero=True,
                trainable=False,
            )
        else:
            self.embedding = Embedding(
                vocab_size, embedding_dim, mask_zero=True)
        self.ling_model = GRU(256, return_sequences=False)
        # self.dropout = Dropout(dropout_rate)
        self.fe = Dense(256, activation=self.activation_function)
        self.rep_vec = RepeatVector(self.max_length)

    def tag_encode_(self, tag):
        tag = self.embedding(tag)
        # tag = self.dropout(tag)
        tag = self.ling_model(tag)
        tag = self.fe(tag)
        tag = self.rep_vec(tag)
        return tag


class EmbeddingLayer:
    def __init__(self, embedding_dim, vocab_size, linguistic_model="gru"):
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        if linguistic_model == "gru":
            self.ling_model1 = GRU(256, return_sequences=True)
            self.ling_model2 = GRU(256, return_sequences=True)
        elif linguistic_model == "bigru":
            self.ling_model1 = Bidirectional(GRU(256, return_sequences=True))
            self.ling_model2 = Bidirectional(GRU(256, return_sequences=True))
        else:
            self.ling_model1 = LSTM(256, return_sequences=True)
            self.ling_model2 = LSTM(256, return_sequences=True)
        self.time_dis = TimeDistributed(Dense(256, activation="relu"))

    def get_embedding(self, x):
        x = self.embedding(x)
        x = self.ling_model1(x)
        x = self.ling_model2(x)
        x = self.time_dis(x)
        return x


class Decoder:
    def __init__(self, max_length, vocab_size, dense_activation="relu", linguistic_model:str='gru'):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.activation_function = dense_activation
        if linguistic_model=='gru':
            self.decoder = GRU(256)
        elif linguistic_model=='bigru':
            self.decoder = Bidirectional(GRU(256))
        else :
            self.decoder = LSTM(256)
        self.fc = Dense(256, activation=self.activation_function)
        self.out = Dense(self.vocab_size, activation="softmax",
                         name="output_layer")

    def decode_(self, x):
        x = self.decoder(x)
        x = self.fc(x)
        x = self.out(x)
        return x
