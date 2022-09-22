# OS imports
import os
from tokenize import Token
from typing import Generator
import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# progress bar, numpy and pickle imports
import pickle
from tqdm import tqdm
import numpy as np

# Tensorflow imports
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
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
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer

# utils imports
from utils.dataset import Dataset
from modules import BeamSearch, GreedySearch
from utils.text_handler import TextHandler

# fetch the directories paths
MODELS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TAGS_EMBEDDINGS_PATH = os.path.join(os.path.dirname(MODELS_DIR_PATH), 'data')


class CNN_RNN:
    def __init__(self, tokenizer:Tokenizer, word_to_idx:dict, idx_to_word:dict, max_length: int, embedding_dim: int, 
    ling_model: str, multi_modal: bool, loss="categorical_crossentropy"):
        """ This class is based on the Show and Tell model. However, unlike the Show and Tell, the image is given in each time step to the Decoder module.
            Acknowledgements: Part of this code is based on https://machinelearningmastery.com/develop-a-caption-generation-model-in-keras/
            It is also worth emphasising that out of 10 teams, I managed to rank 2nd on the primary evaluation metric using this CNN-RNN architecture!
            More details can be found at my thesis paper as well as AUEB's NLP Group publication (http://ceur-ws.org/Vol-3180/paper-101.pdf)

        Args:
            tokenizer (Tokenizer): The Keras Tokenizer that was fitted on the training captions to create the Vocabulary.
            word_to_idx (dict): Dictionary with keys to be the words and values the index number in the created vocabulary
            idx_to_word (dict): Dictionary with keys to be the index number and values the words in the created vocabulary
            max_length (int): The maximum sequence length to be used for the captions.
            embedding_dim (int): The dimensions of the employed Embedding Layer.
            ling_model (str): The linguistic model to be used for RNN. (e.g., LSTM, GRU, Bi-GRU, etc.)
            multi_modal (bool): Flag for multi-modal activation. If True, it is going to be used only for IU X-Ray dataset, utilizing the patients' tags as well.
            loss (str, optional): The loss for training. Defaults to "categorical_crossentropy".
        """
        self.tokenizer, self.word2idx, self.idx2word = tokenizer, word_to_idx, idx_to_word
        self.max_length = max_length

        # initialiase the size of vocabulary that our FFNN will use to output the next word token.
        self.vocab_size = len(self.word2idx)

        self.multi_modal = multi_modal

        # initialise the Image Encoder, Text Encoder as well as the Decoder modules.
        self.encoder = Encoder()
        self.embedding = EmbeddingLayer(
            embedding_dim, self.vocab_size, linguistic_model=ling_model)
        self.decoder = Decoder(max_length, self.vocab_size,
                               linguistic_model=ling_model)
        self.loss = loss
        # fetch the start, end and sequence separator tokens.
        self.start_token, self.end_token, self.seq_sep= TextHandler().get_basic_token()

    def __load_embeddings(self) -> tuple[np.array, dict]:
        """ Private method to load word embeddings from the FastText model. (https://fasttext.cc/docs/en/crawl\-vectors.html)

        Returns:
            tuple[np.array, dict]: The fast text embeddings matrix and the fast text dictionary
        """
        npy_file = os.path.join(TAGS_EMBEDDINGS_PATH, 'fasttext.npy')
        fastttext_voc_file = os.path.join(
            TAGS_EMBEDDINGS_PATH, 'fasttext_voc.pkl')
        fasttext_embed = np.load(npy_file)
        fasttext_word_to_index = pickle.load(open(fastttext_voc_file, 'rb'))
        return fasttext_embed, fasttext_word_to_index

    def __make_multimodal_weights(self, fasttext_embed:np.array, fasttext_word_to_index:dict, max_tags:int, embedding_dim:int, word_index:dict) -> np.array:
        """ Private method that initialise the pre-trained weights that will be used to the Embeddings Layers of both Text and Tag Encoders.

        Args:
            fasttext_embed (np.array): The fast text embeddings matrix loaded from __load_embeddings function
            fasttext_word_to_index (dict): The fast text dictionary loaded from __load_embeddings function
            max_tags (int): The number of unique tags from all the patients available
            embedding_dim (int): The dimension of the embedding layer to be used.
            word_index (dict): Tags dictionary

        Returns:
            np.array: The weights for the Embeddings Layers of both Text and Tag Encoders.
        """
        embedding_matrix = np.zeros((max_tags+2, embedding_dim))  # +2 (pad, <UNK>)

        for word, i in word_index.items():
            if i > max_tags:
                continue
            try:
                # if tag exist, fetch its embedding
                embedding_vector = fasttext_embed[fasttext_word_to_index[word], :]
                embedding_matrix[i] = embedding_vector
            except:
                pass
        return embedding_matrix

    def __tokenize_tags(self, tags:dict, max_tags:int, max_sequence_length:int) -> tuple[dict, Tokenizer, dict]:
        """ Gets all tags and tokenize them using the Keras Tokenizer. It also creates the tags vocab.

        Args:
            tags (dict): Dictionary with keys to be the patients' ids and values the tags.
            max_tags (int): The number of unique tags from all the patients available
            max_sequence_length (int): The maximum tags occurence from all the patients available

        Returns:
            tuple[dict, Tokenizer, dict]: A dictionary with keys to be the patients' ids and values the tokenized tags, the fitted tokenizer and the tags vocab.
        """
        # Init tokenizer
        tokenizer = Tokenizer(num_words=max_tags, oov_token='__UNK__')
        word_index = tokenizer.word_index

        # Fit tokenizer
        tokenizer.fit_on_texts(list(tags.values()))

        # fetch tags per patient.
        tag_patient_pair = {}
        for key, value in tags.items():
            tag_data = tokenizer.texts_to_sequences([value])
            # add pad depending on the maximum occuence in all patients.
            tag_data = pad_sequences(sequences=tag_data, maxlen=max_sequence_length, padding='post')
            tag_patient_pair[key] = tag_data

        return tag_patient_pair, tokenizer, word_index

    def build_multimodal_encoder(self, tags:dict) -> dict:
        """ Builds the multi-modal Tag Encoder module (only for IU X-Ray dataset) if the multi-modal = True.

        Args:
            tags (dict): Dictionary with keys to be the patients' ids and values the tags.

        Returns:
            dict: A dictionary with keys to be the patients' ids and values the tokenized tags, to be feeded in the Tag Encoder.
        """
        # fetch tags
        values = list(tags.values())
        _tags = []
        max_sequence_tags = []
        for i in range(len(values)):
            max_sequence_tags.append(len(values[i]))
            for word in values[i]:
                _tags.append(word)

        # set max number of unique tags and the maximum lenght of tags from all patients available.
        max_tags = len(set(_tags))
        max_sequence_length = max(max_sequence_tags)
        # load FastText embeddings
        fasttext_embed, fasttext_word_to_index = self.__load_embeddings()

        # tokenize the tags
        tag_patient_pair, _, word_index = self.__tokenize_tags(
            tags, max_tags, max_sequence_length)

        # create weights
        embedding_dim = fasttext_embed.shape[1]
        embedding_matrix = self.__make_multimodal_weights(
            fasttext_embed, fasttext_word_to_index, max_tags, embedding_dim, word_index)

        # then build the Tag Encoder using the trainable weights from FastText.
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

    def __make_imageclef_model(self, input_shape:tuple, optimizer:tensorflow.keras.optimizers) -> Model:
        """ Private method that creates the keras CNN-RNN model for ImageCLEF dataset.

        Args:
            input_shape (tuple): The input shape for the image encoder (e.g., (1920,), (1024,), etc.). Inputs extraced from the last average pooling layer of the image encoder.
            optimizer (tensorflow.keras.optimizers): The keras oprimizer to be used during training (e.g., Adam, AdamW, etc.)

        Returns:
            Model: The created model (architecture)
        """
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

        # compile model
        model.compile(loss=self.loss, optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    def __make_iuxray_model(self, input_shape:tuple, optimizer:tensorflow.keras.optimizers):
        """ Private method that creates the keras CNN-RNN model for IU X-Ray dataset.
        It also handles the Multi-modal version

        Args:
            input_shape (tuple): The input shape for the image encoder (e.g., (1920,), (1024,), etc.). Inputs extraced from the last average pooling layer of the image encoder.
            optimizer (tensorflow.keras.optimizers): The keras oprimizer to be used during training (e.g., Adam, AdamW, etc.)

        Returns:
            Model: The created model (architecture)
        """
        # images
        input_image1 = Input(shape=input_shape, name='Input image 1')
        output_image1 = self.encoder.encode_(input_image1)

        input_image2 = Input(shape=input_shape, name='Input image 2')
        output_image2 = self.encoder.encode_(input_image2)

        # concat patient's images (later and frontal)
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
            model = Model(inputs=[input_image1, input_image2,
                          inputTag, input2], outputs=outputs)
        else:
            model = Model(
                inputs=[input_image1, input_image2, input2], outputs=outputs)
        # compile model
        model.compile(loss=self.loss, optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    def __create_imageclef_sequences(self, caption:str, image:np.array) -> list:
        """ Method that creates the image caption sequence pairs for ImageCLEF dataset like the following steps:
        Each description will be split into words. The model will be provided one word and the photo and generate the next word. 
        Then the first two words of the description will be provided to the model as input with the image to generate the next word. 
        This is how the model will be trained. 
        For example, the input sequence “little girl running in field” would be split into 6 input-output pairs to train the model:
        X1,		X2 (text sequence), 						y (word)
        photo	startseq, 									little
        photo	startseq, little,							girl
        photo	startseq, little, girl, 					running
        photo	startseq, little, girl, running, 			in
        photo	startseq, little, girl, running, in, 		field
        photo	startseq, little, girl, running, in, field, endseq


        Args:
            caption (str): Current caption to be used to create image, caption (words) pairs
            image (np.array): The image emmbedings extracted from employed image encoder (i.e. DenseNet121, EfficientNetB0, etc.)

        Returns:
            list: All image, sequences , and next word token pairs in a list
        """
        # init the lists to store current pairs
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

    def __create_iuxray_sequences(self, caption:str, image1:np.array, image2:np.array, tag:np.array):
        """ Method that creates the image caption sequence pairs for IU X-Ray dataset like the following steps:
        Each description will be split into words. The model will be provided one word and the photo and generate the next word. 
        Then the first two words of the description will be provided to the model as input with the image to generate the next word. 
        This is how the model will be trained. 
        For example, the input sequence “little girl running in field” would be split into 6 input-output pairs to train the model:
        X1,		X2 (text sequence), 						y (word)
        photo	startseq, 									little
        photo	startseq, little,							girl
        photo	startseq, little, girl, 					running
        photo	startseq, little, girl, running, 			in
        photo	startseq, little, girl, running, in, 		field
        photo	startseq, little, girl, running, in, field, endseq
        

        Args:
            caption (str): Current caption to be used to create image, caption (words) pairs
            image1 (np.array): The first image emmbedings extracted from employed image encoder (i.e. DenseNet121, EfficientNetB0, etc.)
            image2 (np.array): The second image emmbedings extracted from employed image encoder (i.e. DenseNet121, EfficientNetB0, etc.)
            tag (np.array): Current patient's tags.
        Returns:
            list: All image, sequences , and next word token pairs in a list
        """
        # init the lists to store current pairs. It is worth noting that in IU X-Ray dataset, each patient has 2 images.
        Ximages1, Ximages2, Xtags, XSeq, y = list(), list(), list(), list(), list()

        # integer encode the description
        seq = self.tokenizer.texts_to_sequences([caption])[0]

        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # select
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = tensorflow.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.max_length)[0]

            # encode output sequence
            out_seq = tensorflow.keras.utils.to_categorical( [out_seq], num_classes=self.vocab_size)[0]

            # store
            Ximages1.append(image1)
            Ximages2.append(image2)
            Xtags.append(tag)
            XSeq.append(in_seq)
            y.append(out_seq)

        # if multi-modal store also the tags.
        if self.multi_modal:
            return [Ximages1, Ximages2, Xtags, XSeq, y]

        return [Ximages1, Ximages2, XSeq, y]

    def __imageclef_data_generator(self, captions:dict, image_tuples:dict, n_step:int, validation:bool, validation_num:int) -> Generator[tuple[list,np.array], None, None]:
        """ Method that yields batches of input-output pairs when asked. It stores all input-output pairs in memory for the whole dataset.

        Args:
            captions (dict): Dictionary with keys to be the patients' ids and values the captions.
            image_tuples (dict): Dictionary with keys to be the patients' ids and values the image features.
            n_step (int): Allows us to tune how many images worth of input-output pairs to generate for each batch.
            validation (bool): If to generate a development data set for validate the model accuracy.
            validation_num (int): Allows us to tune how many images worth of input-output pairs to generate for development set.

        Yields:
            Generator[tuple[list,np.array], None, None]: A list with input images in first index and current text sequence in the second index, 
            and the output y in np.array format.
        """
        while True:
            # fetch patients ids
            patients = list(captions.keys())
            # if for development fetch a small batch from training.
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
                    # generate the all input image-text sequences
                    in_img1, in_seq, out_word = self.__create_imageclef_sequences(caption, img)
                    for k in range(len(in_img1)):
                        Ximages.append(in_img1[k])
                        XSeq.append(in_seq[k])
                        y.append(out_word[k])
                # store to memory and yield when called.
                yield [np.array(Ximages), np.array(XSeq)], np.array(y)

    def __iuxray_data_generator(self, captions:dict, image_tuples:dict, tags:dict, n_step:int, validation:bool, validation_num:int) -> Generator[tuple[list,np.array], None, None]:
        """ Method that yields batches of input-output pairs when asked. It stores all input-output pairs in memory for the whole dataset.

        Args:
            captions (dict): Dictionary with keys to be the patients' ids and values the captions.
            image_tuples (dict): Dictionary with keys to be the patients' ids and values the image features.
            tags(dict): Dictionary with keys to be the patients' ids and values the tags' features.
            n_step (int): Allows us to tune how many images worth of input-output pairs to generate for each batch.
            validation (bool): If to generate a development data set for validate the model accuracy.
            validation_num (int): Allows us to tune how many images worth of input-output pairs to generate for development set.

        Yields:
            Generator[tuple[list,np.array], None, None]: A list with input images in first snd second index and current text sequence in the third index, 
            and the output y in np.array format.
        """
        while True:
            # fetch patients ids
            patients = list(captions.keys())
            # if for development fetch a small batch from training.
            if validation:
                assert validation_num > 0
                patients = patients[-validation_num:]
            elif not validation:
                if validation_num > 0:
                    patients = patients[:-validation_num]

            for i in range(0, len(patients), n_step):
                Ximages1, Ximages2, Xtags, XSeq, y = list(), list(), list(), list(), list()
                for j in range(i, min(len(patients), i+n_step)):
                    patient_id = patients[j]
                    # retrieve text input
                    caption = captions[patient_id]

                    # generate input-output pairs (many images in each batch)
                    img1 = image_tuples[patient_id][0][0]
                    img2 = image_tuples[patient_id][1][0]
                    tag = tags[patient_id][0]

                    # case multi-modal
                    if self.multi_modal:
                        # generate the all input image-text sequences
                        in_img1, in_img2, in_tag, in_seq, out_word = self.__create_iuxray_sequences(caption, img1, img2, tag)
                        for k in range(len(in_img1)):
                            Ximages1.append(in_img1[k])
                            Ximages2.append(in_img2[k])
                            Xtags.append(in_tag[k])
                            XSeq.append(in_seq[k])
                            y.append(out_word[k])
                    else:
                        # generate the all input image-text sequences
                        in_img1, in_img2, in_seq, out_word = self.__create_iuxray_sequences(caption, img1, img2, tag)
                        for k in range(len(in_img1)):
                            Ximages1.append(in_img1[k])
                            Ximages2.append(in_img2[k])
                            XSeq.append(in_seq[k])
                            y.append(out_word[k])
                    # yield this batch of samples to the model
                    #print (array(Ximages1).shape)
                if self.multi_modal:
                    # store to memory and yield when called.
                    yield [np.array(Ximages1), np.array(Ximages2), np.array(Xtags), np.array(XSeq)], np.array(y)
                else:
                    # store to memory and yield when called.
                    yield [np.array(Ximages1), np.array(Ximages2), np.array(XSeq)], np.array(y)

    def train_imageclef_model(self, train_data:list, input_shape:tuple, optimizer:tensorflow.keras.optimizers, model_name:str, n_epochs:int, batch_size:int) -> Model:
        """ Method that start the training stage of the model (ImageCLEF). At the end, stores the trained model in directory.

        Args:
            train_data (list): The training dataset
            input_shape (tuple): The input shape for the image encoder (e.g., (1920,), (1024,), etc.). Inputs extraced from the last average pooling layer of the image encoder.
            optimizer (tensorflow.keras.optimizers): The keras oprimizer to be used during training (e.g., Adam, AdamW, etc.)
            model_name (str): The name for model to be saved
            n_epochs (int): The number of epochs to be trained
            batch_size (int): The batch size to be used during training

        Returns:
            Model: The trained model (Keras)
        """
        # display the training procedure
        verbose = 1
        # get a small batch from trainig n set for development set to eval our model.
        val_len = int(0.05 * len(train_data[1]))
        train_len = len(train_data[1])
        train_steps = int(train_len / batch_size)
        val_steps = int(val_len / batch_size)
        # build ImageCLEF architecture
        model = self.__make_imageclef_model(input_shape, optimizer)
        print('val_steps:', val_steps)
        print('steps_per_epoc:', val_steps)

        # generate training data sequences to feed our network
        train_gen = self.__imageclef_data_generator(
            train_data[1],
            train_data[0],
            batch_size,
            validation=False,
            validation_num=val_len,
        )
        # generate development data sequences to eval our network during training
        val_gen = self.__imageclef_data_generator(
            train_data[1],
            train_data[0],
            batch_size,
            validation=True,
            validation_num=val_len,
        )

        # init the early stop for training
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

        # fit model and begin training.
        model.fit_generator(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            epochs=n_epochs,
            verbose=verbose,
            callbacks=cs,
        )

        # save model to directory
        model.save(model_name + ".h5")
        return model

    def train_iuxray_model(self, train_data:list, input_shape:tuple, optimizer:tensorflow.keras.optimizers, model_name:str, n_epochs:int, batch_size:int) -> Model:
        """ Method that start the training stage of the model (IU X-Ray). At the end, stores the trained model in directory.

        Args:
            train_data (list): The training dataset
            input_shape (tuple): The input shape for the image encoder (e.g., (1920,), (1024,), etc.). Inputs extraced from the last average pooling layer of the image encoder.
            optimizer (tensorflow.keras.optimizers): The keras oprimizer to be used during training (e.g., Adam, AdamW, etc.)
            model_name (str): The name for model to be saved
            n_epochs (int): The number of epochs to be trained
            batch_size (int): The batch size to be used during training

        Returns:
            Model: The trained model (Keras)
        """
        # display the training procedure
        verbose = 1
        # get a small batch from trainig n set for development set to eval our model.
        val_len = int(0.05 * len(train_data[1]))
        train_len = len(train_data[1])
        train_steps = int(train_len / batch_size)
        val_steps = int(val_len / batch_size)
        print('val_steps:', val_steps)
        print('steps_per_epoc:', train_steps)
        # build IU X-Ray architecture
        model = self.__make_iuxray_model(input_shape, optimizer)
         # generate training data sequences to feed our network
        train_gen = self.__iuxray_data_generator(
            train_data[1],
            train_data[0],
            train_data[2],
            batch_size,
            validation=False,
            validation_num=val_len,
        )
        # generate development data sequences to eval our network during training
        val_gen = self.__iuxray_data_generator(
            train_data[1],
            train_data[0],
            train_data[2],
            batch_size,
            validation=True,
            validation_num=val_len,
        )
        # init the early stop for training
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
        # fit model and begin training.
        model.fit(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            epochs=n_epochs,
            verbose=verbose,
            callbacks=cs,
        )
        # save model to directory
        model.save(model_name + ".h5")
        return model

    def evaluate_ensemble_model(self, models:list, test_captions:dict, test_images:dict, test_tags:dict, dataset:str='iu_xray', ensemble_method:str='AP')->tuple[dict, dict]:
        """ Method to evaluate our ensemble model for both validation and test sets.

        Args:
            models (list): The trained (Keras) models (with differente image encoders (i.e. DenseNet-121, DenseNet-201, etc.))
            test_captions (dict): Dictionary with keys to be the test set patients' ids and values the captions.
            test_images (dict): Dictionary with keys to be the test set patients' ids and values the image features.
            test_tags (dict): Dictionary with keys to be the test set patients' ids and values the tags' features.
            dataset (str, optional): The dataset in which we want to evaluate our ensemble model. Defaults to 'iu_xray'.
            ensemble_method (str, optional): The ensemble greedy search choice [AP, MVP, MP]. Defaults to 'AP'.

        Returns:
            tuple[dict, dict]: Image, gold_truth pairs and Image, predicted caption pairs
        """
        # init gold truths and predicted dicts to store results
        gold, predicted = {}, {}

        # use our GreedySearch implemented class
        gs = GreedySearch(start_token=self.start_token, end_token=self.end_token, max_length=self.max_length,
                          tokenizer=self.tokenizer, idx_to_word=self.idx2word, word_to_idx=self.word2idx)
        # for each test patient
        for key in tqdm(test_images[0]):
            test_photos = [test_images[i][key]for i in range(len(test_images))]
            current_tags = test_tags[key]
            # case 'AP'
            if ensemble_method == 'AP':
                caption = gs.greedy_search_ensembles_AP(models, test_photos, current_tags, dataset=dataset, multi_modal=self.multi_modal)
            # case 'MVP'
            elif ensemble_method == 'MVP':
                caption = gs.greedy_search_ensembles_MVP(models, test_photos, current_tags, dataset=dataset, multi_modal=self.multi_modal)
            # case 'MP'
            elif ensemble_method == 'MP':
                caption = gs.greedy_search_ensembles_MP( models, test_photos, current_tags, dataset=dataset, multi_modal=self.multi_modal)

            # remove the start, end, and sequence token in order to make the caption look-alike to the gold one
            # and store the result
            predicted[key] = self.remove_basic_tokens(caption)
            gold[key] = test_captions[key]

        return gold, predicted

    def evaluate_model(self, model:Model, test_captions:dict, test_images:dict, test_tags:dict, eval_dataset:str ='iu_xray', evaluator_choice:str='beam_10'):
        """ Method to evaluate our trained model for both validation and test sets.

        Args:
            models (Model): The trained (Keras) models (with differente image encoders (i.e. DenseNet-121, DenseNet-201, etc.))
            test_captions (dict): Dictionary with keys to be the test set patients' ids and values the captions.
            test_images (dict): Dictionary with keys to be the test set patients' ids and values the image features.
            test_tags (dict): Dictionary with keys to be the test set patients' ids and values the tags' features.
            dataset (str, optional): The dataset in which we want to evaluate our ensemble model. Defaults to 'iu_xray'.
            evaluator_choice (str, optional): The sampling method to be used for generating the captions [Greedy or Beam Search]. Defaults to 'beam_10'.

        Returns:
            tuple[dict, dict]: Image, gold_truth pairs and Image, predicted caption pairs
        """
        # init gold truths and predicted dicts to store results
        gold, predicted = {}, {}
        # case Beam Search
        if evaluator_choice.split('_')[0] == 'beam':
             # use our BeamSearch implemented class
            bs = BeamSearch(start_token=self.start_token, end_token=self.end_token, max_length=self.max_length,
                            tokenizer=self.tokenizer, idx_to_word=self.idx2word, 
                            word_to_idx=self.word2idx, beam_index=int(evaluator_choice.split('_')[1]))     # if beam take the number for beam size next tp '_' char
            # for each test patient
            for key in tqdm(test_images, desc='Evaluating model..'):
                # case iu-xray
                if eval_dataset == 'iu_xray':
                    caption = bs.beam_search_predict(
                        model, test_images[key], test_tags[key], dataset=eval_dataset, multi_modal=self.multi_modal)
                else:
                    # case ImageCLEF
                    caption = bs.beam_search_predict(
                        model, test_images[key], None, dataset=eval_dataset, multi_modal=self.multi_modal)
                # remove the start, end, and sequence token in order to make the caption look-alike to the gold one
                # and store the result
                predicted[key] = self.remove_basic_tokens(caption)
                gold[key] = test_captions[key]
        else:
            # use our GreedySearch implemented class
            gs = GreedySearch(start_token=self.start_token, end_token=self.end_token, max_length=self.max_length,
                              tokenizer=self.tokenizer, idx_to_word=self.idx2word, word_to_idx=self.word2idx)
            # for each test patient
            for key in tqdm(test_images, desc='Evaluating model..'):
                # case iu-xray
                if eval_dataset == 'iu_xray':
                    caption = gs.greedy_search_predict(
                        model, test_images[key], tag=test_tags[key], dataset=eval_dataset, multi_modal=self.multi_modal)
                else:
                    # case ImageCLEF
                    caption = gs.greedy_search_predict(
                        model, test_images[key], tag=None, dataset=eval_dataset, multi_modal=self.multi_modal)

                # remove the start, end, and sequence token in order to make the caption look-alike to the gold one
                # and store the result
                predicted[key] = self.remove_basic_tokens(caption)
                gold[key] = test_captions[key]

        return gold, predicted

    def remove_basic_tokens(self, caption:str)->str:
        """ Removes the start, end, sequence separator, <pad> and <unk> from the captions in order to confront with the gold ones.

        Args:
            caption (str): The caption to apply the pre-process

        Returns:
            str: The preprocessed caption
        """
        caption = caption.replace(self.start_token, " ").replace(self.end_token, " ").replace("<pad>", " ").replace("<unk>", " ").replace(self.seq_sep, '. ')
        return caption

class Encoder:
    def __init__(self, dropout_rate: float = 0.2, dense_activation="relu"):
        """ Image encoder to produced image feature representations from the image embeddings extraced from a pre-trained on ImageNet encoders (see modules/image_encoder.py)

        Args:
            dropout_rate (float, optional): The dropout percentage to be used for our features extraction. Defaults to 0.2.
            dense_activation (str, optional): The activation function of the dense layer. Defaults to "relu".
        """
        self.activation_function = dense_activation
        self.fe = Dense(256, activation=self.activation_function)
        self.avg_pool = GlobalMaxPooling2D()
        self.dropout = Dropout(dropout_rate)

    def encode_(self, img:np.array)->np.array:
        """ Produces image feature representations from image embeddings extraced from a pre-trained on ImageNet encoders (see modules/image_encoder.py)

        Args:
            img (np.array): The image embeddings extraced from a pre-trained on ImageNet encoders (see modules/image_encoder.py)
                            shape = (1,N) with N to be the size of the last average pooling layer of the employed image encoder.

        Returns:
            np.array: The image features for input image.
        """
        # img = self.avg_pool(img)        ## for CotNet and BEiT only
        img = self.dropout(img)
        img = self.fe(img)
        return img      # shape = (1,256) with 256 to be the size of the Dense 


class TagEncoder:
    def __init__(self, pretrained: bool, embedding_dim: int, vocab_size: int, max_tags: int, max_sequence: int, 
        weights: list, max_length: int, dropout_rate: float, dense_activation="relu"):
        """ Tag encoder to produced tag feature representations for each patient.

        Args:
            pretrained (bool): If to use the pre-trained weights from FastText model
            embedding_dim (int): The dimension size for embedding layer
            vocab_size (int): The size of tags' vocabulary
            max_tags (int): The maximum number of unique tags
            max_sequence (int): The maximum number of tags' occurence in all patients.
            weights (list): The pre-trained weights from FastText model
            max_length (int): The max length of caption
            dropout_rate (float): The dropout percentage to be used for our features extraction.
            dense_activation (str, optional): The activation function of the dense layer. Defaults to "relu".
        """
        
        self.max_length = max_length
        self.max_sequence = max_sequence
        self.activation_function = dense_activation
        # if to use FastText weights
        if pretrained:
            self.embedding = Embedding(
                input_dim=max_tags + 2,
                output_dim=embedding_dim,
                weights=weights,
                mask_zero=True,
                trainable=False,
            )
        # else build a trainable one
        else:
            self.embedding = Embedding(
                vocab_size, embedding_dim, mask_zero=True)
        self.ling_model = GRU(256, return_sequences=False)
        self.dropout = Dropout(dropout_rate)
        self.fe = Dense(256, activation=self.activation_function)
        self.rep_vec = RepeatVector(self.max_length)

    def tag_encode_(self, tag:np.array)->np.array:
        """ Produces tag feature representations for each patient, using the linguistic model as well as the embedding layer.

        Args:
            tag (np.array): The tokenized tags. Shape = (1, max_sequence)

        Returns:
            np.array: The tag feature representations extracted from the encoder.
        """
        tag = self.embedding(tag)       # shape: (1, max_sequence) --> (1, max_sequence, embedding_dim)
        # tag = self.dropout(tag)
        tag = self.ling_model(tag)      # shape: (1, max_sequence, embedding_dim) --> (1, 256)
        tag = self.fe(tag)              # shape: (1, 256) --> (1, 256)
        tag = self.rep_vec(tag)         # shape: (1, 256) --> (1, max_length, 256)
        return tag


class EmbeddingLayer:
    def __init__(self, embedding_dim:int, vocab_size:int, linguistic_model:str="gru"):
        """Text encoder to produce text feature representations for each patient, at each time step, given the previous generated text sequence.

        Args:
            embedding_dim (int): The dimension size for embedding layer
            vocab_size (int): The size of captions' vocabulary
            linguistic_model (str, optional): The linguistic RNN model to be used. Defaults to "gru".
        """
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

    def get_embedding(self, x:np.array)->np.array:
        """ Produces text sequence word embeddings for caption, using the linguistic model as well as the embedding layer.

        Args:
            x (np.array): Current tokenized text sequence. Shape = (1, max_length)

        Returns:
            np.array: The text sequence word embeddings
        """
        x = self.embedding(x)           # shape: (1, max_length) --> (1, max_length, embedding_dim)
        x = self.ling_model1(x)         # shape: (1, max_length, embedding_dim) --> (1, max_length, 256)
        x = self.ling_model2(x)         # shape: (1, max_length, 256) --> (1, max_length, 256)
        x = self.time_dis(x)            # shape: (1, max_length, 256) --> (1, max_length, 256)
        return x


class Decoder:
    def __init__(self, max_length:int, vocab_size:int, dense_activation:str="relu", linguistic_model:str ='gru'):
        """ Decoder modules that is fed with the concatenation of image features (+ tag features if multi-modal) and current text sequence feature representations
        Its last hidden state is followed by a feed-forward neural network (FFNN). 
        The latter yields a probability distribution over the vocabulary and choose the next word token. 
        Unlike the original Show and Tell model that uses the image only as the initial state of the Decoder, we fed it to the Decoder at each time step.

        Args:
            max_length (int): The maximum length of captions to be utilised
            vocab_size (int): The size of captions' vocabulary
            dense_activation (str, optional): The activation function of the dense layer. Defaults to "relu".
            linguistic_model (str, optional): The linguistic RNN model to be used. Defaults to "gru".
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.activation_function = dense_activation
        if linguistic_model == 'gru':
            self.decoder = GRU(256)
        elif linguistic_model == 'bigru':
            self.decoder = Bidirectional(GRU(256))
        else:
            self.decoder = LSTM(256)
        self.fc = Dense(256, activation=self.activation_function)
        self.out = Dense(self.vocab_size, activation="softmax",
                         name="output_layer")

    def decode_(self, x:np.array)->np.array:
        """ Yields a probability distribution over the vocabulary and choose the next word token. 

        Args:
            x (np.array): The concatenation of image features (+ tag features if multi-modal) and current text sequence feature representations/
                            Shape = (1, max_length, 512) , 512 to be the concatenation of two 256

        Returns:
            np.array: The probability distribution over the vocabulary using 'softmax'.
        """
        x = self.decoder(x)     # shape: (1, max_length, 512) --> (1, 256)
        x = self.fc(x)          # shape: (1, 256) --> (1, 256)
        x = self.out(x)         # shape: (1, 256) --> (1, vocab_size)
        return x
