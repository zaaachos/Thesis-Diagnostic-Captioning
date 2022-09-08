from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
import os
import json
import numpy as np
import pandas as pd
import random
import nltk
import pickle

from utils.text_handler import TextHandler
from tqdm import tqdm

from utils.vocabulary import Vocabulary
from utils.jsonfy import *
from modules.image_encoder import load_encoded_vecs, save_encoded_vecs


nltk.download("punkt", quiet=True)

class Dataset:
    def __init__(
        self,
        image_vectors: dict,
        captions_data: dict,
        clear_long_captions: bool = True,
    ):
        self.image_vectors = image_vectors
        self.captions_data = captions_data
        self.clear_long_captions = clear_long_captions
        self.text_handler = TextHandler()

    def delete_long_captions(self, data: dict, threshold=80):
        filtered_data = {}

        for image_id, caption in data.items():
            tokens = word_tokenize(caption)
            if len(tokens) <= threshold:
                filtered_data[image_id] = caption

        return filtered_data

    def build_splits(self):
    
        image_ids = list( self.captions_data.keys() )
        np.random.shuffle(image_ids)

        test_split_threshold = int(0.15 * len(image_ids))
        train, test = (
            image_ids[:-test_split_threshold],
            image_ids[-test_split_threshold:],
        )

        dev_split_threshold = int(0.1 * len(train))
        train, dev = (
            train[:-dev_split_threshold],
            train[-dev_split_threshold:],
        )

        return train, dev, test

    def __get_image_vectors(self, keys):

        return {
            k: v
            for k, v in tqdm(
                self.image_vectors.items(), desc="Fetching image embeddings.."
            )
            if k in keys
        }

    def __get_captions(self, _ids: list):
        return { key:value for key, value in self.captions_data.items() if key in _ids}
 
    def build_pseudo_cv_splits(self):
        image_ids = self.caption_dataset.ID.to_list()
        np.random.shuffle(image_ids)

        kf = KFold(n_splits=15)
        TRAIN_IDS, TEST_IDS = list(), list()
        for train_index, test_index in kf.split(image_ids):
            train_ids = [image_ids[index] for index in train_index]
            test_ids = [image_ids[index] for index in test_index]
            TRAIN_IDS.append(train_ids)
            TEST_IDS.append(test_ids)

        return TRAIN_IDS, TEST_IDS

    def build_vocab(self, training_captions: list, threshold: int = 3):
        vocab = Vocabulary(texts=training_captions, threshold=threshold)
        tokenizer, word2idx, idx2word = vocab.build_vocab()
        return vocab, tokenizer, word2idx, idx2word
    
    
    
class IuXrayDataset(Dataset):
    def __init__(
            self,
            image_vectors: dict,
            captions_data: dict,
            tags_data: dict):
        self.image_vectors, self.captions_data, self.text_handler = super().__init__(image_vectors=image_vectors, captions_data=captions_data, clear_long_captions=False)
        self.tags_data = tags_data
        self.train_dataset, self.dev_dataset, self.test_dataset = self.build_dataset()

        print(f"Train: patients={len(self.train_dataset[0])}, captions={len(self.train_dataset[1])}, tags={len(self.train_dataset[2])}")
        print(f"Dev: patients={len(self.dev_dataset[0])}, captions={len(self.dev_dataset[1])}, tags={len(self.dev_dataset[2])}")
        print(f"Test: patients={len(self.test_dataset[0])}, captions={len(self.test_dataset[1])}, tags={len(self.test_dataset[2])}")

        self.vocab, self.tokenizer, self.word2idx, self.idx2word = super().build_vocab(
            training_captions=list(self.train_dataset[1].values())
        )
        
    def __get_tags(self, _ids: list):
        return { key:value for key, value in self.tags_data.items() if key in _ids}
        
    def build_dataset(self):
        train_ids, dev_ids, test_ids = super().build_splits()

        train_images = super().__get_image_vectors(train_ids)
        dev_images = super().__get_image_vectors(dev_ids)
        test_images = super().__get_image_vectors(test_ids)

        train_captions = super().__get_captions(train_ids)
        dev_captions = super().__get_captions(dev_ids)
        test_captions = super().__get_captions(test_ids)

        train_captions_prepro = self.text_handler.preprocess_all(
            list(train_captions.values()))
            
        train_captions_prepro = dict( zip( train_ids, train_captions_prepro ) )
            
        train_tags = self.__get_tags(train_ids)
        dev_tags = self.__get_tags(dev_ids)
        test_tags = self.__get_tags(test_ids)
                
        train_dataset = [train_images, train_captions_prepro, train_tags]
        dev_dataset = [dev_images, dev_captions, dev_tags]
        test_dataset = [test_images, test_captions, test_tags]


        return train_dataset, dev_dataset, test_dataset
    


class ImageCLEFDataset(Dataset):
    def __init__(
            self,
            image_vectors: dict,
            captions_data: dict):
        self.image_vectors, self.captions_data, self.text_handler = super().__init__(image_vectors=image_vectors, captions_data=captions_data, clear_long_captions=False)
        self.train_dataset, self.dev_dataset, self.test_dataset = self.build_dataset()

        print(f"Train: patients={len(self.train_dataset[0])}, captions={len(self.train_dataset[1])}, tags={len(self.train_dataset[2])}")
        print(f"Dev: patients={len(self.dev_dataset[0])}, captions={len(self.dev_dataset[1])}, tags={len(self.dev_dataset[2])}")
        print(f"Test: patients={len(self.test_dataset[0])}, captions={len(self.test_dataset[1])}, tags={len(self.test_dataset[2])}")

        self.vocab, self.tokenizer, self.word2idx, self.idx2word = super().build_vocab(
            training_captions=list(self.train_dataset[1].values())
        )
        
    def build_dataset(self):
        train_ids, dev_ids, test_ids = super().build_splits()

        train_images = super().__get_image_vectors(train_ids)
        dev_images = super().__get_image_vectors(dev_ids)
        test_images = super().__get_image_vectors(test_ids)

        train_captions = super().__get_captions(train_ids)
        dev_captions = super().__get_captions(dev_ids)
        test_captions = super().__get_captions(test_ids)

        train_captions_prepro = self.text_handler.preprocess_all(
            list(train_captions.values()))
            
        train_captions_prepro = dict( zip( train_ids, train_captions_prepro ) )
                
        train_dataset = [train_images, train_captions_prepro]
        dev_dataset = [dev_images, dev_captions]
        test_dataset = [test_images, test_captions]


        return train_dataset, dev_dataset, test_dataset


