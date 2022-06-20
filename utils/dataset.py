from operator import imod
from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
import os
import json
import numpy as np
import pandas as pd
import random
import nltk
import pickle

from torch import threshold
from text_handler import TextHandler
from tqdm import tqdm

from utils.vocabulary import Vocabulary

nltk.download("punkt", quiet=True)


def get_df(dataframe, ids):
    temp_df = dataframe[dataframe["ID"].isin(ids)]
    return temp_df


def dictify_r2gen(dataframe, split):
    dataset_folder = "/home/cave-of-time/panthro/dataset/dataset/images/"

    split_list = list()
    ids = dataframe.ID.to_list()
    report = dataframe.caption.to_list()

    for i in tqdm(range(len(ids))):
        image_path = dataset_folder + ids[i] + ".jpg"

        batch_dict = {
            "id": ids[i],
            "report": report[i],
            "image_path": [image_path],
            "split": split,
        }

        split_list.append(batch_dict)

    return split_list


class Dataset:
    def __init__(
        self,
        image_vectors: dict,
        captions_df: pd.DataFrame,
        clear_long_captions: bool = True,
    ):
        self.image_vectors = image_vectors
        self.caption_dataset = captions_df
        self.clear_long_captions = clear_long_captions
        self.text_handler = TextHandler()
        (
            self.train,
            self.train_captions_prepro,
            self.dev,
            self.dev_captions,
            self.test,
            self.test_captions,
        ) = self.build_dataset()

        self.vocab, self.tokenizer, self.word2idx, self.idx2word = self.build_vocab(
            training_captions=self.train_captions_prepro
        )

    def delete_long_captions(self, data: dict, threshold=80):
        filtered_data = {}

        for image_id, caption in data.items():
            tokens = word_tokenize(caption)
            if len(tokens) <= threshold:
                filtered_data[image_id] = caption

        return filtered_data

    def build_splits(self):

        image_ids = self.caption_dataset.ID.to_list()
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

    def __get_image_vectors(self, keys) -> dict:

        return {
            k: v
            for k, v in tqdm(
                self.image_dataset.items(), desc="Fetching image embeddings from the"
            )
            if k in keys
        }

    def __get_captions(self, _ids: list) -> list():
        return self.caption_dataset[self.caption_dataset["ID"].isin(_ids)].to_list()

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

    def build_cv_dataset(self, train_ids, dev_ids):

        train_df = self.caption_dataset[self.caption_dataset["ID"].isin(train_ids)]
        if self.clear_long_captions:
            print(
                "Deleting some outliers. Train dataset before cleaning:", len(train_df)
            )
            train_dict = dict(zip(train_df.ID.to_list(), train_df.caption.to_list()))
            train_dict_filtered = self.delete_long_captions(train_dict)
            print(f"We deleted {len(train_df)-len(train_dict_filtered)} outliers!")

        # train =
        dev = {
            k: v
            for k, v in tqdm(
                self.image_dataset.items(), desc="Fetching image embeddings for dev"
            )
            if k in dev_ids
        }

        train_data = self.caption_dataset[
            self.caption_dataset["ID"].isin(list(train.keys()))
        ]

        dev_dataset = self.caption_dataset[self.caption_dataset["ID"].isin(dev_ids)]

        training_captions = [
            self.text_handler.separate_sequences(caption)
            for caption in train_data.caption.to_list()
        ]

        dev_captions = dev_dataset.caption.to_list()

        return train, training_captions, dev, dev_captions

    def build_dataset(self):
        train_ids, dev_ids, test_ids = self.build_splits()

        train_df = self.caption_dataset[self.caption_dataset["ID"].isin(train_ids)]
        if self.clear_long_captions:
            print(
                "Deleting some outliers. Train dataset before cleaning:", len(train_df)
            )
            train_dict = dict(zip(train_df.ID.to_list(), train_df.caption.to_list()))
            train_dict_filtered = self.delete_long_captions(train_dict)
            print(f"We deleted {len(train_df)-len(train_dict_filtered)} outliers!")

        train = self.__get_image_vectors(train_dict_filtered.keys())
        dev = self.__get_image_vectors(dev_ids)
        test = self.__get_image_vectors(test_ids)

        train_captions = self.__get_captions(list(train.keys()))
        dev_captions = self.__get_captions(dev_ids)
        test_captions = self.__get_captions(test_ids)

        train_captions_prepro = self.text_handler.preprocess_all(train_captions)

        return (
            train,
            train_captions_prepro,
            dev,
            dev_captions,
            test,
            test_captions,
        )

    def build_vocab(self, training_captions: list, threshold: int = 3):
        vocab = Vocabulary(texts=training_captions, threshold=threshold)
        tokenizer, word2idx, idx2word = vocab.build_vocab()
        return vocab, tokenizer, word2idx, idx2word


if __name__ == "__main__":
    dataset_path = (
        "/home/cave-of-time/panthro/dataset/ImageCLEF2022/Imageclef2022_dataset_all.csv"
    )
    dataset = Dataset(captions_path=dataset_path)
    TRAIN_IDS, DEV_IDS = dataset.build_pseudo_cv_splits()
    cv = 5
    for i in range(cv):
        train, training_captions, dev, dev_captions = dataset.build_cv_dataset(
            TRAIN_IDS[i], DEV_IDS[i]
        )

        break
