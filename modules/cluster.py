import sys
sys.path.append('..')

import json
from tqdm import tqdm
from sklearn.metrics import silhouette_samples, silhouette_score
from tensorflow.keras.preprocessing.image import load_img
from random import randint
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models import *
from modules.image_encoder import load_encoded_vecs, save_encoded_vecs
from utils.dataset import ImageCLEFDataset


class Cluster:

    def __init__(self, K:int, clef_dataset:ImageCLEFDataset):
        self.K = K
        self.dataset = clef_dataset

    def do_PCA(self, features):

        feat = np.array(list(features.values()))
        feat = feat.reshape(-1, feat.shape[2])

        pca = PCA(n_components=100, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)
        return x

    def do_Kmeans(self, x):
        kmeans = KMeans(n_clusters=self.K, random_state=22)
        kmeans.fit(x)
        return kmeans

    def load_features(self):
        return self.dataset.get_splits_sets()

    def clustering(self):
        train_features, valid_features, test_features = self.load_features()
        train_ids, val_ids, test_ids = list(train_features[0].keys()), list(valid_features[0].keys()), list(test_features[0].keys())

        all_features = dict(train_features, **valid_features)
        all_features = dict(all_features, **test_features)

        pca = self.do_PCA(all_features)
        kmeans = self.do_Kmeans(pca)
        
        train_index_limit, val_index_limit = len(train_features), len(train_features)+len(valid_features)

        train_k_means_labels = kmeans.labels_[:train_index_limit]
        valid_k_means_labels = kmeans.labels_[train_index_limit:val_index_limit]
        test_k_means_labels = kmeans.labels_[val_index_limit:]
        
        
        print('# train kmeans:',  len(train_k_means_labels))
        print('# dev kmeans:',  len(valid_k_means_labels))
        print('# test kmeans:',  len(test_k_means_labels))
        


        groups_train = {}
        for file, cluster in tqdm(zip(train_ids, train_k_means_labels)):
            if cluster not in groups_train.keys():
                groups_train[cluster] = []
                groups_train[cluster].append(file)
            else:
                groups_train[cluster].append(file)

        groups_valid = {}
        for file, cluster in tqdm(zip(val_ids, valid_k_means_labels)):
            if cluster not in groups_valid.keys():
                groups_valid[cluster] = []
                groups_valid[cluster].append(file)
            else:
                groups_valid[cluster].append(file)
        
        groups_test = {}
        for file, cluster in tqdm(zip(test_ids, test_k_means_labels)):
            if cluster not in groups_test.keys():
                groups_test[cluster] = []
                groups_test[cluster].append(file)
            else:
                groups_test[cluster].append(file)

        print('# train kmeans:',  len(groups_train))
        print('# dev kmeans:',  len(groups_valid))
        print('# test kmeans:',  len(groups_test))
        return groups_train, groups_valid, groups_test



