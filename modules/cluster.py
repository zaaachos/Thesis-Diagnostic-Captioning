# import sys
import sys
sys.path.append('..')

# progress bar import
from tqdm import tqdm

# numpy, sklearn imports
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from tensorflow.keras.preprocessing.image import load_img

# utils imports
from models import *
from utils.dataset import ImageCLEFDataset


class Cluster:

    def __init__(self, K:int, clef_dataset:ImageCLEFDataset):
        """ Class to perform K-Means clustering in ImageCLEF dataset. We used this system for the contest

        Args:
            K (int): The K clusters we want
            clef_dataset (ImageCLEFDataset): The dataset we employed. Only CLEF is acceptable.
        """
        self.K = K
        self.dataset = clef_dataset

    def do_PCA(self, features:dict) -> np.array:
        """ Perforrms Principal Component Analysis (PCA), to reduce the huge size of the arrays

        Args:
            features (dict): The image_ids, image_vectors pairs.

        Returns:
            np.array: The image_ids, image_vectors pairs, with reduced size.
        """

        feat = np.array(list(features.values()))
        feat = feat.reshape(-1, feat.shape[2])

        pca = PCA(n_components=100, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)
        return x

    def do_Kmeans(self, x:np.array) -> KMeans:
        """ Fit the K-Means

        Args:
            x (np.array): The image vectors

        Returns:
            KMeans: The fitted K-Means object
        """
        kmeans = KMeans(n_clusters=self.K, random_state=22)
        kmeans.fit(x)
        return kmeans

    def load_features(self) ->tuple[list[dict], list[dict], list[dict]]:
        """ Loads train, validation, test sets 

        Returns:
            tuple[list[dict], list[dict], list[dict]]: The train, validation, test sets in dictionary format
        """
        return self.dataset.get_splits_sets()

    def clustering(self) -> tuple[dict, dict, dict]:
        """ Performs the k-Means clustering using the fitted K-Means object.

        Returns:
            tuple[dict, dict, dict]: The clustered train, val, test image_ids, image_vectors pairs.
        """
        # load splits
        train_features, valid_features, test_features = self.load_features()
        # get the ids for each split
        train_ids, val_ids, test_ids = list(train_features[0].keys()), list(valid_features[0].keys()), list(test_features[0].keys())

        # concate all features to perform a more efficient K-Means
        all_features = dict(train_features, **valid_features)
        all_features = dict(all_features, **test_features)

        # reduce size for fast training
        pca = self.do_PCA(all_features)
        # perform clustering
        kmeans = self.do_Kmeans(pca)
        
        train_index_limit, val_index_limit = len(train_features), len(train_features)+len(valid_features)
        # get the clustering labels for each set
        train_k_means_labels = kmeans.labels_[:train_index_limit]
        valid_k_means_labels = kmeans.labels_[train_index_limit:val_index_limit]
        test_k_means_labels = kmeans.labels_[val_index_limit:]
        
        
        print('# train kmeans:',  len(train_k_means_labels))
        print('# dev kmeans:',  len(valid_k_means_labels))
        print('# test kmeans:',  len(test_k_means_labels))
        

        # store the clustered train, validation, test set images
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



