# sklearn and nltk imports
from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
nltk.download("punkt", quiet=True)

# tensorflow imports
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

# progress bar
from tqdm import tqdm

# utils imports
from utils.text_handler import TextHandler
from utils.vocabulary import Vocabulary
from utils.jsonfy import *



class Dataset:
    def __init__(self, image_vectors:dict, captions_data:dict, clear_long_captions:bool = True):
        """ Base class to create the employed dataset for my research, i.e. ImageCLEF and IU X-Ray 

        Args:
            image_vectors (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
            captions_data (dict): Dictionary with keys to be the ImageIDs and values the captions.
            clear_long_captions (bool, optional): If we want to drop the outlier long captions. Defaults to True.
        """
        self.image_vectors = image_vectors
        self.captions_data = captions_data
        self.clear_long_captions = clear_long_captions
        # init a text handler object to pre-process training captions
        self.text_handler = TextHandler()

    def delete_long_captions(self, data:dict, threshold:int=80) -> dict:
        """ Function that removes the long captions only from the training set. This method was utilised during ImageCLEF campaign.

        Args:
            data (dict): Dictionary with keys to be the ImageIDs and values the captions.
            threshold (int, optional): The maximum length limit. Defaults to 80.

        Returns:
            dict: Dictionary with keys to be the ImageIDs and values the captions, without the instances whose captions are long.
        """
        filtered_data = {}

        for image_id, caption in data.items():
            tokens = word_tokenize(caption)
            if len(tokens) <= threshold:
                filtered_data[image_id] = caption

        return filtered_data

    def build_splits(self) ->tuple[list, list, list]:
        """ This function makes the split sets for trainig, validation and test.
        In particulare, we followed the next splits:
        train: 80% 
        valid: 5%
        test: 15%

        Returns:
            tuple[list, list, list]: Training, validation, test set ids.
        """
    
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

    def get_image_vectors(self, keys:list) -> dict:
        """ Fetches from the whole dataset the image embeddings according to the utilised set.

        Args:
            keys (list): Split set ids

        Returns:
            dict: Dictionary with keys to be the ImageIDs and values the image embeddings, for each split set.
        """

        return { k: v for k, v in tqdm(self.image_vectors.items(), desc="Fetching image embeddings..") if k in keys }

    def get_captions(self, _ids:list) -> dict:
        return { key:value for key, value in self.captions_data.items() if key in _ids}
 
    def build_pseudo_cv_splits(self) -> tuple[list, list]:
        """ This function makes cross-validaion splis using K-Fold cross validation. It was used only for ImageCLEF campaign.
        More details are described in my Thesis.

        Returns:
            tuple[list, list]: Training and test fold sets.
        """
        image_ids = list( self.captions_data.keys() )
        np.random.shuffle(image_ids)

        # apply 15-Fold CV
        kf = KFold(n_splits=15)
        train_fold_ids, test_fold_ids = list(), list()
        for train_index, test_index in kf.split(image_ids):
            train_ids = [image_ids[index] for index in train_index]
            test_ids = [image_ids[index] for index in test_index]
            train_fold_ids.append(train_ids)
            test_fold_ids.append(test_ids)

        return train_fold_ids, test_fold_ids

    def build_vocab(self, training_captions:list, threshold:int = 3) -> tuple[Vocabulary, Tokenizer, dict, dict]:
        """ This method creates the employed vocabulary given the training captions

        Args:
            training_captions (list): All training captions
            threshold (int, optional): The cut-off frequence for Vocabulary. Defaults to 3.

        Returns:
            tuple[Vocabulary, Tokenizer, dict, dict]: The Vocabulary object, the fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary. 
            The latters are mappers for words and index respectively
        """
        vocab = Vocabulary(texts=training_captions, threshold=threshold)
        tokenizer, word2idx, idx2word = vocab.build_vocab()
        return vocab, tokenizer, word2idx, idx2word
    
    
    
class IuXrayDataset(Dataset):
    def __init__(self, image_vectors: dict, captions_data: dict, tags_data: dict):
        """ Child class to create the employed IU X-Ray, inheriting the base class methods

        Args:
            image_vectors (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
            captions_data (dict): Dictionary with keys to be the ImageIDs and values the captions.
            tags_data (dict): Dictionary with keys to be the ImageIDs and values the tags embeddings.
        """
        super().__init__(image_vectors=image_vectors, captions_data=captions_data, clear_long_captions=False)
        self.tags_data = tags_data
        # get the splits
        self.train_dataset, self.dev_dataset, self.test_dataset = self.build_dataset()
        # build linguistic attributes
        self.vocab, self.tokenizer, self.word2idx, self.idx2word = super().build_vocab(training_captions=list(self.train_dataset[1].values()))
    
    def __str__(self) -> str:
        """ Python built-in function for prints

        Returns:
            str: A modified print.
        """
        text = f"Train: patients={len(self.train_dataset[0])}, captions={len(self.train_dataset[1])}, tags={len(self.train_dataset[2])}"
        text += f"\nDev: patients={len(self.dev_dataset[0])}, captions={len(self.dev_dataset[1])}, tags={len(self.dev_dataset[2])}"
        text += f"\nTest: patients={len(self.test_dataset[0])}, captions={len(self.test_dataset[1])}, tags={len(self.test_dataset[2])}"
        return text
    
    def get_splits_sets(self) ->tuple[list, list, list]:
        """ Fetches the data for each split set.

        Returns:
            tuple[list, list, list]: train_dataset, dev_dataset, test_dataset
        """
        return self.train_dataset, self.dev_dataset, self.test_dataset
    
    def get_tokenizer_utils(self) ->tuple[Vocabulary, Tokenizer, dict, dict]:
        """ Fetches the linguistic utilities.

        Returns:
            tuple[Vocabulary, Tokenizer, dict, dict]:  The Vocabulary object, the fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary. 
            The latters are mappers for words and index respectively
        """
        return self.vocab, self.tokenizer, self.word2idx, self.idx2word
        
    def __get_tags(self, _ids:list) -> dict:
        """ Fetches from the whole dataset the tags embeddings according to the utilised set.

        Args:
            _ids (list): Split set ids

        Returns:
            dict: Dictionary with keys to be the ImageIDs and values the tags embeddings
        """
         
        return { key:value for key, value in self.tags_data.items() if key in _ids}
        
    def build_dataset(self) -> tuple[list, list, list]:
        """ Begins the whole process for the dataset creation.

        Returns:
            tuple[list, list, list]: The training dataset, the validation dataset and the test dataset for our models.
            All sets are in list format. 
            1st index --> image vectors
            2nd index --> captions
            3rd index --> tags
        """
        # random split
        train_ids, dev_ids, test_ids = super().build_splits()

        # fetch images
        train_images = super().get_image_vectors(train_ids)
        dev_images = super().get_image_vectors(dev_ids)
        test_images = super().get_image_vectors(test_ids)
        # fetch captions
        train_captions = super().get_captions(train_ids)
        dev_captions = super().get_captions(dev_ids)
        test_captions = super().get_captions(test_ids)
        # apply preprocess to training captions
        train_captions_prepro = self.text_handler.preprocess_all(
            list(train_captions.values()))
            
        train_captions_prepro = dict( zip( train_ids, train_captions_prepro ) )
        # fetch tags
        train_tags = self.__get_tags(train_ids)
        dev_tags = self.__get_tags(dev_ids)
        test_tags = self.__get_tags(test_ids)
        # build data for each set    
        train_dataset = [train_images, train_captions_prepro, train_tags]
        dev_dataset = [dev_images, dev_captions, dev_tags]
        test_dataset = [test_images, test_captions, test_tags]


        return train_dataset, dev_dataset, test_dataset
    


class ImageCLEFDataset(Dataset):
    def __init__(self, image_vectors: dict, captions_data: dict):
        """_summary_

        Args:
            image_vectors (dict): _description_
            captions_data (dict): _description_
        """
        super().__init__(image_vectors=image_vectors, captions_data=captions_data, clear_long_captions=True)
        self.train_dataset, self.dev_dataset, self.test_dataset = self.build_dataset()
        
        self.vocab, self.tokenizer, self.word2idx, self.idx2word = super().build_vocab(training_captions=list(self.train_dataset[1].values()))
        
    def __str__(self) -> str:
        """ Python built-in function for prints

        Returns:
            str: A modified print.
        """
        text = f"Train: patients={len(self.train_dataset[0])}, captions={len(self.train_dataset[1])}"
        text += f"\nDev: patients={len(self.dev_dataset[0])}, captions={len(self.dev_dataset[1])}"
        text += f"\nTest: patients={len(self.test_dataset[0])}, captions={len(self.test_dataset[1])}"
        return text
    
    def get_splits_sets(self) -> tuple[list, list, list]:
        """ Fetches the data for each split set.

        Returns:
            tuple[list, list, list]: train_dataset, dev_dataset, test_dataset
        """
        return self.train_dataset, self.dev_dataset, self.test_dataset
    
    def get_tokenizer_utils(self) -> tuple[Vocabulary, Tokenizer, dict, dict]:
        """ Fetches the linguistic utilities.

        Returns:
            tuple[Vocabulary, Tokenizer, dict, dict]:  The Vocabulary object, the fitted tokenizer, the word-to-idx dictionary, and idx-to-word dictionary. 
            The latters are mappers for words and index respectively
        """
        return self.vocab, self.tokenizer, self.word2idx, self.idx2word
        
    def build_dataset(self) -> tuple[list, list, list]:
        """ Begins the whole process for the dataset creation.

        Returns:
            tuple[list, list, list]: The training dataset, the validation dataset and the test dataset for our models.
            All sets are in list format. 
            1st index --> image vectors
            2nd index --> captions
        """
        # random split
        train_ids, dev_ids, test_ids = super().build_splits()
        # fetch images
        train_images = super().get_image_vectors(train_ids)
        dev_images = super().get_image_vectors(dev_ids)
        test_images = super().get_image_vectors(test_ids)
        # fetch captions
        train_captions = super().get_captions(train_ids)
        dev_captions = super().get_captions(dev_ids)
        test_captions = super().get_captions(test_ids)
        
        # remove long outlier captions from training set
        train_modified_captions = super().delete_long_captions(data=train_captions)
        # get new training ids after removing
        train_new_ids = list(train_modified_captions.keys())
        train_new_images = {
            key:image_vector for key, image_vector in train_images.items() if key in train_new_ids
        }
        # apply preprocess to training captions
        train_captions_prepro = self.text_handler.preprocess_all(
            list(train_modified_captions.values()))
            
        train_captions_prepro = dict( zip( train_new_ids, train_captions_prepro ) )
        # build data for each set  
        train_dataset = [train_new_images, train_captions_prepro]
        dev_dataset = [dev_images, dev_captions]
        test_dataset = [test_images, test_captions]


        return train_dataset, dev_dataset, test_dataset


