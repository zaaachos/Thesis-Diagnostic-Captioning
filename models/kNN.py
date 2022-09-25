# ds imports
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm as progress_bar

# similarity functions imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

# import datasets
from utils.dataset import Dataset, IuXrayDataset, ImageCLEFDataset

# run on torch conda enviroment in order to use ClinicalBERT. It requires torch!
# from modules.clinical_bert import ClinicalBERT

class KNN:
    
    def __init__(self, dataset:Dataset, k:int=5, similarity_function:str='cosine', text_model:str='tf-idf'):
        """ Retrieval-based approach that utilises the k-NN (k Nearest Neighbors) algorithm.

        Args:
            dataset (Dataset): The dataset to be used. 
            k (int, optional): The k number of the k-NN, to retrieve the k nearest neighbors. Defaults to 5.
            similarity_function (str, optional): The similarity function that k-NN utilises in order to retrieve the closer images. Defaults to 'cosine'.
            text_model (str, optional): The text embedding model that our model employs to find the appropriate caption for the given test instance. Defaults to 'tf-idf'.
        """
        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = self.dataset.get_splits_sets()
        self.train_data
        self.K = k
        self.similarity = similarity_function if similarity_function in ['cosine', 'spatial_distance', 'matmul'] else 'cosine'
        self.text_model_emb = TfidfVectorizer() # if text_model == 'tf-idf' else ClinicalBERT() # run on torch conda enviroment in order to use ClinicalBERT. It requires torch!
    
    def __set_current_k(self, k:int)->None:
        """ Setter for k argument

        Args:
            k (int): The k number of the k-NN, for the k neighbors
        """
        self.K = k
        
    def __spatial(self, test_image_vector:np.array, train_image_vectors:np.array) -> np.array:
        """ It calucaltes the similarity between the test image embedding and training image embeddings using the scipy's spatial distanc

        Args:
            test_image_vector (np.array): The test image embedding
            train_image_vectors (np.array): The training images embeddings

        Returns:
            np.array: The top K nearest neighbors according to the calculation
        """
        sims = [1 - spatial.distance.cosine(test_image_vector, r) for r in train_image_vectors]
        topk = np.argsort(sims)[::-1][:self.K]
        return topk
    
    def __cosine(self, test_image_vector:np.array, train_image_vectors:np.array) -> np.array:
        """ It calucaltes the similarity between the test image embedding and training image embeddings using the cosine similarity formula.

        Args:
            test_image_vector (np.array): The test image embedding
            train_image_vectors (np.array): The training images embeddings

        Returns:
            np.array: The top K nearest neighbors according to the calculation
        """
        sims = np.dot(test_image_vector, train_image_vectors.T) / (np.linalg.norm(test_image_vector)* np.linalg.norm(train_image_vectors.T))
        topk = np.argsort(sims).flatten()[::-1][:self.K]
        return topk
    
    def __matmul(self, test_image_vector:np.array, train_image_vectors:np.array) -> np.array:
        """ It calucaltes the similarity between the test image embedding and training image embeddings using the matrix multiplication.

        Args:
            test_image_vector (np.array): The test image embedding
            train_image_vectors (np.array): The training images embeddings

        Returns:
            np.array: The top K nearest neighbors according to the calculation
        """
         # Compute cosine similarity with every train image
        vec = test_image_vector / np.sum(test_image_vector)
        # Clone to do efficient mat mul dot
        test_mat = np.array([vec] * train_image_vectors.shape[0])
        sims = np.sum(test_mat * train_image_vectors, 1)
        topk = np.argsort(sims)[::-1][:self.K]
        return topk
        
        
    def __find_similar_images(self, test_image:np.array, train_images:np.array) -> np.array:
        """ It calucaltes the similarity between the test image embedding and training image embeddings using the similarity function.
        It returns the K most similar images according to the similary.

        Args:
            test_image_vector (np.array): The test image embedding
            train_image_vectors (np.array): The training images embeddings

        Returns:
            np.array: The top K nearest neighbors according to the calculation
        """
        
        if self.similarity == 'cosine':
            return self.__cosine(test_image_vector=test_image, train_image_vectors=train_images)
        elif self.similarity == 'spatial_distance':
            return self.__spatial(test_image_vector=test_image, train_image_vectors=train_images)
        else:
            return self.__matmul(test_image_vector=test_image, train_image_vectors=train_images)
    
    
    def __find_similar_caption(self, closer_captions:list, ensemble_mode:bool=False) -> str:
        """ It finds the most appropriate caption using the top K nearest training images, and assigns this caption as the prediciton for the given test image.
        It uses the Consensus Caption formula which is defined in the 1st approach of my Thesis.

        Args:
            closer_captions (list): The top K nearest captions of the top K nearest training instances
            ensemble_mode (bool, optional): If we use the K-NN ensemble model. Defaults to False.

        Returns:
            str: The caption closest to the centroid of the k retrieved captions.
        """
        
        # case TF-IDF
        if isinstance(self.text_model_emb, TfidfVectorizer):
            if ensemble_mode:
                caption_embedding_vectors = TfidfVectorizer().fit_transform(closer_captions)
            else:
                caption_embedding_vectors = self.text_model_emb.transform(closer_captions)
        else:
             # case ClinicalBERT
             # begin the vectorization directly to the k retrieved captions.
            caption_embedding_vectors = np.array( [ self.text_model_emb.vectorize(capt) for capt in closer_captions ] )

        # CC method
        similarity = cosine_similarity(caption_embedding_vectors, caption_embedding_vectors)
        s = np.sum(similarity, axis=0)
        best_similar = np.argmax(s)
        
        # fetch the closer caption
        return closer_captions[best_similar]
        
    
    def __save_results(self, similarities_results:dict, dir_path:str) -> None:
        """ Stores the k-NN test images, predicted captions pairs to a CSV file.

        Args:
            similarities_results (dict): The k-NN test images, predicted captions pairs.
            dir_path (str): The path of the system in which we are going to store the results.
        """
        df = pd.DataFrame.from_dict(similarities_results, orient="index")
        df.to_csv(dir_path, sep='|', header=False)
        print('{self.K}-NN results saved in: {dir_path}')
        
    def __get_image_vectors(self, multi_modal:bool=False) -> tuple[dict, dict]:
        """ Fetches and creates the traininig and test image embeddings, according to the employed Dataset (i.e. IU X-Ray or ImageCLEF)
        Dataset info:
        data[0] --> image embeddings
        data[1] --> captions
        data[2] --> tags (if IU X-Ray is employed)

        Args:
            multi_modal (bool, optional): If we would like to run the ensemble K-NN model. Defaults to False.

        Returns:
            tuple[dict, dict]: Training and test image embeddings dictionaries.
        """
        
        # case IU X-Ray
        if isinstance(self.dataset, IuXrayDataset):
            # we use both 2-patient images (frontal and lateral) as input. Thus, we create an image input by concatenating both x-rays.
            train_image_vectors = {key:np.concatenate((value[0], value[1]), axis=1) for key, value in self.train_data[0].items()}
            test_image_vectors = {key:np.concatenate((value[0], value[1]), axis=1) for key, value in self.test_data[0].items()}
            
            # case multi-modal
            if multi_modal:
                # now we use the concatenation of both images and tags as input for our model.
                multimodal_train_image_vectors = {key:np.concatenate((value, self.train_data[2][key]), axis=1) for key, value in train_image_vectors.items()}
                multimodal_test_image_vectors = {key:np.concatenate((value, self.test_data[2][key]), axis=1) for key, value in test_image_vectors.items()}
                del train_image_vectors, test_image_vectors
                
                train_image_vectors, test_image_vectors = multimodal_train_image_vectors, multimodal_test_image_vectors
        else:
            # case ImageCLEF
            train_image_vectors, test_image_vectors = self.train_data[0], self.test_data[0]
                
        return train_image_vectors, test_image_vectors
        
    def generate_raw_vectors(self, train_image_vectors:dict, test_image_vectors:dict) ->tuple[np.array, np.array]:
        """ Gets the raw image embeddings for both training and test set.

        Args:
            train_image_vectors (np.array): The training images embeddings
            test_image_vectors (np.array): The test images embeddings

        Returns:
            tuple[np.array, np.array]: The raw vectors for both training and test set.
        """
        raw_train = np.array(list(train_image_vectors.values()))
        raw_train = raw_train.reshape(raw_train.shape[0], raw_train.shape[2])

        raw_test = np.array(list(test_image_vectors.values()))
        
        return raw_train, raw_test
        
    def run_algo(self, multi_modal:bool=False, results_dir_path:str="results") -> None:
        """ It executes the k-NN algorithm depending on the k number and similarity function selected.

        Args:
            multi_modal (bool, optional): If we would like to run the ensemble K-NN model. Defaults to False.
            results_dir_path (str): The path of the system in which we are going to store the results.
        """
        # first fit the TF-IDF model in training captions, to produce appropriate value data.
        if isinstance(self.text_model_emb, TfidfVectorizer):
            self.text_model_emb.fit(list(self.train_data[1].values()))
    
        # Load train data
        train_ids = list(self.train_data[0].keys())
        train_image_vectors, test_image_vectors = self.__get_image_vectors(multi_modal=multi_modal)
 
        # Save IDs and raw image vectors separately but aligned
        raw_train, raw_test = self.generate_raw_vectors(train_image_vectors=train_image_vectors, test_image_vectors=test_image_vectors)
        test_ids = list(test_image_vectors.keys())
        
        # case matrix multiplication similarity function, modify image vectors.
        if self.similarity_function == 'matmul':
            raw_train = raw_train / np.array([np.sum(raw_train, 1)] * raw_train.shape[1]).transpose()

        sim_test_results = {}

        for i in progress_bar(range(len(raw_test)), desc=f'Running {self.K}-NN',  position=0, leave=True):
            vec = raw_test[i]
            topk = self.__find_similar_images(test_image=vec, train_images=raw_train)
            topKcaptions = [self.train_data[1][train_ids[topk[i]]] for i in range(self.K)]
            sim_test_results[test_ids[i]] = self.__find_similar_caption(closer_captions=topKcaptions)

        self.__save_results(similarities_results=sim_test_results, dir_path=results_dir_path)
        
    def ensemble(self, kNN_csvs_paths_list:list, results_csv_path:str) -> None:
        """ The ensembles of K-NN. It executes the k-NN algorithm for different image encoders.

        Args:
            kNN_csvs_paths_list (list): The path of the system in which the results of different kNNs are stored.
            results_dir_path (str): The path of the system in which we are going to store the results.
        """
        # fetch all DataFrames of results for each image encoder.
        models_dfs = [pd.read_csv(csv, sep='|', names=['ID', 'caption'])
                  for csv in kNN_csvs_paths_list]
        
        # create test id --> all predicted captions list, for each test id
        models_ids_captions_pairs = [ dict(zip(df.ID.to_list(), df.caption.to_list())) for df in models_dfs ]
        test_keys = list(models_ids_captions_pairs[0].keys())
        test_keys = sorted(test_keys)
        
        sim_test_results = {}
        for test_id in progress_bar(test_keys, desc=f'Current k={self.K}'):
            # current predicted captions for currenct test ID
            current_captions_per_model = [models_ids_captions_pairs[i][test_id] for i in range(len(models_ids_captions_pairs))]
            # get the most appropriate caption using the CC method.
            sim_test_results[test_id] = self.__find_similar_caption(closer_captions=current_captions_per_model, ensemble_mode=True)
        
        # save the ensemble predictions.
        self.__save_results(similarities_results=sim_test_results, dir_path=results_csv_path)
    
    def kNN_tuning(self, multi_modal:bool=False, results_csv_path:str="results") -> None:
        """ Starts the tuning of the neural retrival approach of K-NN, with k to be in range [1, self.K]

        Args:
            multi_modal (bool, optional): If we would like to tune the ensemble K-NN model. Defaults to False.
        """
        tuning_cap = self.K
        for k in progress_bar(range(1, tuning_cap+1), desc='Now tuning with k={k}', leave=True, position=0):
            self.__set_current_k(k=k)       # k step
            current_k_results_path = os.path.join(results_csv_path, 'iuxray_{k}-NN_test_captions.csv')
            self.run_algo(multi_modal=multi_modal, results_dir_path=current_k_results_path)
            
