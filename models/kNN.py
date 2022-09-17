from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm as progress_bar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from utils.dataset import Dataset, IuXrayDataset, ImageCLEFDataset

from modules.clinical_bert import ClinicalBERT

from tqdm import tqdm

class KNN:
    
    def __init__(self, dataset:Dataset, k:int=5, similarity_function:str='cosine', text_model:str='tf-idf'):
        self.dataset = dataset
        self.train_data, self.val_data, self.test_data = self.dataset.get_splits_sets()
        self.train_data
        self.K = k
        self.similarity = similarity_function if similarity_function in ['cosine', 'spatial_distance', 'matmul'] else 'cosine'
        self.text_model_emb = TfidfVectorizer() if text_model == 'tf-idf' else ClinicalBERT()
    
    def __set_current_k(self, k:int):
        self.K = k
        
    def __spatial(self, test_image_vector, train_image_vectors):
        sims = [1 - spatial.distance.cosine(test_image_vector, r) for r in train_image_vectors]
        topk = np.argsort(sims)[::-1][:self.K]
        return topk
    
    def __cosine(self, test_image_vector, train_image_vectors):
        sims = np.dot(test_image_vector, train_image_vectors.T) / (np.linalg.norm(test_image_vector)* np.linalg.norm(train_image_vectors.T))
        topk = np.argsort(sims).flatten()[::-1][:self.K]
        return topk
    
    def __matmul(self, test_image_vector, train_image_vectors):
         # Compute cosine similarity with every train image
        vec = test_image_vector / np.sum(test_image_vector)
        # Clone to do efficient mat mul dot
        test_mat = np.array([vec] * train_image_vectors.shape[0])
        sims = np.sum(test_mat * train_image_vectors, 1)
        topk = np.argsort(sims)[::-1][:self.K]
        return topk
        
        
    def __find_similar_images(self, test_image, train_images):
        
        if self.similarity == 'cosine':
            return self.__cosine(test_image_vector=test_image, train_image_vectors=train_images)
        elif self.similarity == 'spatial_distance':
            return self.__spatial(test_image_vector=test_image, train_image_vectors=train_images)
        else:
            return self.__matmul(test_image_vector=test_image, train_image_vectors=train_images)
    
    
    def __find_similar_caption(self, closer_captions:list, ensemble_mode:bool=False):
        
        if isinstance(self.text_model_emb, TfidfVectorizer):
            if ensemble_mode:
                caption_embedding_vectors = TfidfVectorizer().fit_transform(closer_captions)
            else:
                caption_embedding_vectors = self.text_model_emb.transform(closer_captions)
        else:
            caption_embedding_vectors = np.array( [ self.text_model_emb.vectorize(capt) for capt in closer_captions ] )

        similarity = cosine_similarity(caption_embedding_vectors, caption_embedding_vectors)
        s = np.sum(similarity, axis=0)
        best_similar = np.argmax(s)
        
        return closer_captions[best_similar]
        
    
    def __save_results(self, similarities_results:dict, dir_path:str):
        df = pd.DataFrame.from_dict(similarities_results, orient="index")
        df.to_csv(dir_path, sep='|', header=False)
        print('{self.K}-NN results saved in: {dir_path}')
        
    def __get_image_vectors(self, multi_modal:bool=False):
        
        if isinstance(self.dataset, IuXrayDataset):
            train_image_vectors = {key:np.concatenate((value[0], value[1]), axis=1) for key, value in self.train_data[0].items()}
            test_image_vectors = {key:np.concatenate((value[0], value[1]), axis=1) for key, value in self.test_data[0].items()}
            
            if multi_modal:
                multimodal_train_image_vectors = {key:np.concatenate((value, self.train_data[2][key]), axis=1) for key, value in train_image_vectors.items()}
                multimodal_test_image_vectors = {key:np.concatenate((value, self.test_data[2][key]), axis=1) for key, value in test_image_vectors.items()}
                del train_image_vectors, test_image_vectors
                
                train_image_vectors, test_image_vectors = multimodal_train_image_vectors, multimodal_test_image_vectors
        else:
            train_image_vectors, test_image_vectors = self.train_data[0], self.test_data[0]
                
        return train_image_vectors, test_image_vectors
        
    def generate_raw_vectors(self, train_image_vectors:dict, test_image_vectors:dict):
        raw_train = np.array(list(train_image_vectors.values()))
        raw_train = raw_train.reshape(raw_train.shape[0], raw_train.shape[2])

        raw_test = np.array(list(test_image_vectors.values()))
        
        return raw_train, raw_test
        
    def run_algo(self, multi_modal:bool=False, results_dir_path:str):
        
        if isinstance(self.text_model_emb, TfidfVectorizer):
            self.text_model_emb.fit(list(self.train_data[1].values()))
    
        # Load train data
        train_ids = list(self.train_data[0].keys())
        train_image_vectors, test_image_vectors = self.__get_image_vectors(multi_modal=multi_modal)
 
        # Save IDs and raw image vectors separately but aligned
        raw_train, raw_test = self.generate_raw_vectors(train_image_vectors=train_image_vectors, test_image_vectors=test_image_vectors)
        test_ids = list(test_image_vectors.keys())
        
        if self.similarity_function == 'matmul':
            raw_train = raw_train / np.array([np.sum(raw_train, 1)] * raw_train.shape[1]).transpose()

        sim_test_results = {}

        for i in progress_bar(range(len(raw_test)), desc=f'Running {self.K}-NN',  position=0, leave=True):
            vec = raw_test[i]
            topk = self.__find_similar_images(test_image=vec, train_images=raw_train)
            topKcaptions = [self.train_data[1][train_ids[topk[i]]] for i in range(self.K)]
            sim_test_results[test_ids[i]] = self.__find_similar_caption(closer_captions=topKcaptions)

        self.__save_results(similarities_results=sim_test_results, dir_path=results_dir_path)
        
    def ensemble(self, kNN_csvs_paths_list:list):
        models_dfs = [pd.read_csv(csv, sep='|', names=['ID', 'caption'])
                  for csv in kNN_csvs_paths_list]
        
        models_ids_captions_pairs = [ dict(zip(df.ID.to_list(), df.caption.to_list())) for df in models_dfs ]
        test_keys = list(models_ids_captions_pairs[0].keys())
        test_keys = sorted(test_keys)
        
        sim_test_results = {}
        for test_id in tqdm(test_keys, desc=f'Current k={self.K}'):
            current_captions_per_model = [models_ids_captions_pairs[i][test_id] for i in range(len(models_ids_captions_pairs))]
            sim_test_results[test_id] = self.__find_similar_caption(closer_captions=current_captions_per_model, ensemble_mode=True)
        
        self.__save_results(similarities_results=sim_test_results, dir_path='')
    
    def kNN_tuning(self, multi_modal:bool=False):
        tuning_cap = self.K
        for k in tqdm(range(1, tuning_cap+1), desc='Now tuning with k={k}', leave=True, position=0):
            self.__set_current_k(k=k)
            self.run_algo(multi_modal=multi_modal)
            
