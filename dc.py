import argparse, os, pickle
import logging
import errno


from utils.metrics import compute_scores

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

physical_devices = tensorflow.config.list_physical_devices('GPU') 
for device in physical_devices:
    tensorflow.config.experimental.set_memory_growth(device, True)
    
import pandas as pd
from models import *
from modules.image_encoder import load_encoded_vecs
from utils import *

from tqdm import tqdm
import numpy as np
import random
from pprint import pprint
# from evaluator import *

import json
from utils.dataset import IuXrayDataset, ImageCLEFDataset

import nltk
nltk.download('punkt', quiet=True)

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_PATH = os.path.join(DATASET_PATH, 'results')

def make_dir(str_path:str):
    try:
        os.mkdir(str_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
make_dir(RESULTS_PATH)

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


class DiagnosticCaptioning:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parse_agrs()
        
    def parse_agrs(self):

        # Data loader settings
        self.parser.add_argument("--dataset", type=str, default="imageclef", choices=["iu_xray", "imageclef"], help="the dataset to be used.")

        # Employing model
        self.parser.add_argument("--model_choice", type=str, default="cnn_rnn", choices=["cnn_rnn", "knn"], help="Which model to employ for testing.")
        self.parser.add_argument("--k", type=int, default=5, help="k for K-NN")
        
        # Captions settings
        self.parser.add_argument("--max_length", type=int, default=40, help="the maximum sequence length of the reports.")
        self.parser.add_argument("--threshold", type=int, default=3, help="the cut off frequency for the words.")

        # Model settings (for layers)
        self.parser.add_argument("--image_encoder", type=str, default="densenet121", help="the visual encoder to be used.")
        self.parser.add_argument("--embedding_dim", type=int, default=100, help="the embedding dimension for Embedding Layers.")
        self.parser.add_argument("--ling_model", type=str, default="gru", choices=["gru", "lstm", "bigru"], help="the Linguistig Model (RNN) for Decoder module as well as Text encoder.")

        # Model settings
        self.parser.add_argument("--multi_modal", type=bool, default=False, help="if to use multi_modal as our model for CNN-RNN only.")
        self.parser.add_argument("--dropout", type=float, default=0.2, help="the dropout rate of our model.")

        # Generate text apporach related
        self.parser.add_argument("--sample_method", type=str, default="greedy", choices=["greedy", "beam_3", "beam_5", "beam_7"], help="the sample methods to sample a report.")
        
        # Trainer settings
        self.parser.add_argument("--batch_size", type=int, default=32, help="the number of samples for a batch",)
        self.parser.add_argument("--n_gpu", type=int, default=1, help="the number of gpus to be used.")
        self.parser.add_argument("--epochs", type=int, default=100, help="the number of training epochs.")
        self.parser.add_argument("--save_dir",type=str, default="cnn_rnn",help="the path to save the models.")
        self.parser.add_argument("--early_stop", type=int, default=10, help="the patience of training.")
        
    def __init_device(self):
        use_CPU, use_GPU, use_multiGPU = False, False, False

        n_gpus = self.parser.parse_args().n_gpu

        if n_gpus > 0:
            if n_gpus == 1:
                use_GPU = True
            else:
                use_multiGPU = True
        else:
            use_CPU = True

        return use_CPU, use_GPU, use_multiGPU
        
    
    def __load_iuxray_data(self):
        iu_xray_data_path = os.path.join(DATASET_PATH, 'iu_xray')
        print(iu_xray_data_path)
        iu_xray_images_data_path = os.path.join(iu_xray_data_path, 'two_images.json')
        print(iu_xray_images_data_path)
        iu_xray_captions_data_path = os.path.join(iu_xray_data_path, 'two_captions.json')
        iu_xray_tags_data_path = os.path.join(iu_xray_data_path, 'two_tags.json')
        
        with open(iu_xray_images_data_path) as json_file:
            images = json.load(json_file)

        with open(iu_xray_captions_data_path) as json_file:
            captions = json.load(json_file)

        with open(iu_xray_tags_data_path) as json_file:
            tags = json.load(json_file)
            
        encoder = self.parser.parse_args().image_encoder
        
        image_encoded_vectors_path = os.path.join(iu_xray_data_path, f"{encoder}.pkl")

        image_vecs = load_encoded_vecs(image_encoded_vectors_path)
        return image_vecs, captions, tags
    
    def __load_imageclef_data(self):
        imageclef_data_path = os.path.join(DATASET_PATH, 'imageCLEF')
        print(imageclef_data_path)
        imageclef_image_captions_pairs = os.path.join(imageclef_data_path, 'Imageclef2022_dataset_all.csv')
        clef_df = pd.read_csv(imageclef_image_captions_pairs, sep='\t')
        captions = dict( zip( clef_df.ID.to_list(), clef_df.caption.to_list() ) )
        
            
        encoder = self.parser.parse_args().image_encoder
        
        image_encoded_vectors_path = os.path.join(imageclef_data_path, f"{encoder}.pkl")

        image_vecs = load_encoded_vecs(image_encoded_vectors_path)
        return image_vecs, captions
    
    def __create_iu_xray_dataset(self, images:dict, captions:dict, tags:dict):
        iu_xray_dataset = IuXrayDataset(image_vectors=images, captions_data=captions, tags_data=tags)
        logging.info('IU-XRay dataset created.')
        logging.info(iu_xray_dataset)
        return iu_xray_dataset
    
    def __create_imageCLEF_dataset(self, images:dict, captions:dict):
        imageCLEF_dataset = ImageCLEFDataset(image_vectors=images, captions_data=captions)
        logging.info('ImageCLEF dataset created.')
        logging.info(imageCLEF_dataset)
        return imageCLEF_dataset
    
    def train_cnn_rnn(self, dataset):
        which_dataset = self.parser.parse_args().dataset
        epochs = self.parser.parse_args().epochs
        encoder = self.parser.parse_args().image_encoder
        max_length = self.parser.parse_args().max_length
        embedding_dim = self.parser.parse_args().embedding_dim
        ling_model = self.parser.parse_args().ling_model
        multi_modal = self.parser.parse_args().multi_modal
        batch_size = self.parser.parse_args().batch_size
        
        saved_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.parser.parse_args().save_dir)
        make_dir(saved_dir)
        
        _, tokenizer, word2idx, idx2word = dataset.get_tokenizer_utils()
        model_name = f'{which_dataset}_enc{encoder}_epochs{epochs}_maxlen{max_length}_embed{embedding_dim}_lingmodel{ling_model}_multimodal{multi_modal}'
        saved_model_name = os.path.join(saved_dir, model_name)
        logging.info(f'CNN-RNN model will be saved at: {saved_model_name}.h5')
        SnT = CNN_RNN(tokenizer=tokenizer, word_to_idx=word2idx, 
                        idx_to_word=idx2word, max_length=max_length, 
                        embedding_dim=embedding_dim, ling_model=ling_model, 
                        multi_modal=multi_modal, loss="categorical_crossentropy")
        logging.info(f'Utilized vocabulary contains {SnT.vocab_size} words!')
                
        train, dev, test = dataset.get_splits_sets()
        
        if which_dataset == 'iu_xray':
            all_tags = dict(train[2], **dev[2])
            all_tags = dict(all_tags, **test[2])
            print('TAGS:', len(all_tags))
            tags_patient_pair = SnT.build_multimodal_encoder(all_tags)
            train_tags = {
                    key:value for key,value in tags_patient_pair.items() if key in train[1].keys()
            }   
            
                
            dev_tags = {
                    key:value for key,value in tags_patient_pair.items() if key in dev[1].keys()
            }    
            
            train_data = [train[0], train[1], train_tags]
        else:
            train_data = [train[0], train[1]]
                           
        optimizer = tensorflow.keras.optimizers.Adam()
        
        if which_dataset == 'iu_xray':
            image_input_shape = list(train[0].values())[0][0].shape[1]
            trained_model = SnT.train_iuxray_model(train_data=train_data, 
                                                    input_shape=(image_input_shape,), 
                                                    optimizer=optimizer, 
                                                    model_name=saved_model_name, 
                                                    n_epochs=epochs, 
                                                    batch_size=batch_size)
        else:
            image_input_shape = list(train[0].values())[0].shape[1]
            trained_model = SnT.train_imageclef_model(train_data=train_data, 
                                                    input_shape=(image_input_shape,), 
                                                    optimizer=optimizer, 
                                                    model_name=saved_model_name, 
                                                    n_epochs=epochs, 
                                                    batch_size=batch_size)
        return SnT, trained_model
    
    def eval_cnn_rnn(self, cnn_rnn:CNN_RNN, model_to_eval, dataset):
        generate_choice = self.parser.parse_args().sample_method
        which_dataset = self.parser.parse_args().dataset
        
        # fetch dev, test set
        _, dev, test = dataset.get_splits_sets()
        
        # first evaluate our model in validation set
        if which_dataset == 'iu_xray':
            gold, predicted = cnn_rnn.evaluate_model(model=model_to_eval, 
                                                            test_captions=dev[1], 
                                                            test_images=dev[0], 
                                                            test_tags=dev[2], 
                                                            evaluator_choice=generate_choice)
        else:
            gold, predicted = cnn_rnn.evaluate_model(model=model_to_eval, 
                                                            test_captions=dev[1], 
                                                            test_images=dev[0], 
                                                            test_tags=None, 
                                                            evaluator_choice=generate_choice)
        dev_gold_path = os.path.join(RESULTS_PATH, 'dev_gold.csv')
        dev_pred_path = os.path.join(RESULTS_PATH, 'dev_pred.csv')
            
        df_gold = pd.DataFrame.from_dict(gold, orient="index")
        df_gold.to_csv(dev_gold_path, sep='|', header=False)
            
        df_pred = pd.DataFrame.from_dict(predicted, orient="index")
        df_pred.to_csv(dev_pred_path, sep='|', header=False)
        
        scores = compute_scores(gts=dev_gold_path, res=dev_pred_path, scores_filename='dev_set_cnn_rnn_scores', save_scores=True)
        print('CNN_RNN scores in Validation set')
        pprint(scores)
        
        # Now evaluate our model in test set
        if which_dataset == 'iu_xray':
            gold, predicted = cnn_rnn.evaluate_model(model=model_to_eval, 
                                                            test_captions=test[1], 
                                                            test_images=test[0], 
                                                            test_tags=test[2],
                                                            eval_dataset=which_dataset,
                                                            evaluator_choice=generate_choice)
        else:
            gold, predicted = cnn_rnn.evaluate_model(model=model_to_eval, 
                                                            test_captions=test[1], 
                                                            test_images=test[0], 
                                                            test_tags=None,
                                                            eval_dataset=which_dataset, 
                                                            evaluator_choice=generate_choice)
        dev_gold_path = os.path.join(RESULTS_PATH, 'test_gold.csv')
        dev_pred_path = os.path.join(RESULTS_PATH, 'test_pred.csv')
            
        df_gold = pd.DataFrame.from_dict(gold, orient="index")
        df_gold.to_csv(dev_gold_path, sep='|', header=False)
            
        df_pred = pd.DataFrame.from_dict(predicted, orient="index")
        df_pred.to_csv(dev_pred_path, sep='|', header=False)
        
        scores = compute_scores(gts=dev_gold_path, res=dev_pred_path, scores_filename='test_set_cnn_rnn_scores', save_scores=True)
        print('CNN_RNN scores in Test set')
        pprint(scores)
        
    
    def run_process(self):
        which_dataset = self.parser.parse_args().dataset
        employed_model = self.parser.parse_args().model_choice
        
        
        if which_dataset == "iu_xray":
            image_vecs, captions, tags = self.__load_iuxray_data()
            iu_xray_dataset = self.__create_iu_xray_dataset(image_vecs, captions, tags)
            
            
            if employed_model == 'cnn_rnn':
                
                # Train CNN-RNN model
                cnn_rnn, trained_model = self.train_cnn_rnn(dataset=iu_xray_dataset)
                
                # Evaluate in model in Validation and Test set
                self.eval_cnn_rnn(cnn_rnn=cnn_rnn, model_to_eval=trained_model, dataset=iu_xray_dataset)
            else:
                k = self.parser.parse_args().k
                multi_modal = self.parser.parse_args().multi_modal
                kNN = KNN(dataset=iu_xray_dataset, k=k, similarity_function='cosine', text_model='clinical_bert')
                results_path = os.path.join(RESULTS_PATH, 'iuxray_{k}-NN_test_captions.csv')
                kNN.run_algo(multi_modal = multi_modal, results_dir_path=results_path)
        else:
            image_vecs, captions = self.__load_imageclef_data()
            imageCLEF_dataset = self.__create_imageCLEF_dataset(image_vecs, captions)
            
            
            if employed_model == 'cnn_rnn':
                
                # Train CNN-RNN model
                cnn_rnn, trained_model = self.train_cnn_rnn(dataset=imageCLEF_dataset)
                
                # Evaluate in model in Validation and Test set
                self.eval_cnn_rnn(cnn_rnn=cnn_rnn, model_to_eval=trained_model, dataset=imageCLEF_dataset)
            else:
                k = self.parser.parse_args().k
                kNN = KNN(dataset=imageCLEF_dataset, k=k, similarity_function='cosine', text_model='clinical_bert')
                results_path = os.path.join(RESULTS_PATH, 'imageclef_{k}-NN_test_captions.csv')
                kNN.run_algo(results_dir_path=results_path)
                

    def main(self):
    
        use_CPU, use_GPU, _ = self.__init_device()

        if use_CPU:
            logging.info('Using CPU')
            with tensorflow.device("/device:GPU:0"):
                self.run_process()
        elif use_GPU:
            logging.info('Using single GPU')
            with tensorflow.device("/device:GPU:0"):
                self.run_process()
        else:
            logging.info('Using multi GPU')
            tensorflow.debugging.set_log_device_placement(True)
            gpus = tensorflow.config.list_logical_devices("GPU")
            strategy = tensorflow.distribute.MirroredStrategy(gpus)
            with strategy.scope():
                self.run_process()
        

if __name__ == '__main__':
    logging.info(DATASET_PATH)
    dc = DiagnosticCaptioning()
    dc.main()
    
    
    