# python modules
import argparse, os, pickle
import logging
import errno
from pprint import pprint
import pandas as pd
import json

# os modifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1" 

# tensorflow imports
import tensorflow
from tensorflow.keras.models import Model
physical_devices = tensorflow.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    USE_GPU = 1
else:
    USE_GPU = 0
for device in physical_devices:
    tensorflow.config.experimental.set_memory_growth(device, True)

# import utils and models    
from utils.metrics import compute_scores
from models import *
from modules.image_encoder import load_encoded_vecs
from utils import *
from utils.dataset import Dataset, IuXrayDataset, ImageCLEFDataset

# import nltk
import nltk
nltk.download('punkt', quiet=True)

# store dataset as well as results path
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_PATH = os.path.join(DATASET_PATH, 'results')


def make_dir(str_path:str) -> None:
    """ Try to make directory properly

    Args:
        str_path (str): The str path to create our directory
    """
    try:
        os.mkdir(str_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
# make results directory    
make_dir(RESULTS_PATH)
# begin loggings
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


class DiagnosticCaptioning:
    def __init__(self):
        """ Main application to implement my created systems
        """
        # fetch user cmd selections
        self.parser = argparse.ArgumentParser()
        self.parse_agrs()
        
    def parse_agrs(self) -> None:
        """ Parse all arguments selected in execution from the user
        """

        # Data loader settings
        self.parser.add_argument("--dataset", type=str, default="iu_xray", choices=["iu_xray", "imageclef"], help="the dataset to be used.")

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
        self.parser.add_argument("--batch_size", type=int, default=8, help="the number of samples for a batch",)
        self.parser.add_argument("--n_gpu", type=int, default=USE_GPU, help="the number of gpus to be used.")
        self.parser.add_argument("--epochs", type=int, default=100, help="the number of training epochs.")
        self.parser.add_argument("--save_dir",type=str, default="cnn_rnn",help="the path to save the models.")
        self.parser.add_argument("--early_stop", type=int, default=10, help="the patience of training.")
        
    def __init_device(self) -> tuple[bool, bool, bool]: 
        """ Private method to initialize the GPU usage if available else CPU

        Returns:
            tuple[bool, bool, bool]: Bool variables whether to use sinlge or multiple GPUs if available else CPU
        """
        use_CPU, use_GPU, use_multiGPU = False, False, False

        n_gpus = self.parser.parse_args().n_gpu

        # case GPU available
        if n_gpus > 0:
            if n_gpus == 1:
                use_GPU = True
            else:
                use_multiGPU = True
        else:
            # case CPU available
            use_CPU = True

        return use_CPU, use_GPU, use_multiGPU
        
    
    def __load_iuxray_data(self) -> tuple[dict, dict, dict]:
        """ Loads IU X-Ray dataset from directory

        Returns:
            tuple[dict, dict, dict]: Image vectors, captions and tags in dictionary format, with keys to be the Image IDs.
        """
        # get dataset path
        iu_xray_data_path = os.path.join(DATASET_PATH, 'iu_xray')
        iu_xray_images_data_path = os.path.join(iu_xray_data_path, 'two_images.json')
        iu_xray_captions_data_path = os.path.join(iu_xray_data_path, 'two_captions.json')
        iu_xray_tags_data_path = os.path.join(iu_xray_data_path, 'two_tags.json')
        
        # fetch images, captions, tags
        with open(iu_xray_images_data_path) as json_file:
            images = json.load(json_file)

        with open(iu_xray_captions_data_path) as json_file:
            captions = json.load(json_file)

        with open(iu_xray_tags_data_path) as json_file:
            tags = json.load(json_file)
            
        encoder = self.parser.parse_args().image_encoder
        
        image_encoded_vectors_path = os.path.join(iu_xray_data_path, f"{encoder}.pkl")
        # load image embeddings for the employed encoder      
        image_vecs = load_encoded_vecs(image_encoded_vectors_path)
        return image_vecs, captions, tags
    
    def __load_imageclef_data(self) -> tuple[dict, dict]:
        """ Loads ImageCLEF dataset from directory

        Returns:
            tuple[dict, dict]: Image vectors, captions in dictionary format, with keys to be the Image IDs.
        """
         # get dataset path
        imageclef_data_path = os.path.join(DATASET_PATH, 'imageCLEF')
        # fetch images, captions
        imageclef_image_captions_pairs = os.path.join(imageclef_data_path, 'Imageclef2022_dataset_all.csv')
        clef_df = pd.read_csv(imageclef_image_captions_pairs, sep='\t')
        captions = dict( zip( clef_df.ID.to_list(), clef_df.caption.to_list() ) )
        
            
        encoder = self.parser.parse_args().image_encoder
        
        image_encoded_vectors_path = os.path.join(imageclef_data_path, f"{encoder}.pkl")
        # load image embeddings for the employed encoder   
        image_vecs = load_encoded_vecs(image_encoded_vectors_path)
        return image_vecs, captions
    
    def __create_iu_xray_dataset(self, images:dict, captions:dict, tags:dict) -> IuXrayDataset:
        """ Builds the IU X-Ray dataset using the IuXrayDataset loader class

        Args:
            images (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
            captions (dict): Dictionary with keys to be the ImageIDs and values the captions.
            tags (dict): Dictionary with keys to be the ImageIDs and values the tags embeddings.

        Returns:
            IuXrayDataset: the employed IuXrayDataset object
        """
        iu_xray_dataset = IuXrayDataset(image_vectors=images, captions_data=captions, tags_data=tags)
        logging.info('IU-XRay dataset created.')
        logging.info(iu_xray_dataset)
        return iu_xray_dataset
    
    def __create_imageCLEF_dataset(self, images:dict, captions:dict) -> ImageCLEFDataset:
        """ Builds the ImageCLEF dataset using the ImageCLEFDataset loader class

        Args:
            images (dict): Dictionary with keys to be the ImageIDs and values the image embeddings.
            captions (dict): Dictionary with keys to be the ImageIDs and values the captions.

        Returns:
            ImageCLEFDataset: the employed ImageCLEFDataset object
        """
        imageCLEF_dataset = ImageCLEFDataset(image_vectors=images, captions_data=captions)
        logging.info('ImageCLEF dataset created.')
        logging.info(imageCLEF_dataset)
        return imageCLEF_dataset
    
    def train_cnn_rnn(self, dataset:Dataset) -> tuple[CNN_RNN, Model]:
        """ Begins the training process for the implemented CNN-RNN model
        More details are provided in my Thesis

        Args:
            dataset (Dataset): The employed dataset, i.e. IU X-Ray or ImageCLEF

        Returns:
            CNN_RNN, Model: The created CNN-RNN and the trained model
        """
        # fetch important args
        which_dataset = self.parser.parse_args().dataset
        epochs = self.parser.parse_args().epochs
        encoder = self.parser.parse_args().image_encoder
        max_length = self.parser.parse_args().max_length
        embedding_dim = self.parser.parse_args().embedding_dim
        ling_model = self.parser.parse_args().ling_model
        multi_modal = self.parser.parse_args().multi_modal
        logging.info(multi_modal)
        batch_size = self.parser.parse_args().batch_size
        
        # create the save directory for the model
        saved_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.parser.parse_args().save_dir)
        make_dir(saved_dir)
        # get the created vocabulary for our CNN-RNN
        _, tokenizer, word2idx, idx2word = dataset.get_tokenizer_utils()
        # make the model name according to arguments
        model_name = f'{which_dataset}_enc{encoder}_epochs{epochs}_maxlen{max_length}_embed{embedding_dim}_lingmodel{ling_model}_multimodal{multi_modal}'
        saved_model_name = os.path.join(saved_dir, model_name)
        logging.info(f'CNN-RNN model will be saved at: {saved_model_name}.h5')

        # build the CNN-RNN model
        SnT = CNN_RNN(tokenizer=tokenizer, word_to_idx=word2idx, 
                        idx_to_word=idx2word, max_length=max_length, 
                        embedding_dim=embedding_dim, ling_model=ling_model, 
                        multi_modal=multi_modal, loss="categorical_crossentropy")
        logging.info(f'Utilized vocabulary contains {SnT.vocab_size} words!')

        # get dataset splits   
        train, dev, test = dataset.get_splits_sets()
        
        # case IU X-Ray
        if which_dataset == 'iu_xray':
            # fetch all tags
            all_tags = dict(train[2], **dev[2])
            all_tags = dict(all_tags, **test[2])
            print('TAGS:', len(all_tags))
            # initialize the Multi-Modal version if user selected this kind of network
            tags_patient_pair = SnT.build_multimodal_encoder(all_tags)
            train_tags = {
                    key:value for key,value in tags_patient_pair.items() if key in train[1].keys()
            }      
            # store training data we want to utilise
            # 1st index --> image vectors
            # 2nd index --> captions
            # 3rd index --> tags
            train_data = [train[0], train[1], train_tags]
        else:
             # case ImageCLEF
             # store training data we want to utilise
             # 1st index --> image vectors
             # 2nd index --> captions
            train_data = [train[0], train[1]]
        # we use Adam as our optimizer for our training procedure  
        optimizer = tensorflow.keras.optimizers.Adam()
        
        # case IU X-Ray
        if which_dataset == 'iu_xray':
            # get the image embedding input shape. Every patient in IU X-Ray has 2 medical images. Thus, we read the shape from the first one.
            image_input_shape = list(train[0].values())[0][0].shape[1]
            # start train
            trained_model = SnT.train_iuxray_model(train_data=train_data, 
                                                    input_shape=(image_input_shape,), 
                                                    optimizer=optimizer, 
                                                    model_name=saved_model_name, 
                                                    n_epochs=epochs, 
                                                    batch_size=batch_size)
        else:
            # case ImageCLEF
              # get the image embedding input shape.
            image_input_shape = list(train[0].values())[0].shape[1]
            # start train
            trained_model = SnT.train_imageclef_model(train_data=train_data, 
                                                    input_shape=(image_input_shape,), 
                                                    optimizer=optimizer, 
                                                    model_name=saved_model_name, 
                                                    n_epochs=epochs, 
                                                    batch_size=batch_size)
        return SnT, trained_model
    
    def eval_cnn_rnn(self, cnn_rnn:CNN_RNN, model_to_eval:Model, dataset:Dataset) -> None:
        """ Begins the evaluation process for the trained model in the given dataset

        Args:
            cnn_rnn (CNN_RNN): The created CNN-RNN object that we will employ to apply our evaluation method
            model_to_eval (Model): The trained model that will be assessed
            dataset (Dataset): The employed dataset (IU X-Ray, ImageCLEF)
        """
        # fetch the generation algorithm (Greedy or Beam Search)
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
        # get the results path for our results dataframe
        dev_gold_path = os.path.join(RESULTS_PATH, 'dev_gold.csv')
        dev_pred_path = os.path.join(RESULTS_PATH, 'dev_pred.csv')
        
        # save gold truth captions
        df_gold = pd.DataFrame.from_dict(gold, orient="index")
        df_gold.to_csv(dev_gold_path, sep='|', header=False)
        # save predicted captions  
        df_pred = pd.DataFrame.from_dict(predicted, orient="index")
        df_pred.to_csv(dev_pred_path, sep='|', header=False)
        # score
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
        # get the results path for our results dataframe
        dev_gold_path = os.path.join(RESULTS_PATH, 'test_gold.csv')
        dev_pred_path = os.path.join(RESULTS_PATH, 'test_pred.csv')
         # save gold truth captions 
        df_gold = pd.DataFrame.from_dict(gold, orient="index")
        df_gold.to_csv(dev_gold_path, sep='|', header=False)
        # save predicted captions  
        df_pred = pd.DataFrame.from_dict(predicted, orient="index")
        df_pred.to_csv(dev_pred_path, sep='|', header=False)
        # score
        scores = compute_scores(gts=dev_gold_path, res=dev_pred_path, scores_filename='test_set_cnn_rnn_scores', save_scores=True)
        print('CNN_RNN scores in Test set')
        pprint(scores)
        
    
    def run_process(self) -> None:
        """ Begins the whole process according to the user settings.
        It employes the selected dataset in the selected model.
        For the latter we have CNN-RNN and kNN. More details for each of these models are provided in my Thesis.
        """
        which_dataset = self.parser.parse_args().dataset
        employed_model = self.parser.parse_args().model_choice
        
        # case IU X-Ray
        if which_dataset == "iu_xray":
            image_vecs, captions, tags = self.__load_iuxray_data()
            iu_xray_dataset = self.__create_iu_xray_dataset(image_vecs, captions, tags)
            
            # case CNN-RNN
            if employed_model == 'cnn_rnn':
                
                # Train CNN-RNN model
                cnn_rnn, trained_model = self.train_cnn_rnn(dataset=iu_xray_dataset)
                
                # Evaluate in model in Validation and Test set
                self.eval_cnn_rnn(cnn_rnn=cnn_rnn, model_to_eval=trained_model, dataset=iu_xray_dataset)
            else:
                 # case k-NN
                k = self.parser.parse_args().k
                multi_modal = self.parser.parse_args().multi_modal
                kNN = KNN(dataset=iu_xray_dataset, k=k, similarity_function='cosine', text_model='clinical_bert')
                # init the results path
                results_path = os.path.join(RESULTS_PATH, 'iuxray_{k}-NN_test_captions.csv')
                # and execute the k-NN algorithm
                kNN.run_algo(multi_modal = multi_modal, results_dir_path=results_path)
        else:
            # case ImageCLEF
            image_vecs, captions = self.__load_imageclef_data()
            imageCLEF_dataset = self.__create_imageCLEF_dataset(image_vecs, captions)
            
            # case CNN-RNN
            if employed_model == 'cnn_rnn':
                
                # Train CNN-RNN model
                cnn_rnn, trained_model = self.train_cnn_rnn(dataset=imageCLEF_dataset)
                
                # Evaluate in model in Validation and Test set
                self.eval_cnn_rnn(cnn_rnn=cnn_rnn, model_to_eval=trained_model, dataset=imageCLEF_dataset)
            else:
                 # case k-NN
                k = self.parser.parse_args().k
                kNN = KNN(dataset=imageCLEF_dataset, k=k, similarity_function='cosine', text_model='clinical_bert')
                # init the results path
                results_path = os.path.join(RESULTS_PATH, 'imageclef_{k}-NN_test_captions.csv')
                 # and execute the k-NN algorithm
                kNN.run_algo(results_dir_path=results_path)
                

    def main(self) -> None:
        """ Begins the process for this application
        """
        # flags for GPU and CPU usage
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
    
    
    