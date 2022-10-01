# os imports
import os

# numopy and progress bar imports
import numpy as np
from tqdm import tqdm
import pickle

# tensorflow imports
import tensorflow
# CNN image encoders
from tensorflow.keras.applications.densenet import DenseNet201 as dn201
from tensorflow.keras.applications.densenet import DenseNet121 as dn121
from tensorflow.keras.applications.densenet import DenseNet169 as dn169
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB5 as enb5
from tensorflow.keras.applications.efficientnet import EfficientNetB7 as enb7
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as rn50v2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as incres
# preprocessing functions
from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_preprocess

# layers imports
from tensorflow.keras.layers import Flatten, Dropout, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as img

# cotnet import
from keras_cv_attention_models import cotnet


def load_encoded_vecs(filename:str) -> dict:
    """ Loads the image embeddings for each image id, we extracted offline during my research

    Args:
        filename (str): the whole path of npy file

    Returns:
        dict: encoded_vectors from filename
    """
    with open(filename, 'rb') as f:
        print("Image Encoded Vectors loaded from directory path:", filename)
        return pickle.load(f)


def save_encoded_vecs(image_vecs:np.array, output_path:str, filename:str) -> None:
    """ Function which helps us to save the encoded images into a pickle file

    Args:
        image_vecs (np.array): the encoded images vectors that we extracted using the encode_images function
        output_path (str): the output path where we want to save our image embeddings
        filename (str): a name we want to use for our npy file (ex. densenet201_image_vecs). It's not necessary to write '.pkl' at the end!
    """
    path = output_path + filename + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(image_vecs, f, pickle.HIGHEST_PROTOCOL)
    print("Image Encoded Vectors stored in:", path)


class ImageEncoder:

    def __init__(self, encoder:str, images_dir_path:str, weights:str='imagenet'):
        """ This class helps us to extract image embeddings with different Keras CNNs.
        
        Args:
            encoder (str): encoder name you want to use (ex. densenet201 for DenseNet201)
            images_dir_path (str): The directory to store our extracted vectors
            weights (str, optional): the pretrained weights you want to use for your model. It's common to use imagenet as default pretrained weights.. Defaults to 'imagenet'.
        """
        self.encoder_weights = weights
        self.image_dir_path = images_dir_path

        # we extracted the last average pooling layer for each encoder

        if encoder == 'densenet201':
            self.image_shape = 224
            self.preprocess = 'densenet'
            model = dn201(include_top=True, weights=self.encoder_weights,
                          input_shape=(self.image_shape, self.image_shape, 3))
            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)

        elif encoder == 'densenet121':
            self.image_shape = 224
            self.preprocess = 'densenet'
            model = dn121(include_top=True, weights=self.encoder_weights,
                          input_shape=(self.image_shape, self.image_shape, 3))

            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)

        elif encoder == 'densenet169':
            self.image_shape = 224
            self.preprocess = 'densenet'
            model = dn169(include_top=True, weights=self.encoder_weights,
                          input_shape=(self.image_shape, self.image_shape, 3))
            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)

        elif encoder == 'efficientnet5':
            self.image_shape = 456
            self.preprocess = 'efficientnet'
            model = enb5(include_top=True, weights=self.encoder_weights,
                         input_shape=(self.image_shape, self.image_shape, 3))
            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)

        elif encoder == 'efficientnet0':
            self.image_shape = 224
            self.preprocess = 'efficientnet'
            model = EfficientNetB0(include_top=True, weights=self.encoder_weights,
                                   input_shape=(self.image_shape, self.image_shape, 3))
            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)

        elif encoder == 'resnet50v2':
            self.image_shape = 224
            self.preprocess = 'resnet'
            model = rn50v2(include_top=True, weights=self.encoder_weights,
                           input_shape=(self.image_shape, self.image_shape, 3))
            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)
        elif encoder == 'inceptionresnet':
            self.image_shape = 299
            self.preprocess = 'inceptionresnet'
            model = incres(include_top=True, weights=self.encoder_weights,
                           input_shape=(self.image_shape, self.image_shape, 3))
            self.image_encoder = Model(
                inputs=model.input, outputs=model.get_layer('avg_pool').output)

        elif encoder == 'cotnet':
            self.image_shape = 224
            self.preprocess = 'cotnet'
            model = cotnet.CotNet50(pretrained="imagenet", num_classes=0)
            self.image_encoder = Model(
                inputs=model.input, outputs=model.output)

        else:
            print("You have to initialize a valid version of image encoder\n"
                  "Choices are: [densenet201, densenet121, densenet169, efficientnet0, efficientnet5, resnet50v2, inceptionresnet, cotnet]")
            print("Exiting...")
            return

    def get_preprocessor(self) -> str:
        """ Gets the pre-processing function

        Returns:
            str: The pre-processing name we initialized
        """
        return self.preprocess

    def get_image_shape(self) -> int:
        """ Gets the input shape

        Returns:
            int: The input shape for the employed encoder
        """
        return self.image_shape

    def get_image_encoder(self) -> Model:
        """ Gets the image encoder we built

        Returns:
            Model: The CNN encoder
        """
        return self.image_encoder

    def get_images_dirpath(self) -> str:
        """ Gets the image directory path to store our vectors

        Returns:
            str: The image directory path
        """
        return self.image_dir_path

    def encode(self, _image:str, verbose:int=0) -> np.array:
        """ Loads an image and it passes it in CNN encoder to extract its image embeddings.

        Args:
            _image (str): The image id, for which its image we want to encode
            verbose (int, optional): If we want to display the extraction. Defaults to 0.

        Returns:
            np.array: The encoded version of the given image
        """
        # case CoTNet
        if self.get_preprocessor() == 'cotnet':
            image = img.load_img(self.image_dir_path + _image + '.jpg')
            image_array = img.img_to_array(image)

            imm = tensorflow.keras.applications.imagenet_utils.preprocess_input(image_array, mode='torch')
            image_encoded = self.image_encoder(
                tensorflow.expand_dims(tensorflow.image.resize(imm, self.image_encoder.input_shape[1:3]), 0)).numpy()
        else:
            # case othe encoders
            # load the image and convert it to np.array
            image = img.load_img(self.image_dir_path + _image + '.jpg',
                                 target_size=(self.image_shape, self.image_shape))
            image_array = img.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            # pre-process array in order to fit with the employed encoder
            if self.get_preprocessor() == 'densenet':
                preprocessed_image_array = dense_preprocess(image_array)
            elif self.get_preprocessor() == 'efficientnet':
                preprocessed_image_array = efficient_preprocess(image_array)
            elif self.get_preprocessor() == 'resnet':
                preprocessed_image_array = resnet_preprocess(image_array)
            elif self.get_preprocessor() == 'inceptionresnet':
                preprocessed_image_array = inceptionresnet_preprocess(
                    image_array)
            # extract image embeddings
            image_encoded = self.image_encoder.predict(preprocessed_image_array, verbose=verbose)
        return image_encoded

    def encode_images(self, images:list) -> np.array:
        """ Loads an image list with image ids, and extract their image embeddings

        Args:
            images (list): Image IDs list

        Returns:
            np.array: All image vectors
        """
        image_vecs = {_image: self.encode(_image) for _image in
                      tqdm(images, desc="Encoding images", position=0, leave=True)}
        return image_vecs
