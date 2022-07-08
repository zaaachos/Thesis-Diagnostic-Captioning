import pickle
from tqdm import tqdm
import numpy as np
import tensorflow

from tensorflow.keras.preprocessing import image as img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as incres
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as rn50v2
from tensorflow.keras.applications.efficientnet import EfficientNetB7 as enb7
from tensorflow.keras.applications.efficientnet import EfficientNetB5 as enb5
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.densenet import DenseNet169 as dn169
from tensorflow.keras.applications.densenet import DenseNet121 as dn121
from tensorflow.keras.applications.densenet import DenseNet201 as dn201

# from keras_cv_attention_models import cotnet


def save_encoded_vecs(image_vecs, output_path, filename):
    """
    Function which helps us to save the encoded images into a pickle file
    :param image_vecs: the encoded images vectors that we extracted using the encode_images function
    :param output_path: the output path where we want to save our image embeddings
    :param filename: a name we want to use for our npy file (ex. densenet201_image_vecs). It's not necessary to write '.pkl' at the end!
    """
    path = output_path + filename + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(image_vecs, f, pickle.HIGHEST_PROTOCOL)
    print("Image Encoded Vectors stored in:", path)


def load_encoded_vecs(filename):
    """
    :param filename: the whole path of npy file
    :return: encoded_vectors from filename
    """
    with open(filename, 'rb') as f:
        print("Image Encoded Vectors loaded from directory path:", filename)
        return pickle.load(f)


class ImageEncoder:

    def __init__(self, encoder, images_dir_path, weights='imagenet'):
        """
        :param encoder: encoder name you want to use (ex. densenet201 for DenseNet201)
        :param weights: the pretrained weights you want to use for your model. It's common to use imagenet as default pretrained weights.
        """

        self.encoder_weights = weights
        self.image_dir_path = images_dir_path

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
                  "Choices are: [densenet201, densenet121, densenet169, efficientnet5, efficientnet7, resnet50v2, cotnet]")
            print("Exiting...")
            return

    def get_preprocessor(self):
        return self.preprocess

    def get_image_shape(self):
        return self.image_shape

    def get_image_encoder(self):
        return self.image_encoder

    def get_images_dirpath(self):
        return self.image_dir_path

    def encode(self, _image, verbose=0):
        if self.get_preprocessor() == 'cotnet':
            image = img.load_img(self.image_dir_path + _image + '.jpg')
            image_array = img.img_to_array(image)

            imm = tensorflow.keras.applications.imagenet_utils.preprocess_input(
                image_array, mode='torch')
            image_encoded = self.image_encoder(
                tensorflow.expand_dims(tensorflow.image.resize(imm, self.image_encoder.input_shape[1:3]), 0)).numpy()
        else:
            image = img.load_img(self.image_dir_path + _image + '.jpg',
                                 target_size=(self.image_shape, self.image_shape))
            image_array = img.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            if self.get_preprocessor() == 'densenet':
                preprocessed_image_array = dense_preprocess(image_array)
            elif self.get_preprocessor() == 'efficientnet':
                preprocessed_image_array = efficient_preprocess(image_array)
            elif self.get_preprocessor() == 'resnet':
                preprocessed_image_array = resnet_preprocess(image_array)
            elif self.get_preprocessor() == 'inceptionresnet':
                preprocessed_image_array = inceptionresnet_preprocess(
                    image_array)

            image_encoded = self.image_encoder.predict(
                preprocessed_image_array, verbose=verbose)
        return image_encoded

    def encode_images(self, images):

        image_vecs = {_image: self.encode(_image) for _image in
                      tqdm(images, desc="Encoding images", position=0, leave=True)}
        return image_vecs
