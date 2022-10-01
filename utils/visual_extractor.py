# os imports
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# other utils imports
from tqdm import tqdm
from PIL import Image
import h5py
import pandas as pd



class VisualExtractor(nn.Module):
    def __init__(self, model:str="resnet101", pretrained:bool=True):
        """ Class used to produce embeddings for Show, Attend and Tell model (SAnT).

        Args:
            model (str, optional): The CNN we want to employ. Defaults to "resnet101".
            pretrained (bool, optional): If we want to initialize the pretrained on ImageNet encoder. Defaults to True.
        """
        super(VisualExtractor, self).__init__()
        self.visual_extractor = model
        self.pretrained = pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images:torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """ Pass the image from the CNN and extract the image embeddings as well as the average feature vectors

        Args:
            images (torch.tensor): The images to encode

        Returns:
            tuple[torch.tensor, torch.tensor]: The image embeddings as well as the average feature vectors
        """
        images = images.cuda()
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


class FeatureExtractor:
    def __init__(self, image_dir_path:str, dataset:pd.DataFrame, detections_path:str):
        """ Class which utilise the VisualExtractor class to produce image feature representations.

        Args:
            image_dir_path (str): the image directory path where our images are stored
            dataset (pd.DataFrame): The dataframe with image_ids, caption pairs
            detections_path (str): the image directory path to store our vectors
        """
        self.image_dir_path = image_dir_path
        self.dataset = dataset
        self.detections_path = detections_path
        self.ve = VisualExtractor()
        # pass to GPU
        dev = torch.device("cuda:0")
        self.ve.to(dev)
        # augment our images
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare_image(self, split:str, image_id:str) -> torch.tensor:
        """ Preprocess our images into tensor format

        Args:
            split (str): The split from the dataset we used
            image_id (str): Current image id from the split set

        Returns:
            torch.tensor: The pre-processed image in torch.tensor format
        """
        image = Image.open(os.path.join(self.image_dir_path + split + image_id + ".jpg")).convert("RGB")
        image = self.transform(image)
        return torch.stack([image], 0)

    def make_features_detection_file(self) -> None:
        """ Produce the image embeddings that fit to SAnT model.
        """
        image_ids = self.dataset.ID.to_list()

        # create a h5py file to store our feature representations.
        with h5py.File(self.detections_path, "w") as hf:
            for im_id in tqdm(image_ids, position=0, leave=True):
                if "train" in im_id:
                    prepared_image = self.prepare_image("train/", im_id)
                else:
                    prepared_image = self.prepare_image("valid/", im_id)

                # exctract embeddings
                features, _ = self.ve.forward(prepared_image)

                hf.create_dataset(im_id, data=features[0].cpu().detach().numpy(), compression="gzip")
