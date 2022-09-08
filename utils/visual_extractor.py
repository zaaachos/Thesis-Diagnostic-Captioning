import torch
import torch.nn as nn
import torchvision.models as models
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import h5py
import pandas as pd
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class VisualExtractor(nn.Module):
    def __init__(self, model="resnet101", pretrained=True):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = model
        self.pretrained = pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        images = images.cuda()
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


class FeatureExtractor:
    def __init__(self, image_dir_path, dataset, detections_path):
        # super(FeatureExtractor, self).__init__()
        self.image_dir_path = image_dir_path
        self.dataset = dataset
        self.detections_path = detections_path
        self.ve = VisualExtractor()
        dev = torch.device("cuda:0")
        self.ve.to(dev)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare_image(self, split, image_id):
        image = Image.open(
            os.path.join(self.image_dir_path + split + image_id + ".jpg")
        ).convert("RGB")
        image = self.transform(image)
        return torch.stack([image], 0)

    def make_features_detection_file(self):
        image_ids = self.dataset.ID.to_list()

        with h5py.File(self.detections_path, "w") as hf:
            for im_id in tqdm(image_ids, position=0, leave=True):
                if "train" in im_id:
                    prepared_image = self.prepare_image("train/", im_id)
                else:
                    prepared_image = self.prepare_image("valid/", im_id)

                features, _ = self.ve.forward(prepared_image)

                hf.create_dataset(
                    im_id, data=features[0].cpu().detach().numpy(), compression="gzip"
                )


if __name__ == "__main__":
    detections_path = "../dataset/imageclef2022_features_compressed.hdf5"
    image_dir_path = "../dataset/imageclef/"
    train_df = pd.read_csv(
        "../dataset/imageclef/ImageCLEFmedCaption_2022_caption_prediction_train.csv",
        sep="\t",
    )
    valid_df = pd.read_csv(
        "../dataset/imageclef/ImageCLEFmedCaption_2022_caption_prediction_valid.csv",
        sep="\t",
    )

    dataset = pd.concat([train_df, valid_df], join="inner").reset_index(drop=True)

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    fd = FeatureExtractor(
        image_dir_path=image_dir_path, dataset=dataset, detections_path=detections_path
    )
    fd.make_features_detection_file()
