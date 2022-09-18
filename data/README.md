## IU X-Ray dataset
The employed data for IU X-Ray dataset can be found in this [drive link](https://drive.google.com/drive/folders/147hav9_PfmCrpJtJOKwsOn9e24j2lSRH?usp=sharing). Download the dataset (i.e. IU X-Ray) and store it to the `data` directory. Image Vectors were computed offline. Only DenseNet-121 and EfficientNetB0 are provided due to the fact that these image encoders performed better in preliminary experiments.

*You have to have something like this*:
```
.
├── data
│   ├── iu_xray
|   |   ├──two_captions.json
|   |   ├──two_images.json
|   |   ├──two_tags.json
|   |   └──densenet121.pkl     
|   |
|   ├──fasttext_voc.pkl
|   └──fasttext.npy
```

## Preprocess I followed
IU X-Ray contains 3,955 patients with the majority of them having two images (Frontal and Lateral view). Thus, we decided to keep the patients who have exactly two images for our research to keep the consistency with other SOTA models on the IU X-Ray dataset (e.g., R2GenCMN). As a result, following pre-processing, 3,195 patients (or 6,390 chest x-ray images) were retained for the dataset.

We followed the next splits:
| Info | Train | Val | Test |
| --- | --- | --- | --- |
| split size | 80% | 5% | 15% |
| # Images | 4,890 | 542 | 958 |
| # Reports | 2,445 | 271 | 479 |
| # Patients | 2,445 | 271 | 479 |
