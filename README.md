# BSc Thesis research in Diagnostic Captioning

## Thesis paper
To be reviewed and to be added..

## Abstract
Recent years have witnessed an increase in studies associated with image captioning, but little of that knowledge has been utilised in the biomedical field. This repo (as well as this thesis) addresses medical image captioning, referred as Diagnostic Captioning (DC), the task of assisting medical experts in diagnosis/report drafting. We present deep learning uni-modal, cross-modal and multi-modal methods that aim to generate a representative caption for a given medical image. The latter approaches, utilise the radiology concepts (tags) used by clinicians to describe a patient's image (e.g., X-Ray, CT scan, etc.) as an additional input data. These methods, have not been adequately applied to biomedical research. We also experimented with a novel technique that utilises the captions generated from all the systems implemented as part of this thesis. Lastly, this thesis concerns the participation of the AUEB NLP Group, with the author being the main driver, on the 2022 ImageCLEFmedical Caption Prediction task. Out of 10 teams, our team came in second on the primary evaluation metric, using an encoder-decoder approach, and first on the secondary metric, utilising an ensemble technique on our generated caption. More about our paper can be found [here](http://ceur-ws.org/Vol-3180/paper-101.pdf)

## Datasets
As mentioned in the `Abstract` section, I participated in ImageCLEFmedical 2022 Caption Prediction task. The code also handles ImageCLEF dataset but it is not provided, due to the fact that we, as a group, signed an End User Agreement. Thus, only the IU X-Ray dataset is available and can be downloaded by redirecting [here](https://github.com/zaaachos/Thesis-Diagnostic-Captioning/tree/main/data) in `data` directory.

## Enviroment setup
If you have GPU installed on your system, it is highly suggested to use conda as your virtual enviroment to run code. You can download conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

After the installation is completed, open a terminal inside this project and run the following commands, to setup conda enviroment. The latter will be compatible with Tensorflow.
```
  1. conda create --name tf_gpu
  2. activate tf_gpu
  3. conda install tensorflow-gpu
  4. pip install -r requirements.txt
```

If you decide to use **ClinicalBERT** as the main text embeddings extraction model, you have to execute the `dc.py` in Pytorch-based enviroment. Thus, follow the next steps:
After the installation is completed, open a terminal inside this project and run the following commands, to setup conda enviroment. The latter will be compatible with Tensorflow.
```
  1. conda create --name torch_gpu
  2. activate torch_gpu
  3. conda install torch-gpu
  4. pip install -r requirements.txt
```
Then comment-out the imports from `models/__init__.py` and `models/kNN.py`

## Instructions
Go to [Datasets](https://github.com/zaaachos/Thesis-Diagnostic-Captioning/tree/main/data), download the dataset (i.e. IU X-Ray) and store it to the `data` directory

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

Follow the aforementioned steps to use conda and run the following command, to train my implemented methods (i.e. CNN-RNN, kNN)
```py
python3 dc.py
```

For arguments passing, run the following command in order to watch the available args.
```py
python3 dc.py -h
```

For **SOTA_models** please follow the instructions given by each author in their repo.

## License
[MIT License](https://github.com/zaaachos/bsc-thesis-in-diagnostic-captioning/blob/main/LICENSE)
