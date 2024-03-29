# BSc Thesis research in Diagnostic Captioning

## Thesis paper
[Exploring Uni-modal, Cross-modal, and Multi-modal Diagnostic Captioning](http://nlp.cs.aueb.gr/theses/g_zachariadis_bsc_thesis.pdf)

## Abstract
Recent years have witnessed an increase in studies associated with image captioning, but little of that knowledge has been utilised in the biomedical field. This thesis addresses medical image captioning, referred as Diagnostic Captioning (DC), the task of assisting medical experts in diagnosis/report drafting. We present deep learning uni-modal, cross-modal and multi-modal methods that aim to generate a representative ``diagnostic text'' for a given medical image. The multi-modal approaches, utilise the radiology concepts (tags) used by clinicians to describe a patient's image (e.g., X-Ray, CT scan, etc.) as an additional input data. These methods, have not been adequately applied to biomedical research. We also experimented with a novel technique that utilises the captions generated from all the systems implemented as part of this thesis. Lastly, this thesis concerns the participation of AUEB's NLP Group, with the author being the main driver, on the 2022 ImageCLEFmedical Caption Prediction task. Out of 10 teams, our team came in second based on the primary evaluation metric, using an encoder-decoder approach, and first based on the secondary metric, utilising an ensemble technique applied on our generated captions. More about our paper can be found [here](http://ceur-ws.org/Vol-3180/paper-101.pdf)


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
```
  1. conda create --name torch_gpu
  2. activate torch_gpu
  3. conda install torch-gpu
  4. pip install -r requirements.txt
```
Now, your environment will be compatible with Pytorch. Then comment-out the imports from `models/__init__.py` and `models/kNN.py`

## Dataset Instructions
As mentioned in the `Abstract` section, I participated in ImageCLEFmedical 2022 Caption Prediction task. The code also handles ImageCLEF dataset, but the latter as well as evaluation measures are not provided, due to the fact that we, as a group, signed an End User Agreement. Thus, only the IU X-Ray dataset is available. Go to [Datasets](https://github.com/zaaachos/Thesis-Diagnostic-Captioning/tree/main/data), download the dataset (i.e. IU X-Ray) and store it to the `data` directory

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

## Execution Instructions
### Disclaimer
Throughout my research on this Thesis, I experimented with models that had state-of-the-art performance (SOTA) on several biomedical datasets (like IU X-Ray, MIMIC III. etc.). These models are provided in `SOTA_models` directory as submodules repos. More details about each model are provided on my Thesis paper. I do not provide any additional data loaders, which I created for this models. Thus, if you want to further experiment with these models, please do so according to the guidelines provided in each of these repositories.

### Main applications
Follow the aforementioned steps to use conda and run the following command, to train my implemented methods (i.e. CNN-RNN, kNN). Default arguments are set.
```py
python3 dc.py
```

For arguments passing, run the following command in order to watch the available args.
```py
python3 dc.py -h
```

### Particular training procedures
It is suggested to use a Unix-like OS (like Linux) to execute the following specific processes or using WSL in Windows OS.
* Cross-modal CNN-RNN: `bash cross_modal_cnn_rnn.sh`
* Multi-modal CNN-RNN: `bash multi_modal_cnn_rnn.sh`
* Cross-modal k-NN: `bash cross_modal_kNN.sh`
* Multi-modal CNN-RNN: `bash multi_modal_kNN.sh`

## Citations
If you use or extend my work, please cite my paper.
```
@unpublished{Zachariadis2022,
  author = "G. Zachariadis",
  title = "Exploring Uni-modal, Cross-modal, and Multi-modal Diagnostic Captioning",
  year = "2022",
  note = "B.Sc. thesis, Department of Informatics, Athens University of Economics and Business}
}
```

You can read our publication ***"AUEB NLP Group at ImageCLEFmedical Caption 2022", Proceedings of the CLEF 2022*** at this [link](https://ceur-ws.org/Vol-3180/paper-101.pdf). If you use or extend our work, please cite our paper:
```
@article{charalampakos2022aueb,
  title={Aueb nlp group at imageclefmedical caption 2022},
  author={Charalampakos, Foivos and Zachariadis, Giorgos and Pavlopoulos, John and Karatzas, Vasilis and Trakas, Christoforos and Androutsopoulos, Ion},
  year={2022}
}
```

## License
[MIT License](https://github.com/zaaachos/bsc-thesis-in-diagnostic-captioning/blob/main/LICENSE)
