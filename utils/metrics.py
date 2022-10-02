import pandas as pd
import pandas as pd

from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor import Meteor
from .pycocoevalcap.rouge import Rouge
import re

import errno
import os

try:
    os.mkdir('results')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

def preprocess_captions(images_captions):
    """
    :param images_captions: Dictionary with image ids as keys and captions as values
    :return: Dictionary with the processed captions as values
    """

    # Clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                              '').strip().lower())
    pr_captions = {}
    # Apply bio clean to data
    for image in images_captions:
        # Save caption to an array to match MSCOCO format
        pr_captions[image] = [bioclean(images_captions[image])]

    return pr_captions

def compute_scores(gts:str, res:str, save_scores:bool=True):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    # convert pd.Dataframe to dict
    gold_captions_df = pd.read_csv(gts, sep='|', names=['ID', 'caption'])
    pred_captions_df = pd.read_csv(res, sep='|', names=['ID', 'caption'])
    
    gold_captions = preprocess_captions(dict( zip( gold_captions_df.ID.to_list(), gold_captions_df.caption.to_list() ) ) )
    pred_captions = preprocess_captions( dict( zip( pred_captions_df.ID.to_list(), pred_captions_df.caption.to_list() ) ) )
    

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    metrics_scores = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gold_captions, pred_captions, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gold_captions, pred_captions)
        if type(method) == list:
            for sc, m in zip(score, method):
                metrics_scores[m] = [round(sc*100, 1)]
        else:
            metrics_scores[method] = [round(score*100, 1)]

    if save_scores:
        scores_df = pd.DataFrame.from_dict(metrics_scores)
        scores_df.to_csv('results/scores.csv', sep='\t')

    return metrics_scores