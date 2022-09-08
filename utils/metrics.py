import pandas as pd
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
import pickle

import csv
import string
import nltk
import warnings
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from rouge_score import rouge_scorer
import numpy as np
import spacy
import re

import errno
import os

try:
    os.mkdir('results')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# IMAGECLEF 2022 CAPTION - CAPTION PREDICTION
class AIcrowdEvaluator:

    remove_stopwords = True
    stemming = False
    lemmatization = True
    case_sensitive = False

    def __init__(self, ground_truth_path, **kwargs):
        """
        This is the AIcrowd evaluator class which will be used for the evaluation.
        Please note that the class name should be `AIcrowdEvaluator`
        `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
        """
        self.ground_truth_path = ground_truth_path
        self.gt = self.load_gt()

    def _evaluate(self, client_payload, _context={}):
        """
        This is the only method that will be called by the framework
        returns a _result_object that can contain up to 2 different scores
        `client_payload["submission_file_path"]` will hold the path of the submission file
        """
        # print("evaluate...")
        # Load submission file path
        submission_file_path = client_payload["submission_file_path"]
        # Load preditctions and validate format
        predictions = self.load_predictions(submission_file_path)

        bleu_score = self.compute_primary_score(predictions)
        score_secondary = self.compute_secondary_score(predictions)

        _result_object = {
            "score": bleu_score,
            "score_secondary": score_secondary
        }

        assert "score" in _result_object
        assert "score_secondary" in _result_object

        return _result_object

    def load_gt(self):
        """
        Load and return groundtruth data
        """
        # print("loading ground truth...")

        pairs = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
            for row in reader:
                pairs[row[0]] = row[1]
        return pairs

    def load_predictions(self, submission_file_path):
        """
        Load and return a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
        Validation of the runfile format has to be handled here. simply throw an Exception if there is a validation error.
        """
        # print("load predictions...")

        pairs = {}
        image_ids_gt = set(self.gt.keys())
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_images = []
            for row in reader:
                lineCnt += 1

                # less than two pipe separated tokens on line => Error
                if(len(row) < 2):
                    self.raise_exception("Wrong format: Each line must consist of an image ID followed by a '|' (vertical line) and a caption ({}).",
                                         lineCnt, "<imageID><vertical line><caption>")

                image_id = row[0]

                # Image ID does not exist in testset => Error
                if image_id not in image_ids_gt:
                    self.raise_exception(
                        "Image ID '{}' in submission file does not exist in testset.", lineCnt, image_id)
                    # raise Exception("Image ID '{}' in submission file does not exist in testset {}"
                    #     .format(image_id,self.line_nbr_string(lineCnt)))

                # image id occured at least twice in file => Error
                if image_id in occured_images:
                    self.raise_exception(
                        "Image ID '{}' was specified more than once in submission file.", lineCnt, image_id)
                    # raise Exception("Image ID '{}' was specified more than once in submission file {}"
                    #     .format(image_id, self.line_nbr_string(lineCnt)))

                occured_images.append(image_id)

                pairs[row[0]] = row[1]

            # In case not all images from the testset are contained in the file => Error
            if(len(occured_images) != len(image_ids_gt)):
                self.raise_exception(
                    "Number of image IDs in submission file not equal to number of image IDs in testset.", lineCnt)

        return pairs

    def raise_exception(self, message, record_count, *args):
        raise Exception(message.format(
            *args)+" Error occured at record line {}.".format(record_count))

    def compute_primary_score(self, predictions):
        """
        Compute and return the primary score
        `predictions` : valid predictions in correct format
        NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
        Valiation should be handled in the load_predictions method
        """
        # print("compute primary score...")

        return self.compute_bleu(predictions)

    def compute_bleu(self, candidate_pairs):
        # Hide warnings
        warnings.filterwarnings('ignore')

        # NLTK
        # Download Punkt tokenizer (for word_tokenize method)
        # Download stopwords (for stopword removal)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # English Stopwords
        stops = set(stopwords.words("english"))

        # Stemming
        stemmer = SnowballStemmer("english")

        # Loading Spacy model
        nlp = spacy.load("en_core_web_lg")

        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        # Define max score and current score
        max_score = len(self.gt)
        current_score = 0

        i = 0
        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Optional - Go to lowercase
            if not type(self).case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # Split caption into individual words (remove punctuation)
            candidate_words = nltk.tokenize.word_tokenize(
                candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(
                gt_caption.translate(translator))

            # Optional - Remove stopwords
            if type(self).remove_stopwords:
                candidate_words = [
                    word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower()
                            not in stops]

            # Optional - Apply lemmatization
            if not type(self).lemmatization:
                candidate_doc = nlp(" ".join(candidate_words))
                candidate_words = [token.lemma_ for token in candidate_doc]

                gt_doc = nlp(" ".join(gt_words))
                gt_words = [token.lemma_ for token in gt_doc]

            # Optional - Apply stemming
            if type(self).stemming:
                candidate_words = [stemmer.stem(word)
                                   for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]

            # Calculate BLEU score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_words) == 0 and len(candidate_words) == 0:
                    bleu_score = 1
                # Calculate the BLEU score
                else:
                    bleu_score = nltk.translate.bleu_score.sentence_bleu(
                        [gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
            # Handle problematic cases where BLEU score calculation is impossible
            except ZeroDivisionError:
                pass
                #raise Exception('Problem with {} {}', gt_words, candidate_words)

            # Increase calculated score
            current_score += bleu_score

        return current_score / max_score

    def compute_secondary_score(self, predictions):
        """
        Compute and return the secondary score
        Ignore or remove this method if you do not have a secondary score to provide
        `predictions` : valid predictions in correct format
        NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
        Valiation should be handled in the load_predictions method
        """
        # print("compute secondary score...")

        return self.compute_rouge(predictions)

    def compute_rouge(self, candidate_pairs):
        # Hide warnings
        warnings.filterwarnings('ignore')

        # ROUGE scorer (ROUGE-1 (Unigram) scoring)
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)

        # NLTK
        # Download Punkt tokenizer (for word_tokenize method)
        # Download stopwords (for stopword removal)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # English Stopwords
        stops = set(stopwords.words("english"))

        # Stemming
        stemmer = SnowballStemmer("english")

        # Loading Spacy model
        nlp = spacy.load("en_core_web_lg")

        # Remove punctuation from string
        translator = str.maketrans('', '', string.punctuation)

        scores = []

        i = 0
        for image_key in candidate_pairs:

            # Get candidate and GT caption
            candidate_caption = candidate_pairs[image_key]
            gt_caption = self.gt[image_key]

            # Optional - Go to lowercase
            if not type(self).case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()

            # Split caption into individual words (remove punctuation)
            candidate_words = nltk.tokenize.word_tokenize(
                candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(
                gt_caption.translate(translator))

            # Optional - Remove stopwords
            if type(self).remove_stopwords:
                candidate_words = [
                    word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower()
                            not in stops]

            # Optional - Apply stemming
            if type(self).stemming:
                candidate_words = [stemmer.stem(word)
                                   for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]

            # Optional - Apply lemmatization
            if not type(self).lemmatization:
                candidate_doc = nlp(" ".join(candidate_words))
                candidate_words = [token.lemma_ for token in candidate_doc]

                gt_doc = nlp(" ".join(gt_words))
                gt_words = [token.lemma_ for token in gt_doc]

            gt_caption = " ".join(gt_words)
            candidate_caption = " ".join(candidate_words)

            # Calculate ROUGE score for the current caption
            try:
                # If both the GT and candidate are empty, assign a score of 1 for this caption
                if len(gt_caption) == 0 and len(candidate_caption) == 0:
                    rouge_score_f = 1
                # Calculate the ROUGE score
                else:
                    rouge_score_f = scorer.score(gt_caption, candidate_caption)[
                        "rouge1"].fmeasure
            # Handle problematic cases where ROUGE score calculation is impossible
            except ZeroDivisionError:
                pass
                #raise Exception('Problem with {} {}', gt_caption, candidate_caption)

            # Append the score to the list of scores
            scores.append(rouge_score_f)

        # Calculate the average of all scores
        return np.mean(scores)

        # PUT AUXILIARY METHODS BELOW
        # ...
        # ...
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

    # compute ImageCLEF metrics (NLTK-BLEU as well as ROUGE-1)
    _client_payload = {}
    _client_payload["submission_file_path"] = res

    # Instaiate a dummy context
    _context = {}

    # Instantiate an evaluator
    aicrowd_evaluator = AIcrowdEvaluator(gts)

    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    
    metrics_scores['BLEU'] = [round(100*result['score'],1)]
    metrics_scores['ROUGE_1'] = [round(100*result['score_secondary'],1)]
    if save_scores:
        scores_df = pd.DataFrame.from_dict(metrics_scores)
        scores_df.to_csv('results/scores.csv', sep='\t')

    return metrics_scores