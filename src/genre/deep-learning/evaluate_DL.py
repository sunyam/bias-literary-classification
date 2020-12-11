"""
This file contains functions to evaluate the model: compute multiple classification metrics and return model's predictions.
"""
import numpy as np
import torch
from typing import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score

from allennlp.data.iterators import DataIterator
from allennlp.data.iterators import BasicIterator
from allennlp.data import Instance
from allennlp.models import Model
from allennlp.nn import util as nn_util


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator, cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return out_dict["class_probabilities"]

    def predict(self, dataset: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(dataset, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(dataset))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))

        return np.concatenate(preds, axis=0)

def make_predictions(model, vocab, test_dataset, batch_size, use_gpu=False):
    """
    Runs the given 'model' on the given 'test_dataset' to get predictions.
    Returns the predictions
    """
    assert vocab.get_token_from_index(index=0, namespace='labels') == 'fic'
    assert vocab.get_token_from_index(index=1, namespace='labels') == 'non'
    print("\n\nThe ordering of labels is ['fic', 'non']\n\n")
    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, cuda_device=0 if use_gpu else -1)
    preds = predictor.predict(test_dataset)
    return preds


def prob_to_label(probs):
    """
    Converts the predicted probablities to the corresponding 'hard' label.
    Note: The order of labels is ['fic', 'non']
    """
    if probs[0] >= probs[1]:
        return "fic"
    else:
        return "non"


def map_id_prediction(preds, test_dataset):
    """
    For saving predictions: maps the ID to the corresponding predicted Probability('fic')

    Input is predicted probabilities.
    Returns a dictionary with key: ID | value: probability_fic
    """
    out = {}
    for prediction, sample in zip(preds, test_dataset):
        sample_id = sample.fields['ID'].metadata
        out[sample_id] = prediction[0] # because order is ['fic', 'non']
    return out


def compute_metrics(preds, test_dataset):
    """
    Computes the classification metrics given the predictions and true labels.
    """
    y_preds = [prob_to_label(list(probabilities)) for probabilities in preds] # predicted labels
    prob_positive = [list(prob)[0] for prob in preds] # probabilities of the positive class 'fic' (for AUPRC); order is ['fic', 'non']
    prob_greater = [list(prob)[1] for prob in preds] # probabilities of the greater class (for AUROC); order is ['fic', 'non']

    y_true = []
    for instance in test_dataset:
        y_true.append(instance.fields['label'].label)

    # Compute classification metrics:
    f1 = f1_score(y_true, y_preds, pos_label='fic')
    auroc = roc_auc_score(y_true, y_score=prob_greater)
    w_f1 = f1_score(y_true, y_preds, average='weighted')
    precision = precision_score(y_true, y_preds, pos_label='fic')
    recall = recall_score(y_true, y_preds, pos_label='fic')
    acc = accuracy_score(y_true, y_preds)
    auprc = average_precision_score(y_true, y_score=prob_positive, pos_label='fic')

    return round(f1,4), round(auroc,4), round(w_f1,4), round(precision,4), round(recall,4), round(acc,4), round(auprc,4)
