"""
This file runs (train/evaluate) several deep learning models on the UQ task.
"""
import data_reader
import train
import evaluate_DL

import torch
import os
import sys
import pickle
import numpy as np

from pathlib import Path
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer # for ELMo

def tokenizer(x: str):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

def run_model(name, case, augmentation=None, use_elmo=False, save_model=False):
    """
    Runs the given model 'name' for the given 'case' and with the given method of 'augmentation'.
    Optionally saves the trained model (with vocabulary) and predictions.
    Allowed names: lstm | bilstm | stacked_bilstm | cnn

    If use_elmo=True, uses ELMo's pre-trained language model for embeddings; else, use GloVe embeddings

    Returns classification metrics and predictions
    """
    if use_elmo:
        token_indexer = ELMoTokenCharactersIndexer() # token indexer is responsible for mapping tokens to integers: this makes sure that the mapping is consistent with what was used in the original ELMo training.
    else:
        token_indexer = SingleIdTokenIndexer()

    reader = data_reader.NovelDatasetReader(scenario=case, augmentation=augmentation, tokenizer=tokenizer, token_indexers={"tokens": token_indexer})
    train_dataset = reader.read(file_path='train')
    test_dataset = reader.read(file_path='test')
    print("Train: ", len(train_dataset), "| Test:", len(test_dataset))

    print("\n#####################################################\n")

    # Train model:
    if name == 'lstm':
        model, vocab, ep = train.train_lstm(train_dataset, BATCH_SIZE, epochs=15,
                                            num_layers=1, bidirectional=False, use_elmo=use_elmo)
    elif name == 'bilstm':
        model, vocab, ep = train.train_lstm(train_dataset, BATCH_SIZE, epochs=15,
                                            num_layers=1, bidirectional=True, use_elmo=use_elmo)
    elif name == 'stacked_bilstm':
        model, vocab, ep = train.train_lstm(train_dataset, BATCH_SIZE, epochs=15,
                                            num_layers=2, bidirectional=True, use_elmo=use_elmo)

    elif name == 'cnn':
        filter_sizes = (2,3,4) # kernels can not be bigger than the shortest sentence
        model, vocab, ep = train.train_cnn(train_dataset, BATCH_SIZE, epochs=15,
                                           num_filters=100, filter_sizes=filter_sizes, use_elmo=use_elmo)
    else:
        sys.exit("'name' not valid")


    # Predict and evaluate model on test set:
    preds = evaluate_DL.make_predictions(model, vocab, test_dataset, BATCH_SIZE, use_gpu=False) # NOTE: preds is of shape (number of samples, 2) - the columns represent the probabilities for the two classes in order ['yes_unp', 'not_unp']
    print("Predictions:", preds.shape)
    f1, auroc, w_f1, precision, recall, acc, auprc = evaluate_DL.compute_metrics(preds, test_dataset)

    # save individual predictions for other metrics:
    id_pred = evaluate_DL.map_id_prediction(preds, test_dataset)

    if save_model: # save the model weights and vocabulary
        with open('./'+name+'_model.th', 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files("./"+name+"_vocabulary")

    print("\nF1 = {} | AUROC = {} | AUPRC = {}".format(f1, auroc, auprc))
    print("Total predictions: {} | Number of Epochs: {}".format(len(id_pred), ep))

    return f1, auroc, w_f1, precision, recall, acc, auprc, id_pred

BATCH_SIZE = 32
torch.manual_seed(42)
