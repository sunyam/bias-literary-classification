# Given the algorithm's predictions (TSV), this file computes several evaluation metrics (both classification & bias) and writes the results to a file.

import pickle
import argparse
import os
import scipy.stats
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def classification_metrics(df):
    """
        Computes classification metrics for the given predictions-DataFrame.
        """
    preds = df['prediction'].tolist() # list of predicted (binary) labels
    y_test = [] # list of truth (binary) labels
    for fname in df['fname'].tolist():
        y_test.append(fname[:3])
    
    assert Counter(y_test)['fic'] == Counter(y_test)['non'] == 100 # test set is balanced
    
    # Compute classification metrics:
    f1 = f1_score(y_test, preds, pos_label="fic")
    precision = precision_score(y_test, preds, pos_label="fic")
    recall = recall_score(y_test, preds, pos_label="fic")
    acc = accuracy_score(y_test, preds)
    
    return round(f1,4), round(precision,4), round(recall,4), round(acc,4)

def dialog_bias(df):
    """
    Computes (1) dialog-distribution of True Positives and (2) its relative entropy with ideal test-set distribution.
    Returns the normlalized TP-dialog distribution and relative entropy.
    """
    true_positive_dial = []; check = []
    for pred, fname in df[['prediction', 'fname']].values:
        binary_truth = fname.split('____')[0][:3]
        check.append(fname.split('____')[0])
        if pred == 'fic' and binary_truth == 'fic': # condition on true positives
            true_positive_dial.append(fname.split('____')[0])

    assert Counter(check) == Counter({'non': 100, 'ficWithDialog': 50, 'ficNoDialog': 50})

    dist = Counter(true_positive_dial)
    if len(dist) == 1 and 'ficWithDialog' in dist:
        dist['ficNoDialog'] = 0
    if len(dist) == 1 and 'ficNoDialog' in dist:
        dist['ficWithDialog'] = 0

    # Normalize:
    total = sum(dist.values())
    for key in dist:
        dist[key] /= total
        dist[key] = round(dist[key], 4)
    dist = dict(dist)

    # Compute a single measure to capture how balanced the difference-distribution is:
    ideal = {'ficWithDialog': 0.5, 'ficNoDialog': 0.5} # Test Set has: 50 "Dialog" + 50 "Non-Dialog"
    assert ideal.keys() == dist.keys()
    
    ent = scipy.stats.entropy(pk=list(dist.values()), qk=list(ideal.values())) # computes relative entropy (KL)
    
#    print("Ideal Distribution:", ideal)
    print("Distribution:", dist)
#    print("Relative Entropy:", ent)
    return dist, ent


def evaluate_classifier(path):
    """
    The main fuction to evaluate: 1) f1, precision, recall, accuracy & 2) Bias Measure

    Input: path to the predictions TSV (output of main.py)
    """
    df = pd.read_csv(path, delimiter='\t')
    assert not df.isnull().values.any() # make sure there are no NaN values

    # 1. Metrics:
    f1, precision, recall, acc = classification_metrics(df)

    # 2. Bias:
    dist, ent = dialog_bias(df)

    return f1, precision, recall, acc, dist, ent

if __name__ == '__main__':
    results_path = '/path/Augmentation-for-Literary-Data/dialog-results/dialog_bias_results_500words.tsv'
    results_file = open(results_path, "w")
    results_file.write("Algorithm\t% Train Set with Dialog\tF1-score\tPrecision\tRecall\tAccuracy\tTP-dialog\tRelative-Entropy\n")

    path = '/path/Augmentation-for-Literary-Data/dialog-results/predictions/'

    print("Read predictions from {}\nWrite results to: {}".format(path, results_path))

    for fname in os.listdir(path):
        if fname.startswith('.'):
            continue
        print("Process:", fname)
        f1, precision, recall, acc, dist, ent = evaluate_classifier(path+fname)

        algorithm = fname.split('_pred')[0]
        case = fname[:-4].split('_')[-1]
        results_file.write(algorithm+'\t'+case+'\t'+str(f1)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(acc)+'\t'+str(dist)+'\t'+str(ent)+'\n')

    print("Results file:", results_path)
