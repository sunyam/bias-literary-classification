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
        if fname.startswith('non'):
            y_test.append(fname[:3])
        else:
            y_test.append(fname[2:5])
    
    print(Counter(y_test))
    assert Counter(y_test)['fic'] == Counter(y_test)['non'] == 100 # test set is balanced
    
    # Compute classification metrics:
    f1 = f1_score(y_test, preds, pos_label="fic")
    precision = precision_score(y_test, preds, pos_label="fic")
    recall = recall_score(y_test, preds, pos_label="fic")
    acc = accuracy_score(y_test, preds)
    
    return round(f1,4), round(precision,4), round(recall,4), round(acc,4)

def gender_bias(df):
    """
    Computes (1) gender-distribution of True Positives and (2) its relative entropy with ideal test-set distribution.
    Returns the normlalized TP-gender distribution and relative entropy.
    """
    true_positive_gender = []; check = []
    for pred, fname in df[['prediction', 'fname']].values:
        if fname.startswith('non'):
            binary_truth = fname[:3]
        else:
            binary_truth = fname[2:5]
        
        check.append(fname[:5])
        if pred == 'fic' and binary_truth == 'fic': # condition on true positives
            true_positive_gender.append(fname[:5])

    # Sanity check:
    assert Counter(check) == Counter({'non__': 100, 'M_fic': 50, 'F_fic': 50})
    
    dist = Counter(true_positive_gender)
    if len(dist) == 1 and 'F_fic' in dist:
        dist['M_fic'] = 0
    if len(dist) == 1 and 'M_fic' in dist:
        dist['F_fic'] = 0

    # Normalize:
    total = sum(dist.values())
    for key in dist:
        dist[key] /= total
        dist[key] = round(dist[key], 4)
    dist = dict(dist)
    
    # Compute a single measure to capture how balanced the difference-distribution is:
    ideal = {'F_fic': 0.5, 'M_fic': 0.5} # Test Set has: 50 "Male" + 50 "Female"
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
    dist, ent = gender_bias(df)
    return f1, precision, recall, acc, dist, ent

if __name__ == '__main__':
    results_path = '/path/Augmentation-for-Literary-Data/gender-results/gender_bias_results_500words.tsv'
    results_file = open(results_path, "w")
    results_file.write("Algorithm\t% Train Set by Male Author\tF1-score\tPrecision\tRecall\tAccuracy\tTP-gender\tRelative-Entropy\n")

    path = '/path/Augmentation-for-Literary-Data/gender-results/predictions/'

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
