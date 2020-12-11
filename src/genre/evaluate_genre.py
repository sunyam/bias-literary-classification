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
    preds = df['label'].tolist() # list of predicted (binary) labels
    y_test = [] # list of truth (binary) labels
    for genre in df['genre'].tolist():
        binary = map_binary[genre]
        y_test.append(binary)

    assert Counter(y_test)['fic'] == Counter(y_test)['non'] == 99 # test set is balanced

    # Compute classification metrics:
    f1 = f1_score(y_test, preds, pos_label="fic")
    precision = precision_score(y_test, preds, pos_label="fic")
    recall = recall_score(y_test, preds, pos_label="fic")
    acc = accuracy_score(y_test, preds)

    return round(f1,4), round(precision,4), round(recall,4), round(acc,4)


def genre_bias(df):
    """
    Computes (1) genre-distribution of True Positives and (2) its relative entropy with ideal test-set distribution.
    Returns the normlalized TP-genre distribution and relative entropy.
    """
    true_positive_genres = [] # genres for True Positives
    for pred, genre in df[['label', 'genre']].values:
        binary_truth = map_binary[genre]
        if pred == 'fic' and binary_truth == 'fic': # condition on true positives
            true_positive_genres.append(genre)

    dist = Counter(true_positive_genres)

    # Normalize:
    total = sum(dist.values())
    for key in dist:
        dist[key] /= total
        dist[key] = round(dist[key], 4)
    dist = dict(dist)

    ideal = {'rom': 0.33333, 'mys': 0.33333, 'sci': 0.33333} # Test Set has: 33 "Mys" + 33 "Rom" + 33 "SciFi"
    assert ideal.keys() == dist.keys()

    ent = scipy.stats.entropy(pk=list(dist.values()), qk=list(ideal.values())) # computes relative entropy (KL)

    return dist, ent


def evaluate_classifier(path):
    """
    The main fuction to evaluate: 1) f1, precision, recall, accuracy & 2) Bias Measure

    Input: path to the predictions TSV (output of main.py)
    """
    df = pd.read_csv(path, delimiter='\t')
    df['genre'] = df['fname'].map(GROUND_TRUTH)
    assert not df.isnull().values.any() # make sure there are no NaN values

    # 1. Metrics:
    f1, precision, recall, acc = classification_metrics(df)

    # 2. Bias:
    dist, ent = genre_bias(df)

    return f1, precision, recall, acc, dist, ent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiment Number', required=True)
    parser.add_argument('--doclen', help='Document Length', required=True)
    args = parser.parse_args()

    experiment = args.exp
    DOCUMENT_LENGTH = args.doclen

    results_path = '/path/Augmentation-for-Literary-Data/genre-results/'+DOCUMENT_LENGTH+'-word-results/genre_bias_results_exp'+experiment+'.tsv'
    results_file = open(results_path, "w")
    results_file.write("Algorithm\tCase\tF1-score\tPrecision\tRecall\tAccuracy\tTP-Genre\tRelative-Entropy\n")

    map_binary = {'non': 'non', 'mys': 'fic', 'rom': 'fic', 'sci': 'fic'} # maps genre to the corresponding binary label
    with open('/path/Augmentation-for-Literary-Data/genre-results/ground_truth.pickle', 'rb') as f:
        GROUND_TRUTH = pickle.load(f)

    path = '/path/Augmentation-for-Literary-Data/genre-results/'+DOCUMENT_LENGTH+'-word-results/predictions-experiment-'+experiment+'/'

    print("Running for document length: {} | Experiment number: {}".format(DOCUMENT_LENGTH, experiment))
    print("Read predictions from {}\nWrite results to: {}".format(path, results_path))

    for fname in os.listdir(path):
        if fname.startswith('.'):
            continue
        print("Process:", fname)
        f1, precision, recall, acc, dist, ent = evaluate_classifier(path+fname)

        algorithm = fname.split('_pred')[0]
        case = fname.split('_')[-1][0]
        results_file.write(algorithm+'\t'+case+'\t'+str(f1)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(acc)+'\t'+str(dist)+'\t'+str(ent)+'\n')

    print("Results file:", results_path)
