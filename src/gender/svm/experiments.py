"""
- Run multiple word n-gram experiments for different values of n & max_features (vocab size)
- 5-fold cross validated hyperparamter tuning for SVM
- Report the best f1-score along with precision, recall, AUROC, AUPRC, Weighted f1, and accuracy.
"""
import vectorize
import sys
sys.path.append('/path/Augmentation-for-Literary-Data/src/gender/')
import gender_data_loader

import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score

def hyperparameter_tuning(algo, X, Y, ngram_range, max_features, vectorize_fn):
    """
    Runs SVM with hyperparameter tuning (5-fold CV) for the given ngram_range and max_features - optimises for the best f1.
    Note that the method to vectorize the text is also passed as input in 'vectorize_fn'.

    Parameters
    ----------
    algo: the ML algorithm
    X: list of strings where each element is one training instance
    Y: corresponding list of labels
    ngram_range: tuple for the value of n (for TFIDF)
    max_features: vocabulary size (for TFIDF)
    vectorize_fn: Callable method to vectorize X (usually from vectorize.py)

    Returns
    -------
    best f1 and corresponding AUROC, weighted f1, precision, recall, accuracy, AUPRC
    """
    best_f1 = -1 # can change it to 0.0

    for param_dict in param_object:
        print("\n\nRunning for parameters:", param_dict)
        # Set the desired hyperparameters:
        algo.set_params(**param_dict)
        split_no = 1
        f1s = []; AUROCs = []; weighted_f1s = []; precision_s = []; recall_s = []; accuracies = []; AUPRCs = []

        for train_indices, test_indices in skf.split(X=np.zeros(len(Y)), y=Y): # only really need Y for splitting
            X_train, X_test = vectorize_fn(train_sentences=X[train_indices],
                                           test_sentences=X[test_indices],
                                           ngram_range=ngram_range,
                                           max_features=max_features)

            y_train = Y[train_indices]
            y_test = Y[test_indices]

            print("Split number: {} | Train: {} & {} | Test: {} & {}".format(split_no, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
            split_no += 1

            # Scale the data: (Performance was worse for batch 1)
#             scaler = StandardScaler(with_mean=False)
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)

            clf = algo.fit(X_train, y_train)
            preds = clf.predict(X_test)
            preds_with_probs = clf.predict_proba(X_test) # for AUROC & AUPRC

            print("Ordering:", clf.classes_)
            assert clf.classes_.tolist()[0] == 'fic' # make sure that the class ordering is ['fic' 'non']

            # Compute classification metrics:
            f1 = f1_score(y_test, preds, pos_label="fic")
            w_f1 = f1_score(y_test, preds, average='weighted')
            precision = precision_score(y_test, preds, pos_label="fic")
            recall = recall_score(y_test, preds, pos_label="fic")
            acc = accuracy_score(y_test, preds)
            auroc = roc_auc_score(y_test, preds_with_probs[:,1]) # need to pass probabilities for "greater label"
            auprc = average_precision_score(y_test, preds_with_probs[:,0], pos_label="fic") # need to pass probabilities for positive class ("fic")

            f1s.append(f1); AUROCs.append(auroc); weighted_f1s.append(w_f1); precision_s.append(precision); recall_s.append(recall); accuracies.append(acc); AUPRCs.append(auprc)

        # Compute mean:
        f1s = np.array(f1s); AUROCs = np.array(AUROCs); weighted_f1s = np.array(weighted_f1s); precision_s = np.array(precision_s); recall_s = np.array(recall_s); accuracies = np.array(accuracies); AUPRCs = np.array(AUPRCs)
        mean_f1 = f1s.mean(); mean_auroc = AUROCs.mean(); mean_weighted_f1 = weighted_f1s.mean(); mean_precision = precision_s.mean(); mean_recall = recall_s.mean(); mean_accuracy = accuracies.mean(); mean_auprc = AUPRCs.mean()

        if mean_f1 > best_f1: # Keep track of best f1 and corresponding metrics
            best_f1 = mean_f1
            corresponding_auroc = mean_auroc
            corresponding_weighted_f1 = mean_weighted_f1
            corresponding_precision = mean_precision
            corresponding_recall = mean_recall
            corresponding_accuracy = mean_accuracy
            corresponding_auprc = mean_auprc
            corresponding_params = param_dict

    return round(best_f1, 4), round(corresponding_auroc, 4), round(corresponding_weighted_f1, 4), round(corresponding_precision, 4), round(corresponding_recall, 4), round(corresponding_accuracy, 4), round(corresponding_auprc, 4), corresponding_params


def run_experiments(X, Y):
    """
    Runs experiments for different values of n-grams and max_features
    """
    for ngram_range in NGRAMS:
        for max_features in MAX_FEATURES:
            print("\n----------\nRunning word ngram-range = {} | max_features = {}".format(ngram_range, max_features))
            results_file.write(str(ngram_range)+' | '+str(max_features)+'\t')

            f1, auc, weighted_f1, prec, rec, accuracy, auprc, params = hyperparameter_tuning(algo=algo, X=X, Y=Y,
                                                                                             ngram_range=ngram_range,
                                                                                             max_features=max_features,
                                                                                             vectorize_fn=vectorize.ngrams_vectorize)
            print("F1:", f1)
            results_file.write(str(f1)+'\t'+str(auc)+'\t'+str(weighted_f1)+'\t'+str(prec)+'\t'+str(rec)+'\t'+str(accuracy)+'\t'+str(auprc)+'\t'+str(params)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--male', help='Percent of Male-fiction in Train Set (0 to 100)', required=True)
    args = parser.parse_args()

    print("Running SVM for male: {}% and female: {}%".format(args.male, 100-int(args.male)))
    algo = SVC(probability=True)
    tuned_parameters = [{'C': [0.01, 1, 1000], 'kernel': ['linear']},
                        {'C': [0.01, 1, 1000], 'gamma': ['auto'], 'kernel': ['rbf']}] # 'gamma': ['auto', 0.001, 0.0001]

    param_object = ParameterGrid(tuned_parameters)
    NUMBER_OF_FOLDS = 5
    skf = StratifiedKFold(n_splits=NUMBER_OF_FOLDS) # Splits the data into stratified folds
    NGRAMS = [(1,1), (2,2), (3,3), (1,3)] # Unigrams, Bigrams, Trigrams, UniBiTri_combined
    MAX_FEATURES = [None, 1000, 100000]

    X, Y, IDs = gender_data_loader.load_train_data(male_pct=int(args.male)/100, return_ids=True)
    t = [i[:5] for i in IDs]
    print("X: {} | Y: {} | Y-Dist: {} | IDs: {} | IDs-dist: {}".format(len(X), len(Y), Counter(Y), len(IDs), Counter(t)))

    results_path = '/path/Augmentation-for-Literary-Data/gender-results/hyperparm/svm_'+str(NUMBER_OF_FOLDS)+'foldCV_Male_'+str(args.male)+'.tsv' # name of output file
    print("\n-------\nResults path:", results_path, "\n\n")
    results_file = open(results_path, "w")
    results_file.write("Model\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\tParameters\n")
    run_experiments(X, Y)
