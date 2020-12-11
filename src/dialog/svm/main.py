# While experiments.py runs hyperparameter tuning with cross-validation on the training set, this main.py is for predicting on the test data. It uses the best hyperparameters (f1-score).

import vectorize
import sys
sys.path.append('/path/Augmentation-for-Literary-Data/src/dialog/')
import dialog_data_loader
import ast
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import SVC

def predict(algo, train_passages, Y_train, test_passages, ngrams, max_features):
    """
    Train SVM on all the fiction and non-fiction volumes.
    Predicts on the test set.

    Returns
    -------
    List of probabilities for fiction & non-fiction
    """
    X_train, X_test = vectorize.ngrams_vectorize(train_sentences=train_passages,
                                                 test_sentences=test_passages,
                                                 ngram_range=ngrams,
                                                 max_features=max_features)

    clf = algo.fit(X_train, Y_train)
    assert clf.classes_.tolist()[0] == 'fic' # make sure that the class ordering is ['fic' 'non']
    preds_with_probs = clf.predict_proba(X_test)

    print("Train: {} & {} | Test: {}".format(X_train.shape, Y_train.shape, X_test.shape))
    print("Ordering:", clf.classes_)
    print("Predicitions shape:", preds_with_probs.shape)

    return preds_with_probs

def get_best_parameters(dial):
    """
    Returns the best hyperparamters for SVM given the dialog proportion in training set (corresponding to best f1)
    """
    results_path = '/path/Augmentation-for-Literary-Data/dialog-results/hyperparm/svm_5foldCV_dial_'+str(dial)+'.tsv'
#    print("Reading parameters from:", results_path)
    df = pd.read_csv(results_path, delimiter='\t')
    best = df.loc[df['F1-score']==df['F1-score'].max()]
#    print("Best:", best)

    model = best['Model'].values[0]
    try:
        MAX_FEATURES = int(model.split('|')[1])
    except:
        if model.split('|')[1].strip() == 'None':
            MAX_FEATURES = None

    temp = model.split('|')[0].strip().split(',')
    NGRAMS = (int(temp[0][1]), int(temp[1][1]))

    param_dict = ast.literal_eval(best['Parameters'].values[0])
    return MAX_FEATURES, NGRAMS, param_dict

if __name__ == '__main__':

    # Load test data (same for each scenario):
    test_passages, Y_test, pct_quoted_fic_test, test_IDs = dialog_data_loader.load_test_data()
    t = [i.split('____')[0] for i in test_IDs]
    print("Test Set ---- X: {} | Y: {} | Distribution: {} | Dialog dist in test: {} | Test IDs: {}, preview: {}".format(len(test_passages), len(Y_test), Counter(Y_test), Counter(t), len(test_IDs), test_IDs[:3]))
    assert len(test_passages) == len(Y_test) == 200
    with open('/path/Augmentation-for-Literary-Data/dialog-results/pct_dialog/test_data_dialog.pickle', 'wb') as f:
        pickle.dump(pct_quoted_fic_test, f)

    for dial in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]: # run the experiments (increasing dialog percent in training set)
        nondial = 100 - dial
        print("\n\n\n\n\n\n\n-------------------------\nRunning for dialog: {}% | non-dialog: {}%".format(dial, nondial))
        
        MAX_FEATURES, NGRAMS, param_dict = get_best_parameters(dial) # best parameters for the given scenario
        algo = SVC(probability=True)
        algo.set_params(**param_dict) # set hyperparameters
        print("Max-features = {} | N-grams = {}| Parameters: {}".format(MAX_FEATURES, NGRAMS, param_dict))
    
        train_passages, Y_train, pct_quoted_fic, train_IDs = dialog_data_loader.load_train_data(dial=dial/100, no_dial=nondial/100)
        t = [i.split('____')[0] for i in train_IDs]
        print("X_train: {} | Y_train: {} | Y Distribution: {} | Dialog Dist: {}".format(len(train_passages), len(Y_train), Counter(Y_train), Counter(t)))
        assert len(train_passages) == len(Y_train) == 400
        with open('/path/Augmentation-for-Literary-Data/dialog-results/pct_dialog/train_dial_'+str(dial)+'.pickle', 'wb') as f:
            pickle.dump(pct_quoted_fic, f)

        prediction_probs = predict(algo, train_passages, Y_train, test_passages, NGRAMS, MAX_FEATURES)
        

        # Write results to TSV:
        output_preds = '/path/Augmentation-for-Literary-Data/dialog-results/predictions/SVM_predictions_dial_'+str(dial)+'.tsv'
        print("Write predictions to:", output_preds)
        with open(output_preds, 'w') as f:
            f.write('fname\tprobability_fiction\tprediction\n')
            for ID, probs in zip(test_IDs, prediction_probs):
                if probs[0] >= 0.5: # ordering is ['fic' 'non']
                    f.write(ID+'\t'+str(probs[0])+'\tfic\n')
                else:
                    f.write(ID+'\t'+str(probs[0])+'\tnon\n')
