# While experiments.py runs hyperparameter tuning with cross-validation on the training set, this main.py is for predicting on the test data. It uses the best hyperparameters (f1-score).

import vectorize
import data_loader

import argparse
import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def predict(algo):
    """
    Train the given algorithm on all the fiction and non-fiction volumes.
    Predict on the test set.

    Returns
    -------
    List of probabilities
    """

    X_train, X_test = vectorize.ngrams_vectorize(train_sentences=train_passages,
                                                 test_sentences=test_passages,
                                                 ngram_range=NGRAMS,
                                                 max_features=MAX_FEATURES)

    clf = algo.fit(X_train, Y_train)
    assert clf.classes_.tolist()[0] == 'fic' # make sure that the class ordering is ['fic' 'non']
    preds_with_probs = clf.predict_proba(X_test) # for AUROC & AUPRC

    print("Train: {} & {} | Test: {}".format(X_train.shape, Y_train.shape, X_test.shape))
    print("Ordering:", clf.classes_)
    print("Y test shape:", preds_with_probs.shape)

    return preds_with_probs

def get_best_parameters(algorithm_name, case, experiment):
    """
    Returns the best hyperparamters for the given algorithm, scenario, and experiment (corresponding to best f1)
    """
    results_path = '/path/Augmentation-for-Literary-Data/results/'+algorithm_name+'-params-'+str(DOCUMENT_LENGTH)+'/'+algorithm_name+'_5foldCV_Case_'+case+'_exp'+experiment+'.tsv'
    print("Reading parameters from:", results_path)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='Training Data Scenario (A/B/C/D)', required=True)
    parser.add_argument('--algo', help='ML Algorithm to run (logreg/svm/rf/gbc)', required=True)
    parser.add_argument('--eda', help='Use EDA for Data Augmentation', action="store_true") # Use EDA; default: no Data Augmentation
    parser.add_argument('--cda', help='Use CDA for Data Augmentation', action="store_true") # Use CDA; default: no Data Augmentation
    args = parser.parse_args()

    DOCUMENT_LENGTH = 10000
    print("\nNOTE: Each passage has {} words".format(DOCUMENT_LENGTH))

    print("\nRunning:", args.algo, "| Scenario:", args.scenario, "| EDA:", args.eda, "| CDA:", args.cda)

    # Select algorithm:
    if args.algo == 'logreg': # Logistic Regression
        algo = LogisticRegression()
    elif args.algo == 'svm': # Support Vector Machines
        algo = SVC(probability=True)
    elif args.algo == 'rf': # Random Forest
        algo = RandomForestClassifier()
    elif args.algo == 'gbc': # Gradient Boosting Classifier
        algo = GradientBoostingClassifier()
    else:
        print("ERROR: Invalid algorithm name", args.algo)


    for experiment in ['1', '2', '3', '4', '5', '6']: # run the six experiments (increasing imbalance)
        print("\n\n\n\n\n\n\n-------------------------\nRunning experiment:", experiment)
        MAX_FEATURES, NGRAMS, param_dict = get_best_parameters(algorithm_name=args.algo, case=args.scenario, experiment=experiment) # best parameters for the given scenario & experiment
        print("Max-features = {} | N-grams = {}| Parameters: {}".format(MAX_FEATURES, NGRAMS, param_dict))
        algo.set_params(**param_dict) # set hyperparameters

        # Load training data:
        if args.eda and not args.cda: # only EDA
            train_passages, Y_train = data_loader.load_train_data_with_EDA(scenario=args.scenario)

        elif args.cda and not args.eda: # only CDA
            train_passages, Y_train = data_loader.load_train_data_with_CDA(scenario=args.scenario)

        elif args.eda and args.cda: # both EDA and CDA
            train_passages, Y_train = data_loader.load_train_data_with_EDA_and_CDA(scenario=args.scenario)
            
        else: # wihtout any Data Augmentation
            train_passages, Y_train = data_loader.load_train_data(scenario=args.scenario, N_WORDS=DOCUMENT_LENGTH, exp=experiment)

        print("\nTrain Set ---- X: {} | Y: {} | Distribution: {}".format(len(train_passages), len(Y_train), Counter(Y_train)))
        print("Y train preview:", Y_train[:3])

        # Load test data (same for each scenario, with or without augmentation):
        test_passages, Y_test, test_IDs = data_loader.load_test_data(N_WORDS=DOCUMENT_LENGTH)
        print("Test Set ---- X: {} | Y: {} | Distribution: {} | Test IDs: {}, preview: {}".format(len(test_passages), len(Y_test), Counter(Y_test), len(test_IDs), test_IDs[:3]))
        print("Y test preview:", Y_test[:3])

        # Sanity check:
        if args.scenario == 'A':
            assert len(train_passages) == len(Y_train) == 401
        else:
            assert len(train_passages) == len(Y_train) == 400

        assert len(test_passages) == len(Y_test) == 198

        prediction_probs = predict(algo)
        
        # Write results to TSV:
        output_preds = '/path/Augmentation-for-Literary-Data/results/'+str(DOCUMENT_LENGTH)+'-word-results/predictions-experiment-'+experiment+'/'+args.algo+'_EDA'+str(args.eda)+'_CDA'+str(args.cda)+'_preds_for_Case_'+str(args.scenario)+'.tsv'
        print("Write predictions to:", output_preds)
        with open(output_preds, 'w') as f:
            f.write('fname\tprobability_fiction\tlabel\n')
            for ID, probs in zip(test_IDs, prediction_probs):
                if probs[0] >= 0.5: # ordering is ['fic' 'non']
                    f.write(ID+'\t'+str(probs[0])+'\tfic\n')
                else:
                    f.write(ID+'\t'+str(probs[0])+'\tnon\n')
