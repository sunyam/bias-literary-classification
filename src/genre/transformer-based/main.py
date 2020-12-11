import numpy as np
import argparse
import sys; sys.path.append('/path/Augmentation-for-Literary-Data/src/genre-bias/traditional-machine-learning/')
import data_loader
from sklearn.metrics import f1_score #, precision_score, recall_score, accuracy_score, average_precision_score
import bert
import xlnet

def run_bert():
    """
    Runs the BERT model:
    1) Prepares data loaders.
    2) Fine-tunes the BERT model.
    3) Returns the predictions on the test set.
    """
    # DataLoader:
    train_dataloader = bert.prepare_dataloader(texts=X_train, labels=labels_train)

    print("Beginning training now..")
    # Train/fine-tune:
    bert_model = bert.train(train_dataloader)

    # Predict on test set:
    test_dataloader = bert.prepare_dataloader(texts=X_test, labels=labels_test, IDs=testIDs_idx)
    predictions, prob_fiction, true_labels, IDs_idx = bert.predict(bert_model, test_dataloader)
    print("Predictions: {}\n\nLabels:{}\n\nIDs_idx:{}".format(predictions, true_labels, IDs_idx))
    print("\n\n\n\nF1=", f1_score(true_labels, predictions, pos_label=1))
    write_predictions(IDs_idx, prob_fiction, predictions)


def run_xlnet():
    """
    Runs the XLNet model:
    1) Prepares data loaders.
    2) Fine-tunes the XLNet model.
    3) Returns the predictions on the test set.
        """
    # DataLoader:
    train_dataloader = xlnet.prepare_dataloader(texts=X_train, labels=labels_train)
    
    # Train/fine-tune:
    bert_model = xlnet.train(train_dataloader)
    
    # Predict on test set:
    test_dataloader = xlnet.prepare_dataloader(texts=X_test, labels=labels_test, IDs=testIDs_idx)
    predictions, prob_fiction, true_labels, IDs_idx = xlnet.predict(bert_model, test_dataloader)
    print("Predictions: {}\n\nLabels:{}\n\nIDs_idx:{}".format(predictions, true_labels, IDs_idx))
    print("\n\n\n\nF1=", f1_score(true_labels, predictions, pos_label=1))
    write_predictions(IDs_idx, prob_fiction, predictions)


def write_predictions(IDs_idx, prob_fiction, predictions):
    # Save predictions:
    preds_path = '/path/Augmentation-for-Literary-Data/results/predictions/'+args.model+'_preds_for_Case_'+args.scenario+'.tsv'
    print("Write predictions to:", preds_path)

    with open(preds_path, 'w') as f:
        f.write('fname\tprobability_fiction\tlabel\n')
        for index, prob, pred in zip(IDs_idx, prob_fiction, predictions):
            ID = test_IDs[int(index)]

            if prob >= 0.5:
                f.write(ID+'\t'+str(prob)+'\tfic\n')
                assert pred == 1
            else:
                f.write(ID+'\t'+str(prob)+'\tnon\n')
                assert pred == 0


def labels_str_to_int(Y):
    """
    Given the input labels, it converts them to integeres (fiction: 1 | non-fiction: 0)
    """
    labels = []
    for l in Y:
        if l == 'fic':
            labels.append(1)
        elif l == 'non':
            labels.append(0)
        else:
            print("Error:", l)
    return labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Name of the Transformer-based model to be run (bert/xlnet)', required=True)
    parser.add_argument('--scenario', help='Training Data Case (A/B/C/D)', required=True)
    parser.add_argument('--save_model', help='Save model weights & vocabulary', action="store_true")
    parser.add_argument('--eda', help='Use EDA for Data Augmentation', action="store_true") # Use EDA; default: no Data Augmentation
    parser.add_argument('--cda', help='Use CDA for Data Augmentation', action="store_true") # Use CDA; default: no Data Augmentation

    args = parser.parse_args()
    
    # Load training data:
    if args.eda: # with EDA
        X_train, Y_train = data_loader.load_train_data_with_EDA(scenario=args.scenario)
    
    elif args.cda: # with CDA
        pass
    
    else: # wihtout any Data Augmentation
        X_train, Y_train = data_loader.load_train_data(args.scenario)
    
    X_train = X_train.tolist(); Y_train = Y_train.tolist() # convert to list
    labels_train = labels_str_to_int(Y_train) # convert labels to integers

    # Test data:
    X_test, Y_test, test_IDs = data_loader.load_test_data()
    X_test = X_test.tolist(); Y_test = Y_test.tolist(); test_IDs = test_IDs.tolist() # convert to list
    labels_test = labels_str_to_int(Y_test) # convert labels to integers
    testIDs_idx = np.linspace(0, len(test_IDs), len(test_IDs), False) # can't create a tensor of strings, so create a corresponding list of indexes; we use that to index into test_IDs
    print("testIDs indexes:", len(testIDs_idx))


    if args.model == 'bert':
        run_bert()

    elif args.model == 'xlnet':
        run_xlnet()

    else:
        print("Invalid model:", args.model)
