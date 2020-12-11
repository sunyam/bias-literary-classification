import pickle
import os
import numpy as np
import random; random.seed(41)
from nltk.tokenize import word_tokenize
import sys
sys.path.append('/path/Augmentation-for-Literary-Data/src/augmentation/')
import eda
import cda


DATA_PATH = '/path/Augmentation-for-Literary-Data/data/genre-data/'

def get_passage(fname, N, two_passages=False):
    """
    Returns a (continuous) passage of N words from the given txt/fname.
    If 'two_passages' is True, it returns two passages (list) instead of one.

    Note that the beginning and end (20/30%) of the txt is skipped.
    """
    pct = 0.2 # 0.3 for most experiments; 0.2 for document length > 5k
    with open(fname, 'r') as f:
        text = f.read()

    all_words = word_tokenize(text)
    start = int(pct*len(all_words))
    end = int(len(all_words) - pct*len(all_words))

    # print("Total words: {} | Preview: {}".format(len(all_words), all_words[10:12]))
    # print("Start:", start, "| End:", end)

    if two_passages:
        #assert start+N+N < end
        if start+N+N > end:
            print("Not enough words.. using all the avaialable words. Total words: {} | Start: {} | End: {}".format(len(all_words), start, end))
            words1 = all_words[start:start+N]
            words2 = all_words[start+N:]

        else:
            words1 = all_words[start:start+N]
            words2 = all_words[start+N:start+N+N]
        # print("Words1: {} | Words2: {}".format(len(words1), len(words2)))
        return [' '.join(words1), ' '.join(words2)]
    else:
        words = all_words[start:start+N]
        # print("Words:", len(words))
        return ' '.join(words)


######## Train Set ########
def load_train_data(scenario, exp, N_WORDS, return_ids=False):
    """
    Returns X and Y for training (len=400), given the experiment and the scenario. Also returns the IDs if flag is set to True.
    Note: loads two 500-word instances per "Non-Fiction" volume; for scenarios that don't have 200 fiction fnames, loads two instances for a few fnames.
    """
    print("Load training data for Experiment:", exp)
    with open(DATA_PATH + 'train_fnames_scenario_dict_exp_'+str(exp)+'.pickle', 'rb') as f: # contains training txt filenames for each scenario
        TRAIN_FNAMES = pickle.load(f)

    # Load train fnames:
    fiction_fnames = TRAIN_FNAMES[scenario]['fiction_fnames']
    non_fiction_fnames = TRAIN_FNAMES[scenario]['non_fiction_fnames']

    if len(fiction_fnames) != 200 and scenario != 'A': # because scenario A has 201 (67+67+67) fnames
        N_two_fic = 200 - len(fiction_fnames) # number of fiction txts we need two 500-word passages from
        print("We have {} fiction fnames".format(len(fiction_fnames)))

    else:
        print("We have exactly {} fiction fnames".format(len(fiction_fnames)))
        N_two_fic = 0

    assert len(non_fiction_fnames) == 100

    print("Intersection between fic and nonfic fnames:", set(fiction_fnames).intersection(set(non_fiction_fnames)))
    X = [] # list of training texts
    Y = [] # corresponding list of training labels
    IDs = [] # corresponding list of unique IDs

    if N_two_fic != 0:
        print("Getting 2 passages from {} fiction files".format(N_two_fic))
        for fname in fiction_fnames[:N_two_fic]:
            print("Get 2 passages from:", fname)
            X.extend(get_passage(fname, N_WORDS, two_passages=True))
            Y.append("fic")
            Y.append("fic")
            IDs.append(fname.split('/')[-1][:-4]+'__1')
            IDs.append(fname.split('/')[-1][:-4]+'__2')

    print("X: {} | Y: {}".format(len(X), len(Y)))

    for fname in fiction_fnames[N_two_fic:]:
        X.append(get_passage(fname, N_WORDS))
        Y.append("fic")
        IDs.append(fname.split('/')[-1][:-4])

    for fname in non_fiction_fnames: # need two "passages" per txt
        X.extend(get_passage(fname, N_WORDS, two_passages=True))
        Y.append("non")
        Y.append("non")
        IDs.append(fname.split('/')[-1][:-4]+'__1')
        IDs.append(fname.split('/')[-1][:-4]+'__2')

    if return_ids:
        return np.array(X), np.array(Y), np.array(IDs)
    else:
        return np.array(X), np.array(Y)


######## Test Set ########
def load_test_fnames():
    """
    Returns a list of filenames to be used as test-data.
    Test Data for all cases: 198 docs (99 "Non" & 99 fiction: 33 "Mys" + 33 "Rom" + 33 "SciFi")
    """
    test_path = DATA_PATH + 'Test-Set/'

    mys = [test_path+'Mystery_TestSet/'+fname for fname in os.listdir(test_path+'Mystery_TestSet/')]
    rom = [test_path+'Romance_TestSet/'+fname for fname in os.listdir(test_path+'Romance_TestSet/')]
    sci = [test_path+'SciFi_TestSet/'+fname for fname in os.listdir(test_path+'SciFi_TestSet/')]
    fiction_fnames = mys + sci + rom
    random.shuffle(fiction_fnames)

    non_fiction_fnames = [test_path+'NonNovel_TestSet/'+fname for fname in os.listdir(test_path+'NonNovel_TestSet/')]
    print("Test Fiction fnames:", len(fiction_fnames), "| Test Non-Fiction fnames:", len(non_fiction_fnames))
    return fiction_fnames, non_fiction_fnames


def load_test_data(N_WORDS):
    """
    Returns X and Y for test set. Also returns a corresponding list of IDs.
    """
    fiction_fnames, non_fiction_fnames = load_test_fnames()

    X = [] # list of texts
    Y = [] # corresponding list of labels
    IDs = [] # corresponding list of unique IDs

    for fname in fiction_fnames:
        IDs.append(fname.split('/')[-1])
        X.append(get_passage(fname, N_WORDS))
        Y.append("fic")

    for fname in non_fiction_fnames:
        IDs.append(fname.split('/')[-1])
        X.append(get_passage(fname, N_WORDS))
        Y.append("non")

    return np.array(X), np.array(Y), np.array(IDs)


######## Train Set with Augmentation ########
def load_train_data_with_EDA(scenario, N_aug=16):
    """
    Returns X and Y for training, given the scenario. Data is augmented 16 folds using one of the four EDA techniques at random. Only used with Experiment 1.
    """
    print("Generate {} new instances per instance using EDA".format(N_aug))
    X, Y = load_train_data(scenario)

    augmented_X = X.tolist()
    augmented_Y = Y.tolist()

    operations = [eda.synonym_replacement, eda.random_insertion, eda.random_swap, eda.random_deletion]
    for instance, label in zip(X, Y):
        print("X so far:", len(augmented_X), "| Y so far:", len(augmented_Y))#, "| Y:", augmented_Y)
        for _ in range(N_aug):
            operation = random.choice(operations)
            new_text = operation(instance)
            augmented_X.append(new_text)
            augmented_Y.append(label)

    return np.array(augmented_X), np.array(augmented_Y)



def load_train_data_with_CDA(scenario):
    """
    Returns X and Y for training, given the scenario. Data is augmented 16 folds using our CDA technique. Only used with Experiment 1.
    | Back translation (4) | Crossover (10 or 12) | Proper Name substitution & deletion (2 or 0) |
    """
    print("Generate 16 new instances per instance using CDA")
    X, Y, IDs = load_train_data(scenario, return_ids=True)

    augmented_X = X.tolist()
    augmented_Y = Y.tolist()

    for instance, label, ID in zip(X, Y, IDs):
        print("X so far:", len(augmented_X), "| Y so far:", len(augmented_Y))#, "| Y:", augmented_Y)

        # BackTranslation: 4 back-translated augmented instances:
        fname = ID + '__' + label
        translated = cda.back_translation(fname)
        augmented_X.extend(translated)

        # Check for Proper Names
        ner_tagged = cda.proper_name_present(instance)
        if ner_tagged == -1:
            print("No proper names present in", ID)
            N_crossover = 12

        else: # if Proper Names exist, get 2 augmented instances (substitute all/delete all)
            pn_deleted = cda.proper_names(ner_tagged, action='delete')
            pn_substituted = cda.proper_names(ner_tagged, action='substitute')
            augmented_X.append(pn_deleted)
            augmented_X.append(pn_substituted)
            N_crossover = 10

        # Get the rest (10 or 12) crossover augmented instances:
        random_X = random.sample(list(X), N_crossover)
        crossed = cda.perform_crossover(instance, random_X)
        augmented_X.extend(crossed)

        # Add labels:
        augmented_Y.extend([label]*16)

    return np.array(augmented_X), np.array(augmented_Y)


def load_train_data_with_EDA_and_CDA(scenario):
    """
    Returns X and Y for training, given the scenario. Data is augmented 16 folds using both EDA (8) and CDA (8). Only used with Experiment 1.
    """
    print("Generate 16 new instances per instance using a combination of EDA and CDA.")
    X, Y, IDs = load_train_data(scenario, return_ids=True)

    augmented_X = X.tolist()
    augmented_Y = Y.tolist()

    operations = [eda.synonym_replacement, eda.random_insertion, eda.random_swap, eda.random_deletion]

    for instance, label, ID in zip(X, Y, IDs):
        #### 8 EDA Instances ####
        N_eda = 8
        print("Generate {} new instances per instance using EDA".format(N_eda))
        print("X so far:", len(augmented_X), "| Y so far:", len(augmented_Y))
        for _ in range(N_eda):
            operation = random.choice(operations)
            new_text = operation(instance)
            augmented_X.append(new_text)
        print("After EDA: X so far:", len(augmented_X))


        #### 8 CDA Instances: 4 BT + 2/0 ProperNames + 2/4 Crossover ####
        fname = ID + '__' + label
        translated = cda.back_translation(fname)
        augmented_X.extend(translated)

        ner_tagged = cda.proper_name_present(instance)
        if ner_tagged == -1:
            print("No proper names present in", ID)
            N_crossover = 4
        else: # if Proper Names exist, get 2 augmented instances (substitute all/delete all)
            pn_deleted = cda.proper_names(ner_tagged, action='delete')
            pn_substituted = cda.proper_names(ner_tagged, action='substitute')
            augmented_X.append(pn_deleted)
            augmented_X.append(pn_substituted)
            N_crossover = 2

        # Crossover augmented instances:
        random_X = random.sample(list(X), N_crossover)
        crossed = cda.perform_crossover(instance, random_X)
        augmented_X.extend(crossed)
        print("After CDA: X so far:", len(augmented_X))

        # Add labels:
        augmented_Y.extend([label]*16)

    return np.array(augmented_X), np.array(augmented_Y)
