# Dataset Loader for both train and test set
import os
import numpy as np
import pandas as pd
import random; random.seed(41)
from nltk.tokenize import word_tokenize


DATA_PATH = '/path/Augmentation-for-Literary-Data/data/gender-data/'
df = pd.read_csv('/path/Augmentation-for-Literary-Data/gender-results/NovelEnglish_Contemporary_Meta.csv')
df = df.loc[df['Author_Gender'].isin(['F','M'])]
GENDER_DICT = dict(zip(df.ID, df.Author_Gender))
print("{} files in Gender Dictionary.".format(len(GENDER_DICT)))


def get_passage(fname, two_passages=False, three_passages=False, N=500):
    """
    Returns a (continuous) passage of N words from the given txt/fname.
    If 'two_passages' (or three passages) is set to True, returns two (or three) passages in a list.
    
    Note that the beginning and end (30%) of the txt is skipped.
    """
    pct = 0.3
    with open(fname, 'r') as f:
        text = f.read()

    all_words = word_tokenize(text)
    start = int(pct*len(all_words))
    end = int(len(all_words) - pct*len(all_words))

    # print("Total words: {} | Preview: {}".format(len(all_words), all_words[10:12]))
    # print("Start:", start, "| End:", end)

    if two_passages:
        words1 = all_words[start:start+N]
        words2 = all_words[start+N:start+N+N]
#        print("Words1: {} | Words2: {}".format(len(words1), len(words2)))
        return [' '.join(words1), ' '.join(words2)]

    elif three_passages:
        words1 = all_words[start:start+N]
        words2 = all_words[start+N:start+N+N]
        words3 = all_words[start+N+N:start+N+N+N]
        print("Words1: {} | Words2: {} | Words3: {}".format(len(words1), len(words2), len(words3)))
        return [' '.join(words1), ' '.join(words2), ' '.join(words3)]

    else:
        words = all_words[start:start+N]
#        print("Words:", len(words))
        return ' '.join(words)


######## Train Set ########
def load_train_fnames():
    """
    Returns a list of filenames to be used for train-data.
    """
    fiction_fnames = [DATA_PATH+'Train/NovelEnglish_Mystery/'+fname for fname in os.listdir(DATA_PATH+'Train/NovelEnglish_Mystery/')]
    non_fiction_fnames = [DATA_PATH+'Train/NonNovel_English_Contemporary_Mixed/'+fname for fname in os.listdir(DATA_PATH+'Train/NonNovel_English_Contemporary_Mixed/')]
    print("Train Fiction fnames:", len(fiction_fnames), "| Train Non-Fiction fnames:", len(non_fiction_fnames))
    return fiction_fnames, non_fiction_fnames


def load_train_data(male_pct, return_ids=False):
    """
    Returns X and Y for training (400: 200 Fiction and 200 Non-Fiction) given the scenario. Also returns the IDs if flag is set to True.
    male_pct (between 0 & 1) represents the ratio of fiction passages written by male authors. Female = 1 - male_pct
    
    Note: loads 2-3 500-word instances per 'fiction' volume; for scenarios that don't have 200 fiction fnames, loads two instances for a few fnames.
    """
    fiction_fnames, non_fiction_fnames = load_train_fnames()
    
    MALE_FIC = male_pct*200
    FEMALE_FIC = 200 - MALE_FIC
    
    print("Target for Male Fiction: {} | Target for Female Fiction: {}".format(MALE_FIC, FEMALE_FIC))
    
    X = [] # list of training texts
    Y = [] # corresponding list of training labels
    IDs = [] # corresponding list of unique IDs
    
    male_fic_fnames, female_fic_fnames = [], []
    for fname in fiction_fnames:
        txt = fname.split('/')[-1]
        if GENDER_DICT[txt] == 'M':
            male_fic_fnames.append(fname)
        elif GENDER_DICT[txt] == 'F':
            female_fic_fnames.append(fname)
        else:
            print("Not possible!")

    N_three_fic_male = int(max(0, MALE_FIC-len(male_fic_fnames)*2))
    N_three_fic_female = int(max(0, FEMALE_FIC-len(female_fic_fnames)*2))

    male_counter, female_counter = 0, 0

    print("\nWe have {} male-fiction files and {} female-fiction files".format(len(male_fic_fnames), len(female_fic_fnames)))
    print("\n\nFor MALE: we need 2 passages from <= {} and 3 passages from {} files.".format(len(male_fic_fnames)-N_three_fic_male, N_three_fic_male))
    print("For FEMALE: we need 2 passages from <= {} and 3 passages from {} files.\n".format(len(female_fic_fnames)-N_three_fic_female, N_three_fic_female))

    if N_three_fic_male != 0:
        print("Get 3 passages from {} files: male".format(N_three_fic_male))
        for fname in male_fic_fnames[:N_three_fic_male]:
            g = GENDER_DICT[fname.split('/')[-1]]
            assert g == 'M'
            X.extend(get_passage(fname, three_passages=True))
            Y.extend(["fic", "fic", "fic"])
            IDs.append(g+'_fic_1____'+txt)
            IDs.append(g+'_fic_2____'+txt)
            IDs.append(g+'_fic_3____'+txt)
#            print(fname, "has gender ", g)
            male_counter += 3

    if N_three_fic_female != 0:
        print("Get 3 passages from {} files: female".format(N_three_fic_female))
        for fname in female_fic_fnames[:N_three_fic_female]:
            g = GENDER_DICT[fname.split('/')[-1]]
            assert g == 'F'
            X.extend(get_passage(fname, three_passages=True))
            Y.extend(["fic", "fic", "fic"])
            IDs.append(g+'_fic_1____'+txt)
            IDs.append(g+'_fic_2____'+txt)
            IDs.append(g+'_fic_3____'+txt)
#            print(fname, "has gender ", g)
            female_counter += 3

    for fname in male_fic_fnames[N_three_fic_male:]:
        if male_counter == MALE_FIC:
            print("Reached male target. Break", male_counter)
            break
        g = GENDER_DICT[fname.split('/')[-1]]
        assert g == 'M'
        X.extend(get_passage(fname, two_passages=True))
        Y.extend(["fic", "fic"])
        IDs.append(g+'_fic_1____'+txt)
        IDs.append(g+'_fic_2____'+txt)
#        print(fname, "has gender ", g)
        male_counter += 2


    for fname in female_fic_fnames[N_three_fic_female:]:
        if female_counter == FEMALE_FIC:
            print("Reached female target. Break", female_counter)
            break
        g = GENDER_DICT[fname.split('/')[-1]]
        assert g == 'F'
        X.extend(get_passage(fname, two_passages=True))
        Y.extend(["fic", "fic"])
        IDs.append(g+'_fic_1____'+txt)
        IDs.append(g+'_fic_2____'+txt)
#        print(fname, "has gender ", g)
        female_counter += 2


    for fname in non_fiction_fnames: # need two passages per txt
        X.extend(get_passage(fname, two_passages=True))
        Y.append("non")
        Y.append("non")
        IDs.append('non1____'+fname.split('/')[-1])
        IDs.append('non2____'+fname.split('/')[-1])

    if return_ids:
        return np.array(X), np.array(Y), np.array(IDs)
    else:
        return np.array(X), np.array(Y)


######## Test Set ########
def load_test_fnames():
    """
    Returns a list of filenames to be used as test-data.
    Test Data for all cases: 200 docs (100 "Non" & 100 fiction: 50 "Male" + 50 "Female")
    There are 25 'M' files and 25 'F' files in fiction. Take two passages from each fiction and one from non-fiction.
    """
    test_path = DATA_PATH + 'Test/'
    fiction_fnames = [test_path+'NovelEnglish_Mystery/'+fname for fname in os.listdir(test_path+'NovelEnglish_Mystery/')]
    non_fiction_fnames = [test_path+'NonNovel_English_Contemporary_Mixed/'+fname for fname in os.listdir(test_path+'NonNovel_English_Contemporary_Mixed/')]
    print("Test Fiction fnames:", len(fiction_fnames), "| Test Non-Fiction fnames:", len(non_fiction_fnames))
    
    return fiction_fnames, non_fiction_fnames


def load_test_data():
    """
    Returns X and Y for test set. Also returns a corresponding list of IDs.
    Take two passages from each fiction and one passage from non-fiction.
    """
    fiction_fnames, non_fiction_fnames = load_test_fnames()

    X = [] # list of texts
    Y = [] # corresponding list of labels
    IDs = [] # corresponding list of unique IDs

    for fname in fiction_fnames:
        txt = fname.split('/')[-1]
        g = GENDER_DICT[txt]
        X.extend(get_passage(fname, two_passages=True))
        Y.extend(["fic", "fic"])
        IDs.append(g+'_fic_1____'+txt)
        IDs.append(g+'_fic_2____'+txt)
    
    for fname in non_fiction_fnames:
        X.append(get_passage(fname))
        Y.append("non")
        IDs.append('non____'+fname.split('/')[-1])

    return np.array(X), np.array(Y), np.array(IDs)
