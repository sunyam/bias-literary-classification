import csv
import os
import numpy as np
import pandas as pd

DATA_PATH = '/path/Augmentation-for-Literary-Data/data/dialogue-data/'
FIC_BOOKNLP_PATH = '/path/Augmentation-for-Literary-Data/dialog-mys-data-booknlp-output/'
NON_BOOKNLP_PATH = '/path/Augmentation-for-Literary-Data/dialog-non-data-booknlp-output/'


def get_passage(df, start, N=500):
    """
    Given the of the BookNLP output DataFrame & index of the starting word (start), this function
    returns a 500-word passage starting at index 'start'. Also returns % of words in dialogue.
    """
    id_list = list((range(start, start+N)))
#     print("Get 500 words from index {} to {}".format(id_list[0], id_list[-1]))
    df = df.loc[df['tokenId'].isin(id_list)]
    words = df['originalWord'].tolist()
    if len(words) != N:
        print("Word-count is:", len(words))

    quoted_words = df.loc[df['inQuotation']=='I-QUOTE']['originalWord'].tolist() # filter again incase BookNLP missed any
#     print("Quoted:", len(quoted_words), quoted_words[:4])
    return ' '.join(words), len(quoted_words)


def sample_random_text(fname, two_passages=False):
    """
    Returns a random 500-word passage from the middle of the given volume (used for non-fiction).
    """
    N_WORDS = 500; pct = 0.3
    df = pd.read_csv(fname, delimiter='\t', quoting=csv.QUOTE_NONE) # no quotechar
    df.dropna(subset=['originalWord'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    total_words = df.shape[0]
    first = int(pct*total_words)
    last = int(total_words - pct*total_words)

    if two_passages:
#         print("Sample 2 random passages from", fname)
        start_index = first
        text1, _ = get_passage(df, start_index)
        text2, _ = get_passage(df, start_index+N_WORDS)
        return [text1, text2]

    else:
#         print("Sample 1 passage from", fname)
        start_index = first
        text, _ = get_passage(df, start_index)
        return text


def sample_texts(fname, dialog=True):
    """
    If dialog is False, returns two 500-word passages with zero dialogue.

    If dialog is True, samples all possible 500 word passages from the given novel (30% text from either side is skipped)
    And returns the top two samples with most dialogue along with the % dialog in those two texts.
    """
    N_WORDS = 500; pct = 0.3
    df = pd.read_csv(fname, delimiter='\t', quoting=csv.QUOTE_NONE) # no quotechar
    df.dropna(subset=['originalWord'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    total_words = df.shape[0]
    inside_quotes = df['inQuotation'].value_counts().to_dict()["I-QUOTE"]
    first = int(pct*total_words)
    last = int(total_words - pct*total_words)

#     print("Total words in the volume: {} | Words inside quotes (BookNLP): {}".format(total_words, inside_quotes))
#     print("Sample random passages from word index {} to {}".format(first, last))

    map_i_tup = {} # maps run number 'i' to a tuple of (quoted words & corresponding text)
    non_dialog_texts = []
    start_index = first

    i = 0
    while True:
        if start_index + N_WORDS >= last:
            break

        text, quoted = get_passage(df, start_index)
        start_index += N_WORDS

        if len(non_dialog_texts) < 2 and quoted == 0:
            non_dialog_texts.append(text)

        map_i_tup[i] = (quoted, text)
        i += 1

    if not dialog:
        return non_dialog_texts

    if dialog:
        sorted_keys = sorted(map_i_tup, key=lambda k: map_i_tup[k][0])
        dialog_texts, pct_quoted = [], []
        for key in sorted_keys[-2:]:
            if map_i_tup[key][0] == 0: # did not have quoted words, return -1
#                print("OOOOPS::::: ID number:", key, "has quoted words:", map_i_tup[key][0])
                return -1

#             print("ID number:", key, "has quoted words:", map_i_tup[key][0])
            dialog_texts.append(map_i_tup[key][1])
            pct_quoted.append(map_i_tup[key][0]/N_WORDS)

        return dialog_texts, pct_quoted


#### Train-Data ####

def load_train_fnames():
    """
    Returns a list of filenames to be used for train-data.
    """
    fiction_fnames = [DATA_PATH+'Train/NovelEnglish_Mystery/'+fname for fname in os.listdir(DATA_PATH+'Train/NovelEnglish_Mystery/')]
    non_fiction_fnames = [DATA_PATH+'Train/NonNovel_English_Contemporary_Mixed/'+fname for fname in os.listdir(DATA_PATH+'Train/NonNovel_English_Contemporary_Mixed/')]
    print("Train Fiction fnames:", len(fiction_fnames), "| Train Non-Fiction fnames:", len(non_fiction_fnames))
    return fiction_fnames, non_fiction_fnames


def load_train_data(dial, no_dial, return_ids=True, N_WORDS=500):
    """
    Returns X and Y for training (len=400), given the experiment and the scenario. Also returns the IDs if flag is set to True.
    Training = 200 fic / 200 nonfic

    The 200 fiction volumes has dialogue/no-dialogue distributions as specified by 'dial' & 'no_dial'.
    dial represents the percent of the fiction-train-set that should have dialog and no_dial represents without-dialog.
    They should add up to 1.

    The 200 nonfic has random 500-word passasges from the non-fiction volumes.

    2 passages per volume.
    """
    fiction_fnames, non_fiction_fnames = load_train_fnames()
    assert len(fiction_fnames) > len(non_fiction_fnames) == 100

    assert dial + no_dial == 1

    X, Y, IDs = [], [], [] # corresponding list of texts, labels, and unique IDs

    with_dial, without_dial = 0, 0 # counters
    pct_quoted_fic = [] # keep track of how much "dialog" we have in our dialog data

    for fname in fiction_fnames:
        if with_dial == dial*200 and without_dial == no_dial*200:
            break

        fname = FIC_BOOKNLP_PATH + fname.split('/')[-1] + '/' + fname.split('/')[-1]
        if not os.path.isfile(fname+'.tokens'):
            print(fname, "doesn't exist. Skip!")
            continue

        if without_dial < no_dial*200: # look for passages without-dialog
            try:
                ret = sample_texts(fname+'.tokens', dialog=False)
                assert len(ret) == 2
                X.extend(ret)
                without_dial += 2
                IDs.append("ficNoDialog1____" + fname.split('/')[-1])
                IDs.append("ficNoDialog2____" + fname.split('/')[-1])
            except:
                if with_dial >= dial*200:
#                    print("Have already reached the limit for with-dialogs: {} {}\tSkip!".format(with_dial, dial*200))
                    continue
#                print("Could not find zero-dialog passages in {} | Try for with-dialogue..".format(fname.split('/')[-1]))
                ret = sample_texts(fname+'.tokens', dialog=True)
                if ret == -1:
                    print("Returned -1. Skip!")
                    continue
                X.append(ret[0][0])
                X.append(ret[0][1])
                pct_quoted_fic.append(ret[1][0])
                pct_quoted_fic.append(ret[1][1])
                with_dial += 2
                IDs.append("ficWithDialog1____" + fname.split('/')[-1])
                IDs.append("ficWithDialog2____" + fname.split('/')[-1])

        elif with_dial < dial*200: # look for passages with-dialog
            ret = sample_texts(fname+'.tokens', dialog=True)
            try:
                X.append(ret[0][0])
                X.append(ret[0][1])
                pct_quoted_fic.append(ret[1][0])
                pct_quoted_fic.append(ret[1][1])
                with_dial += 2
                IDs.append("ficWithDialog1____" + fname.split('/')[-1])
                IDs.append("ficWithDialog2____" + fname.split('/')[-1])
            except:
                print(fname, "does not have quoted words.. Skip!")
                continue

        Y.append("fic")
        Y.append("fic")
#        print("With dial: {} | Without dial: {} | Pct quoted: {} | Y: {} | X: {}".format(with_dial, without_dial, len(pct_quoted_fic), len(Y), len(X)))
#    print("End of fiction-fnames! X: {} | Y: {}".format(len(X), len(Y)))

    for fname in non_fiction_fnames: # all random
        IDs.append("non1____" + fname.split('/')[-1])
        IDs.append("non2____" + fname.split('/')[-1])
        fname = NON_BOOKNLP_PATH + fname.split('/')[-1] + '/' + fname.split('/')[-1] + '.tokens'
        X.extend(sample_random_text(fname, two_passages=True))
        Y.extend(["non", "non"])

    assert with_dial == dial*200 == len(pct_quoted_fic)
    assert without_dial == no_dial*200
    assert len(X) == len(Y) == len(IDs) == 400

    if return_ids:
        return np.array(X), np.array(Y), np.array(pct_quoted_fic), np.array(IDs)
    else:
        return np.array(X), np.array(Y), np.array(pct_quoted_fic)


#### Test-Data ####

def load_test_fnames():
    """
    Returns a list of filenames to be used as test-data.
    Test Data for all cases: 200 docs (100 "Non" & 100 fiction: 50 "with dialog" + 50 "without dialog")
    """
    test_path = DATA_PATH + 'Test/'
    fiction_fnames = [test_path+'NovelEnglish_Mystery/'+fname for fname in os.listdir(test_path+'NovelEnglish_Mystery/')]
    non_fiction_fnames = [test_path+'NonNovel_English_Contemporary_Mixed/'+fname for fname in os.listdir(test_path+'NonNovel_English_Contemporary_Mixed/')]
    print("Test Fiction fnames:", len(fiction_fnames), "| Test Non-Fiction fnames:", len(non_fiction_fnames))

    return fiction_fnames, non_fiction_fnames


def load_test_data():
    """
    Returns X and Y for test set. Also returns a corresponding list of IDs.
    100 random non-fiction passages + 50 fiction passages with-dialog + 50 fiction passages without-dialog

    Each passage is contiguous 500-words from the volume. Uses one passage per volume.
    """
    fiction_fnames, non_fiction_fnames = load_test_fnames()

    assert len(fiction_fnames) == len(non_fiction_fnames) == 100

    X, Y, IDs = [], [], [] # corresponding list of texts, labels, and unique IDs

    with_dial, without_dial = 0, 0
    pct_quoted_fic = []

    for fname in fiction_fnames:
        fname = FIC_BOOKNLP_PATH + fname.split('/')[-1] + '/' + fname.split('/')[-1]

        if without_dial < 50:
            try:
                X.append(sample_texts(fname+'.tokens', dialog=False)[0])
                without_dial += 1
                IDs.append("ficNoDialog____" + fname.split('/')[-1])
            except:
#                print("Could not find zero-dialog passages in {} | Try for with-dialogue..".format(fname.split('/')[-1]))
                ret = sample_texts(fname+'.tokens', dialog=True)
                X.append(ret[0][1])
                pct_quoted_fic.append(ret[1][1])
                with_dial += 1
                IDs.append("ficWithDialog____" + fname.split('/')[-1])

        else:
            ret = sample_texts(fname+'.tokens', dialog=True)
            try:
                X.append(ret[0][1])
                pct_quoted_fic.append(ret[1][1])
                with_dial += 1
                IDs.append("ficWithDialog____" + fname.split('/')[-1])
            except:
#                print(fname, "does not have quoted words. Skip!")
                continue

        Y.append("fic")
#        print("With dial: {} | Without dial: {} | Pct quoted: {} | Y: {}".format(with_dial, without_dial, len(pct_quoted_fic), len(Y)))
        if with_dial == without_dial == 50:
            break

    for fname in non_fiction_fnames: # random passages
        IDs.append("non____" + fname.split('/')[-1])
        fname = NON_BOOKNLP_PATH + fname.split('/')[-1] + '/' + fname.split('/')[-1] + '.tokens'
        X.append(sample_random_text(fname))
        Y.append("non")

    assert with_dial == without_dial == len(pct_quoted_fic) == 50
    assert len(X) == len(Y) == len(IDs) == 200
    return np.array(X), np.array(Y), np.array(pct_quoted_fic), np.array(IDs)
