# Easy Data Augmentation (EDA) techniques from https://www.aclweb.org/anthology/D19-1670.pdf
# As recommended in Table 3, we use alpha=0.05 (for RD, p=alpha); n = 25 | generate 16 instances per training instance
# Note that in order to generate a new instance, randomly choose and perform one of the four EDA operations

import nlpaug.augmenter.word as naw
import nlpaug.model.word_dict as nmw
import random; random.seed(41)
import nltk
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')
ALPHA = 0.05
N = int(ALPHA*500)
print("EDA Parameters: N = {} | alpha = {}".format(N, ALPHA))

# SR:
def synonym_replacement(text, n=N):
    """
    Randomly choose n words from the sentence that are not stop words. Replace each of these words with one of its
    synonyms chosen at random.
    """
    aug = naw.SynonymAug(aug_src='wordnet', aug_min=n, aug_max=n, stopwords=english_stopwords)
    augmented_text = aug.augment(text)
    return augmented_text

# ------------------- #

# RS:
def random_swap(text):
    """
    Performs random swap N times.
    """
    for i in range(N):
        text = random_swap_helper(text)
        # print("After run {}, text is {}".format(i+1, text))
    return text

def random_swap_helper(text):
    """
    Randomly choose two words in the sentence and swap their positions.
    """
    aug = naw.RandomWordAug(action='swap', aug_min=1, aug_max=1)
    augmented_text = aug.augment(text)
    return augmented_text

# ------------------- #

# RD:
def random_deletion(text, p=ALPHA):
    """
    Randomly remove each word in the sentence with probability p=0.05
    """
    aug = naw.RandomWordAug(action='delete', aug_p=p)
    augmented_text = aug.augment(text)
    return augmented_text

# ------------------- #

# RI:
def random_insertion(text):
    """
    Performs random insertion N times.
    """
    for i in range(N):
        text = random_insertion_helper(text)
        # print("After run {}, text is: {}".format(i+1, text))
    return text

def random_insertion_helper(text):
    """
    Find a random synonym of a random word in the sentence that is not a stop word.
    Insert that synonym into a random position in the sentence.
    """
    original_words = nltk.word_tokenize(text)

    # pick a random word and get its synonyms:
    candidate_syns, candidate_word = get_random_words_synonyms(original_words)

    # pick a random synonym:
    final_synonym = random.choice(candidate_syns)

    # insert at a random position:
    rand_index = random.randint(0, len(original_words)-1)
    original_words.insert(rand_index, final_synonym)

    # print("Original word:", candidate_word)
    # print("Final synonym:", final_synonym)

    return ' '.join(original_words)

def get_random_words_synonyms(original_words):
    """
    Helper for RI: picks a random word in 'original_words' which is not a stopword. Returns a list of its synonyms (and the word).
    """
    model = nmw.WordNet(lang='eng', is_synonym=True)
    filtered_words = [w for w in original_words if w not in english_stopwords] # remove stopwords
    while True:
        candidate_word = random.choice(filtered_words)
        # print("Candidate:", candidate_word)
        candidate_syns = model.predict(candidate_word)
        # print("Before:", candidate_syns)
        if candidate_word in candidate_syns: # remove all occurrences of candidate_word in candidate_syns
            candidate_syns = list(filter(lambda a: a != candidate_word, candidate_syns))
            # print("After:", candidate_syns)
        if candidate_syns: # return, if not empty
            return candidate_syns, candidate_word

# ------------------- #

# # Testing:
# text = "The quick brown fox jumps over the lazy dog. The lazy dog is sleeping."
# print("Running for N={} | alpha={}".format(N, ALPHA))
#
# print("\nOriginal:")
# print(text)
# print("\n\n-------\n SR Augmented Text:")
# print(synonym_replacement(text))
#
# print("\n\n-------\n RD Augmented Text:")
# print(random_deletion(text))
#
# print("\n\n-------\n RS Augmented Text:")
# print(random_swap(text))
#
# print("\n\n-------\n RI Augmented Text:")
# print(random_insertion(text))
