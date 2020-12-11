# Our Custom Data Augmentation (CDA) technique involves (1) Back-Translation, (2) Crossover, (3) Substituting & Deleting Proper Names | generate 16 instances per training instance

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import random; random.seed(41)


# Using the first-names "all" version & surnames "us" version from https://github.com/smashew/NameDatabases
with open('/path/Augmentation-for-Literary-Data/data/names/first_names_all.txt', 'r', errors='ignore', encoding='utf8') as r:
    FIRST_NAMES = set(r.read().strip().split('\n'))
    
with open('/path/Augmentation-for-Literary-Data/data/names/surnames_us.txt', 'r', errors='ignore', encoding='utf8') as r:
    LAST_NAMES = set(r.read().strip().split('\n'))
    
print("We have unique {} first names and {} last names".format(len(FIRST_NAMES), len(LAST_NAMES)))


TAGGER = StanfordNERTagger('/path/stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                           '/path/stanford-ner-4.0.0/stanford-ner-4.0.0.jar',
                           encoding='utf-8')


def proper_name_present(text):
    """
    Returns -1 if proper name is not present in the given text.
    If present, returns the NER tagged text.
    """    
    classified_text = TAGGER.tag(word_tokenize(text)) # NER Tagging

    for word, tag in classified_text:
        if tag == 'PERSON':
            return classified_text
        
    # if no tag in the loop is 'PERSON'
    return -1


def proper_names(classified_text, action):
    """
    Given the NER-classified text, we can perform two actions: 'delete' or 'substitute'.
    - deletes all proper names
    - substitutes all proper names with random names from https://github.com/smashew/NameDatabases
    """
    augmented_text = ""
    
    for i, tup in enumerate(classified_text):
        word, tag = tup
        surname = False; first_name = False
        if tag == 'PERSON': # for substitue (need to figure out first/surname)
            if action == 'delete':
                continue
            
            elif action == 'substitute':
                if classified_text[i-1][1] == 'PERSON':
                    surname = True
                else:
                    first_name = True

                if first_name: # randomly substitute one
                    augmented_text += " " + random.sample(FIRST_NAMES, 1)[0]
                elif surname:
                    augmented_text += " " + random.sample(LAST_NAMES, 1)[0]

        else:
            augmented_text += " " + word
    
    return augmented_text


def back_translation(fname):
    """
    Given the fname, this funciton returns 4 back-translated passages (French, German, Korean, Spanish).
    
    See Back-Translation notebook for the translation details.
    """
    path = '/path/Augmentation-for-Literary-Data/data/back-translated/'
    languages = ['fr', 'ko', 'de', 'es'] # French, Korean, German, Spanish
    
    texts = []
    for lang in languages:
        with open(path+fname+'__lang_'+lang+'.txt', 'r') as f:
            t = f.read()
        texts.append(t)
    return texts


def crossover(text1, text2):
    """
    Returns a new text instance after performing crossover (index: half of text1).
    First half of text1 + second half of text 2
    """
    text1 = text1.split(' ')
    text2 = text2.split(' ')
    
    i = int(len(text1)/2)
            
    text1_part1 = text1[:i]
    text1_part2 = text1[i:]
    
    text2_part1 = text2[:i]
    text2_part2 = text2[i:]
    
    new_text = text1_part1 + text2_part2
    
    return ' '.join(new_text)


def perform_crossover(main_text, second_texts):
    """
    Crossovers the given main_text with each of the second_texts.
    Return len(second_texts) augmented instances
    """
    assert len(second_texts) <= 16
    X = []
    
    for second in second_texts:
        X.append(crossover(main_text, second))
        
    return X