# Contains methods to vectorize a piece of text

from sklearn.feature_extraction.text import TfidfVectorizer

def ngrams_vectorize(train_sentences, test_sentences, ngram_range, max_features):
    """
    Vectorizes the input text using bag of n-grams approach. Uses word TFIDF vectorizer.
    Note that we also ignore words that appear in >90% of pages.

    Parameters
    ----------
    train_sentences: numpy array of train sentences (pages)
    test_sentences: numpy array of test sentences (pages)
    ngram_range: For TFIDF - tuple for the value of n in n-grams
    max_features: For TFIDF - build a vocabulary that only consider the top max_features ordered by term frequency across the corpus

    Returns
    -------
    X_train, X_test
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                 max_features=max_features,
                                 analyzer='word',
                                 encoding='utf-8',
                                 decode_error='ignore',
                                 max_df=0.9)

    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    return X_train, X_test
