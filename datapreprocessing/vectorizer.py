# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

# Custom Lemmatizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]

# Custom Tfidf Vectorizer
class TfidfVectors(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 1), tokenizer=LemmaTokenizer())

    def fit(self, df, y=None):
        self.tfidf_vectorizer.fit(df)
        return self

    def transform(self, df):
        features = self.tfidf_vectorizer.transform(df)
        return features
