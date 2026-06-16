"""
Classi custom per il pipeline di sentiment analysis.
Necessarie per deserializzare il modello pickle.
"""

import pandas as pd
import simplemma
from sklearn.base import BaseEstimator, TransformerMixin


def clean_colums(df, column, patterns):
    """Pulisce il testo secondo i pattern forniti"""
    for pattern, replacement in patterns.items():
        df[column] = df[column].str.replace(pattern, replacement, regex=True)
    df[column] = df[column].str.lower()
    return df


class text_cleaner(BaseEstimator, TransformerMixin):
    """Transformer per pulire il testo"""
    def __init__(self, patterns):
        self.patterns = patterns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_list = list(X) if not isinstance(X, list) else X
        df = pd.DataFrame({'text': X_list})
        df = clean_colums(df, 'text', self.patterns)
        return df['text'].tolist()


class LemmatizzatoreTransformer(BaseEstimator, TransformerMixin):
    """Transformer per lemmatizzare il testo"""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        testi = pd.Series(X).astype(str).tolist()
        lemmi_totali = []
        for testo in testi:
            lemmi = ' '.join(simplemma.lemmatize(word, lang='en') for word in testo.split())
            lemmi_totali.append(lemmi)
        return lemmi_totali
