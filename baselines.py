import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_majority(train: pd.DataFrame):
    """Train the most common label baseline."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train["text"], train["label"])
    return model


def train_tfidf(train: pd.DataFrame, seed: int = 42):
    """Train the TF-IDF and logistic regression baseline."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=seed,
        )),
    ])

    model.fit(train["text"], train["label"])
    return model