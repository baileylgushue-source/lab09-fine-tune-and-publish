import pandas as pd
from sklearn.model_selection import train_test_split

from tickets_data import make_tickets

SEED = 42
SAMPLE_SIZES = [50, 100, 300]


def split_data(seed: int = SEED):
    """Split the dataset into training, validation and test sets."""
    data = make_tickets()

    train, remaining = train_test_split(
        data,
        test_size=0.4,
        random_state=seed,
        stratify=data["label"],
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=seed,
        stratify=remaining["label"],
    )

    return train, validation, test


def sample_training(train: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Draw a stratified sample from the training data."""
    if n == len(train):
        return train.copy()

    sample, _ = train_test_split(
        train,
        train_size=n,
        random_state=seed,
        stratify=train["label"],
    )

    return sample
