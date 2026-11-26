from baselines import train_majority, train_tfidf
from prepare import sample_training, split_data


def test_tfidf_beats_majority():
    train, val, _ = split_data()
    sample = sample_training(train, 300)

    majority = train_majority(sample)
    tfidf = train_tfidf(sample)

    majority_accuracy = majority.score(
        val["text"], val["label"]
    )

    tfidf_accuracy = tfidf.score(
        val["text"], val["label"]
    )

    assert tfidf_accuracy > majority_accuracy