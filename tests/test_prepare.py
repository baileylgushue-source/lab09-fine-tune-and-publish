import pytest

from prepare import sample_training, split_data


@pytest.mark.parametrize("seed", [42, 7, 11])
def test_split(seed):
    train, val, test = split_data(seed)

    assert len(train) == 720
    assert len(val) == 240
    assert len(test) == 240

    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_sample_size():
    train, _, _ = split_data()

    for n in [50, 100, 300]:
        sample = sample_training(train, n)
        assert len(sample) == n


def test_sample_labels():
    train, _, _ = split_data()
    sample = sample_training(train, 50)

    assert set(sample["label"]) == set(train["label"])


def test_sample_from_training():
    train, _, _ = split_data()
    sample = sample_training(train, 100)

    assert set(sample.index).issubset(train.index)
