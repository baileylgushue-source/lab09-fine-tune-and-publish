import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import (
    LABEL_TO_ID,
    MAX_LENGTH,
    MODEL_NAME,
    build_model,
    predict,
    tokenize_data,
    train_model,
)


def small_batch():
    return pd.DataFrame(
        {
            "text": [
                "I was charged twice",
                "My bike has a flat tire",
                "The station has no bikes",
                "I cannot access my account",
            ],
            "label": [
                "billing",
                "bike_fault",
                "station",
                "account",
            ],
        }
    )


def test_tokenization():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data = pd.DataFrame(
        {
            "text": ["bike " * 500],
            "label": ["bike_fault"],
        }
    )

    encoded = tokenize_data(data, tokenizer)

    assert len(encoded[0]["input_ids"]) == MAX_LENGTH
    assert encoded[0]["labels"] == LABEL_TO_ID["bike_fault"]


def test_freezing_body():
    model = build_model(head_only=True)

    assert all(not p.requires_grad for p in model.distilbert.parameters())

    assert all(p.requires_grad for p in model.classifier.parameters())


def test_head_only_weights_unchanged():
    data = small_batch()
    original = build_model(head_only=True, seed=42)

    trained, _, _ = train_model(data, data, head_only=True, seed=42, epochs=1)

    assert all(
        torch.equal(before, after)
        for before, after in zip(
            original.distilbert.parameters(),
            trained.distilbert.parameters(),
            strict=True,
        )
    )


def test_full_training_changes_weights():
    data = small_batch()
    original = build_model(seed=42)

    trained, _, _ = train_model(data, data, seed=42, epochs=1)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            original.distilbert.parameters(),
            trained.distilbert.parameters(),
            strict=True,
        )
    )


def test_same_seed_repeats_run():
    data = small_batch()

    first, _, history1 = train_model(data, data, seed=42, epochs=1)

    second, _, history2 = train_model(data, data, seed=42, epochs=1)

    assert history1 == history2

    assert all(
        torch.equal(before, after)
        for before, after in zip(
            first.parameters(),
            second.parameters(),
            strict=True,
        )
    )


def test_memorize_small_batch():
    data = small_batch().iloc[[0]]
    data = pd.concat([data] * 16, ignore_index=True)

    model, tokenizer, _ = train_model(
        data,
        data,
        seed=42,
        learning_rate=0.001,
        epochs=1,
    )

    assert predict(model, tokenizer, data) == data["label"].tolist()


def test_saved_model_reload(tmp_path):
    data = small_batch()

    model, tokenizer, _ = train_model(data, data, seed=42, epochs=1)

    before = predict(model, tokenizer, data)

    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    loaded_model = AutoModelForSequenceClassification.from_pretrained(tmp_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(tmp_path)

    after = predict(loaded_model, loaded_tokenizer, data)

    assert before == after
