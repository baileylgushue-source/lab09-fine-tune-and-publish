from pathlib import Path

import gradio as gr
import pytest

from app import classify_message
from tickets_data import LABELS


def test_model_card_headings():
    card = Path("MODEL_CARD.md").read_text()

    for heading in [
        "## Intended use",
        "## Training data",
        "## Results",
        "## Limitations",
    ]:
        assert heading in card


def test_app_scores():
    scores = classify_message("My bike has a flat tire")

    assert set(scores) == set(LABELS)
    assert sum(scores.values()) == pytest.approx(1.0)
    assert all(0 <= score <= 1 for score in scores.values())


def test_empty_message():
    with pytest.raises(gr.Error, match="Please enter"):
        classify_message("")
