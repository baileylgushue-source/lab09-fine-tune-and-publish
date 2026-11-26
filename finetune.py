import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from tickets_data import LABELS

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16

LABEL_TO_ID = {
    label: i for i, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    i: label for i, label in enumerate(LABELS)
}


def tokenize_data(data, tokenizer):
    """Tokenize messages and preserve their labels."""
    encoded = tokenizer(
        data["text"].tolist(),
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    return [
        {
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "labels": LABEL_TO_ID[label],
        }
        for i, label in enumerate(data["label"])
    ]


def build_model(head_only=False, seed=42):
    """Load DistilBERT and optionally freeze its body."""
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    if head_only:
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False

    return model


def predict(model, tokenizer, data):
    """Predict a category for each message."""
    collator = DataCollatorWithPadding(tokenizer)

    loader = DataLoader(
        tokenize_data(data, tokenizer),
        batch_size=BATCH_SIZE,
        collate_fn=collator,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            outputs = model(**batch)
            ids = outputs.logits.argmax(dim=1).tolist()
            predictions.extend(ID_TO_LABEL[i] for i in ids)

    return predictions


def train_model(
    train,
    val,
    head_only=False,
    seed=42,
    learning_rate=5e-5,
    epochs=3,
):
    """Fine-tune DistilBERT and log each epoch."""
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(head_only, seed)

    collator = DataCollatorWithPadding(tokenizer)

    loader = DataLoader(
        tokenize_data(train, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        predictions = predict(model, tokenizer, val)
        accuracy = accuracy_score(val["label"], predictions)

        history.append({
            "epoch": epoch + 1,
            "training_loss": average_loss,
            "validation_accuracy": accuracy,
        })

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={average_loss:.4f}, "
            f"Validation accuracy={accuracy:.4f}"
        )

    return model, tokenizer, history