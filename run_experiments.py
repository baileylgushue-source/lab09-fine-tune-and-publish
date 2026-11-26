from evaluate import save_confusion_matrix
import json
from pathlib import Path

import pandas as pd

from baselines import train_majority, train_tfidf
from evaluate import calculate_scores, save_learning_curve
from finetune import predict, train_model
from prepare import sample_training, split_data

SIZES = [50, 100, 300]
SEEDS = [42, 7, 11]
LEARNING_RATE = 5e-5
EPOCHS = 3

REPORTS = Path("reports")

def final_evaluation():
    train, val, test = split_data()
    sample = sample_training(train, 300, seed=42)

    majority = train_majority(sample)
    tfidf = train_tfidf(sample, seed=42)

    model, tokenizer, _ = train_model(
        sample,
        val,
        head_only=False,
        seed=42,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )

    predictions = predict(model, tokenizer, test)

    own = pd.read_csv("data/own_messages.csv")
    own_predictions = predict(model, tokenizer, own)

    results = {
        "training_size": 300,
        "seed": 42,
        "test": {
            "majority": calculate_scores(
                test["label"], majority.predict(test["text"])
            ),
            "tfidf": calculate_scores(
                test["label"], tfidf.predict(test["text"])
            ),
            "full": calculate_scores(
                test["label"], predictions
            ),
        },
        "own_messages": calculate_scores(
            own["label"], own_predictions
        ),
    }

    save_confusion_matrix(
        test["label"],
        predictions,
        REPORTS / "confusion_matrix.png",
    )

    with open(REPORTS / "final.json", "w") as file:
        json.dump(results, file, indent=4)

    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    print(json.dumps(results, indent=4))

def final_evaluation():
    from evaluate import save_confusion_matrix

    train, val, test = split_data()
    sample = sample_training(train, 300, seed=42)
    own = pd.read_csv("data/own_messages.csv")

    majority = train_majority(sample)
    tfidf = train_tfidf(sample, seed=42)

    model, tokenizer, _ = train_model(
        sample,
        val,
        head_only=False,
        seed=42,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )

    test_predictions = predict(model, tokenizer, test)
    own_predictions = predict(model, tokenizer, own)

    results = {
        "test": {
            "majority": calculate_scores(
                test["label"], majority.predict(test["text"])
            ),
            "tfidf": calculate_scores(
                test["label"], tfidf.predict(test["text"])
            ),
            "full": calculate_scores(
                test["label"], test_predictions
            ),
        },
        "own_messages": {
            "majority": calculate_scores(
                own["label"], majority.predict(own["text"])
            ),
            "tfidf": calculate_scores(
                own["label"], tfidf.predict(own["text"])
            ),
            "full": calculate_scores(
                own["label"], own_predictions
            ),
        },
    }

    save_confusion_matrix(
        test["label"],
        test_predictions,
        REPORTS / "confusion_matrix.png",
    )

    with open(REPORTS / "final.json", "w") as file:
        json.dump(results, file, indent=4)

    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    print(json.dumps(results, indent=4))

def main():
    REPORTS.mkdir(exist_ok=True)

    train, val, _ = split_data()
    results = []

    for n in SIZES:
        for seed in SEEDS:
            print(f"\nTraining size: {n}, Seed: {seed}")

            sample = sample_training(train, n, seed=seed)

            # Baseline 1: Most common label
            majority = train_majority(sample)
            predictions = majority.predict(val["text"])

            results.append(
                {
                    "method": "majority",
                    "n": n,
                    "seed": seed,
                    **calculate_scores(val["label"], predictions),
                }
            )

            # Baseline 2: TF-IDF
            tfidf = train_tfidf(sample, seed=seed)
            predictions = tfidf.predict(val["text"])

            results.append(
                {
                    "method": "tfidf",
                    "n": n,
                    "seed": seed,
                    **calculate_scores(val["label"], predictions),
                }
            )

            # DistilBERT: Head only and full fine-tuning
            for head_only in [True, False]:
                method = "head_only" if head_only else "full"

                print(f"\nMethod: {method}")

                model, tokenizer, history = train_model(
                    sample,
                    val,
                    head_only=head_only,
                    seed=seed,
                    learning_rate=LEARNING_RATE,
                    epochs=EPOCHS,
                )

                predictions = predict(model, tokenizer, val)

                results.append(
                    {
                        "method": method,
                        "n": n,
                        "seed": seed,
                        "learning_rate": LEARNING_RATE,
                        "epochs": EPOCHS,
                        "history": json.dumps(history),
                        **calculate_scores(val["label"], predictions),
                    }
                )

                pd.DataFrame(results).to_csv(
                    REPORTS / "results.csv",
                    index=False,
                )

                del model, tokenizer

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORTS / "results.csv",
        index=False,
    )

    save_learning_curve(
        results_df,
        REPORTS / "learning_curve.png",
    )

    summary = results_df.groupby(["method", "n"])["accuracy"].agg(
        ["mean", "min", "max"]
    )

    print("\nValidation Results:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
    final_evaluation()