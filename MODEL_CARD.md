---
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
pipeline_tag: text-classification
---

# Lakeshore Ticket Sorter

## Intended use

This model classifies customer support messages for
Lakeshore Bike Share into four categories: billing,
bike_fault, station and account. It is intended to
help support staff sort incoming customer messages.

## Training data

The original dataset contains 1,200 invented customer
support messages. A stratified 60/20/20 split was used,
with 720 training, 240 validation and 240 test messages.

The final model was fully fine-tuned using 300 training
messages, a learning rate of 5e-5, three epochs and
seed 42.

## Results

All methods used the same sampled training rows and
evaluation datasets.

| Method | Mean Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| Most common label | 28.75% | 29.17% | 0.1129 |
| TF-IDF | 86.67% | 82.92% | 0.8286 |
| Full DistilBERT | 91.39% | 88.75% | 0.8899 |

The final model achieved 82.50% accuracy and a macro F1
of 0.8070 on the additional 40 messages.

## Limitations

The original training messages were invented and may
not accurately represent real customer complaints.
Some messages are vague or contain multiple topics,
making classification difficult.

The model should assist support staff rather than
replace human judgment. Further testing with real
customer messages is needed before production use.