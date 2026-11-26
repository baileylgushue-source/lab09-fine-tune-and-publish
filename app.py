import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="model",
    tokenizer="model",
    top_k=None,
)


def classify_message(text):
    if not text.strip():
        raise gr.Error("Please enter a customer support message.")

    results = classifier(text, truncation=True, max_length=128)[0]

    return {
        result["label"]: float(result["score"])
        for result in results
    }


demo = gr.Interface(
    fn=classify_message,
    inputs=gr.Textbox(
        lines=4,
        label="Customer Support Message",
        placeholder="Enter your message here...",
    ),
    outputs=gr.Label(
        num_top_classes=4,
        label="Predicted Support Queue",
    ),
    title="Lakeshore Ticket Sorter",
    description="Classify customer support messages into four support queues.",
)


if __name__ == "__main__":
    demo.launch()