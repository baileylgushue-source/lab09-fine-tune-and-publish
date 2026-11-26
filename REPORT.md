# Lakeshore Ticket Sorter — Findings and Recommendations

## Overall Results

When using 300 training examples, the TF-IDF model had an average validation accuracy of 86.67%, while the fully fine-tuned DistilBERT model reached 91.39%. When tested on new messages, TF-IDF achieved 82.92%, while DistilBERT achieved 88.75%.

## Where Fine-Tuning Helped

Looking at the learning curve, I noticed that full fine-tuning became more effective as more training examples were added. With only 50 examples, TF-IDF actually performed slightly better. At 100 examples, full fine-tuning started to perform better, although the results varied between runs. The biggest improvement was seen at 300 examples, where full fine-tuning consistently performed better than TF-IDF. The head-only model performed worse than TF-IDF at every training size.

## Possible Reasons

I think one reason for these results is that DistilBERT needs more examples to understand the differences between customer complaints. I checked this by comparing the results at 50, 100 and 300 examples.

Another possible reason is that the head-only model was not given enough time to learn because most of the model was frozen during training. However, I did not test different learning rates or training times, so I cannot say for certain that this was the reason.

## My Own 40 Messages

When testing the 40 messages I created myself, DistilBERT achieved 82.50% accuracy, while TF-IDF achieved 77.50%. Both models performed worse than they did on the original test data. This suggests that the generated messages may have been easier to classify than messages written differently.

## Recommendation

I would recommend using the fully fine-tuned DistilBERT model to help customer support staff organize incoming complaints. It performed the best overall and could help staff save time. However, I would still recommend having employees review the results rather than relying entirely on the model. If the model performed poorly on real customer complaints, I would reconsider using it.

## Limitations

One major limitation is that the training messages were invented rather than collected from actual customers. My own 40 messages also represent a small sample. Although the results are promising, more testing with real customer complaints would be needed before the company could confidently use the model.