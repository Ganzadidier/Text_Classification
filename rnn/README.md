# RNN Text Classification (SimpleRNN)

## Analysis & Conclusion

### Performance Summary
We evaluated a SimpleRNN architecture across three different embedding strategies. The comparative results on the test set are summarized below:

| Model Variant | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RNN + CBOW** | **0.70** | **0.34** | **0.74** | **0.47** | **0.73** |
| **RNN + Skip-gram** | 0.45 | 0.23 | 0.86 | 0.36 | 0.62 |
| **RNN + Standard** | 0.26 | 0.19 | 0.98 | 0.32 | 0.54 |

### Comparative Discussion
**1. The Winner: RNN + CBOW**
The **Continuous Bag of Words (CBOW)** embedding strategy provided the most balanced performance (F1: 0.47). It successfully distinguished hate speech from normal text without collapsing into single-class predictions.

**2. The Instability of SimpleRNN**
Skip-gram and Standard embeddings showed "trigger-happy" behavior, flagging nearly everything as hate speech to achieve high recall but terrible precision. This highlights the vanishing gradient limitations of SimpleRNN compared to LSTMs.
