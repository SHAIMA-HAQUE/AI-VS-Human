This project uses the dataset released by Kaggle - [LLM Detect AI-Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) and a dataset curated by [DAREK KŁECZEK](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)

Methods Tried:
- Transfer learning by using [DistilBertPreprocessor](https://huggingface.co/docs/transformers/model_doc/distilbert)
  We use a pre-trained model called DistilBert, which has already learned the structure and nuances of the English language from a large text dataset. We add a classification layer on top of the pre-trained BERT model and then fine-tune the entire model on our dataset of movie reviews. Fine-tuning means we continue the training process for a few more epochs with a smaller learning rate. This allows the BERT model to adjust its weights slightly to better understand the specific classification task while preserving the knowledge it has learned from the larger text dataset.
  This approach leverages the knowledge DistilBert has already learned from the large text dataset to achieve better performance on our task with less data, resources, and time.

  This approach required CPU and GPU T4 x2 on Kaggle to receive an F1 score ranging from 0.85-0.90.

- Trying a simple approach for text classification which is to convert text passages into vectors and then use standard ML algorithms such as logistic regression or tree-based models
  Instead of using Deep learning methods we can use statistical methods like tf-idf + machine learning algorithms

  Logistic regresssion model gave a ROC value
  Average ROC AUC: 0.9976
  Standard deviation: 0.0005

  On submitting this model to the competition the private score was 0.65687 and the public score was 0.825057
