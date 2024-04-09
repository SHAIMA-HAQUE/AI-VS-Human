This project uses the dataset released by Kaggle - [LLM Detect AI-Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) and a dataset curated by [DAREK KŁECZEK](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)

Methods Tried:
- Transfer learning by using [DistilBertPreprocessor](https://huggingface.co/docs/transformers/model_doc/distilbert)
  We use a pre-trained model called DistilBert, which has already learned the structure and nuances of the English language from a large text dataset. We add a classification layer on top of the pre-trained BERT model and then fine-tune the entire model on our dataset of movie reviews. Fine-tuning means we continue the training process for a few more epochs with a smaller learning rate. This allows the BERT model to adjust its weights slightly to better understand the specific classification task while preserving the knowledge it has learned from the larger text dataset.
  This approach leverages the knowledge DistilBert has already learned from the large text dataset to achieve better performance on our task with less data, resources, and time.

  This approach required CPU and GPU T4 x2 on Kaggle to receive an F1 score ranging from 0.85-0.90.

- Trying a simple approach for text classification which is to convert text passages into vectors and then use standard ML algorithms such as logistic regression or tree-based models
  Instead of using Deep learning methods we can use statistical methods like tf-idf + machine learning algorithms

    **Logistic regression** model gave a ROC value
    Average ROC AUC: 0.9976
    Standard deviation: 0.0005
  
    On submitting this model to the competition the private score was 0.656879 and the public score was 0.825057
  
  
    **XGBoost**
  
    Average ROC AUC: 0.9983
    Standard deviation: 0.0005
  
    On submitting this model to the competition the private score was 0.654729 and the public score was 0.781105

    **Ensemble Learning with Logistic Regression, XGBoost, CatBoost Classifier**

    The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve        generalizability/robustness over a single estimator.
```
                  precision    recall  f1-score   support

           0       0.99      1.00      0.99      5959
           1       0.99      0.98      0.98      2882

    accuracy                           0.99      8841

   macro avg       0.99      0.99      0.99      8841

weighted avg       0.99      0.99      0.99      8841

    Accuracy: 0.9869781000612676
```
   On submitting this model to the competition the private score was 0.656998 and the public score was 0.821040
