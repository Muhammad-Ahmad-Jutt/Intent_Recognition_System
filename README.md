# Dataset Sources and Contributions

## Training Dataset

The intent recognition model is trained using the **Customer Support Intent Classification Dataset** available on Kaggle:

Dataset link:
https://www.kaggle.com/datasets/akshat14s/customer-support-intent-classification-dataset

This dataset provides labeled customer support queries across multiple intent categories and is used as the primary training dataset for fine-tuning the Hugging Face **DistilBERT** model for intent classification.

The dataset was used for:
- Fine-tuning the DistilBERT transformer model.
- Training the intent classification model.
- Evaluating model performance using validation metrics such as accuracy and loss.

Dataset source attribution:
- Dataset creator: Akshat14
- Platform: Kaggle
- Purpose in this research: Supervised training data for customer support intent recognition.

## External Evaluation Dataset

An additional synthetic evaluation dataset (`external_evaluation_dataset.json`) was generated using a Large Language Model (LLM) to test the model's generalization ability on unseen queries.

The synthetic dataset was not used during model training. It was only used for external evaluation to measure how well the trained model performs on new customer support queries.

Generation reference:
https://chatgpt.com/share/6a68bfa2-863c-83eb-bc3e-d98b72c94069