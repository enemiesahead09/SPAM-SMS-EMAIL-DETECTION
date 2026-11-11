SpamGuard: SMS and Email Spam Detection

SpamGuard is a Streamlit dashboard that classifies text messages as **spam** or **ham** (legitimate). It uses natural language processing and a Multinomial Naive Bayes classifier, and includes interactive views for exploring the dataset and model performance.

> The included dataset contains short SMS messages. Results on long emails, new spam tactics, or phishing messages may differ.

## Features

- Classify a message and view its predicted class and spam probability.
- Inspect the text preprocessing steps used by the classifier.
- Explore message counts, class balance, message lengths, and common words.
- Review evaluation metrics, a confusion matrix, and ROC analysis.
- Compare several classifiers on the same train/test split.

## Run the dashboard

Use Python 3.9 or newer. From the project directory, install the dashboard dependencies:

```bash
python -m pip install streamlit pandas numpy plotly nltk scikit-learn
```

Then start the app:

```bash
streamlit run app.py
```

The dashboard expects `model.pkl`, `vectorizer.pkl`, and `spam.csv` beside `app.py`. On its first run, it may download the required English NLTK tokenizer and stop-word data.

## Train the standalone model

The project also includes a separate training script:

```bash
python -m pip install pandas scikit-learn joblib
python train_model.py
```

This script reads `spam.csv` and saves `spam_pipeline.joblib` and `model_metrics.json`. These outputs belong to the standalone training workflow; the current Streamlit dashboard instead loads `model.pkl` and `vectorizer.pkl`.

## How it works

The dashboard preprocesses text by lowercasing, tokenizing, removing non-alphanumeric tokens and English stop words, and applying Porter stemming. It transforms the result into TF-IDF features and classifies it with Multinomial Naive Bayes. Evaluation and comparison views use the SMS Spam Collection data included as `spam.csv`.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and message analyzer |
| `sms-spam-detection.ipynb` | Notebook for exploration, preprocessing, and model work |
| `spam.csv` | SMS message dataset used by the dashboard and training script |
| `model.pkl`, `vectorizer.pkl` | Model and TF-IDF vectorizer loaded by the dashboard |
| `train_model.py` | Standalone model-training script |
| `spam_pipeline.joblib`, `model_metrics.json` | Outputs from the standalone training script |
| `requirements.txt` | Python package list |

## Dataset and limitations

The project uses the SMS Spam Collection dataset. The classes are imbalanced, with legitimate messages making up most of the data. A high overall accuracy can therefore hide errors on spam messages; review precision, recall, and the confusion matrix as well. This project is for educational use and should not be treated as a production email security filter.
