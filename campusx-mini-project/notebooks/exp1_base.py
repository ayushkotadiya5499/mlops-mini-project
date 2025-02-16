import mlflow
import pandas as pd
import mlflow.sklearn
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df = df.drop(columns=['tweet_id'])

# Text Preprocessing Functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# Apply Normalization
def normalize_text(df):
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

df = normalize_text(df)

# Filter Sentiments (Only Happiness & Sadness)
df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Convert labels to binary
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# Convert text into numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure MLflow and DAGsHub
mlflow.set_tracking_uri('https://dagshub.com/ayushkotadiya5499/mlops-mini-project.mlflow')

import dagshub
dagshub.init(repo_owner='ayushkotadiya5499', repo_name='mlops-mini-project', mlflow=True)

# Start MLflow experiment
mlflow.set_experiment("Logistic Regression Baseline")

with mlflow.start_run():
    # Log preprocessing parameters
    mlflow.log_param("vectorizer", "Bag of Words")
    mlflow.log_param("num_features", 1000)
    mlflow.log_param("test_size", 0.2)

    # Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Log Model Parameters
    mlflow.log_param("model", "Logistic Regression")

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log Model
    mlflow.sklearn.log_model(model, "model")

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
