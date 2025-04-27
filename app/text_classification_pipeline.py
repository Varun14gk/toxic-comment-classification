# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:08:47 2024

@author: AB937LH
"""
# importing the libraries for text_classification_pipline
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import pickle
import contractions

# Download necessary NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the WordNetLemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def remove_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def to_lower_case(self, text):
        return text.lower()

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_text)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

    def remove_extra_spaces(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def lemmatize_text(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def preprocess(self, text):
        text = self.remove_html_tags(text)
        text = self.remove_punctuation(text)
        text = self.to_lower_case(text)
        text = self.remove_numbers(text)
        text = self.expand_contractions(text)
        text = self.remove_stopwords(text)
        text = self.clean_text(text)
        text = self.remove_extra_spaces(text)
        text = self.lemmatize_text(text)  # Lemmatization step added here
        return text

# Load the data
data = pd.read_csv('data.csv')

# Create an instance of the TextPreprocessor class
preprocessor = TextPreprocessor()

# Apply the preprocessing to the 'comment_text' column
data['comment_text'] = data['comment_text'].apply(preprocessor.preprocess)

# Drop rows with null values or blank strings in 'comment_text' column
data = data.dropna(subset=['comment_text'])
data = data[data['comment_text'].str.strip() != '']

# Function to plot word cloud for a given series of text
def plot_word_cloud(series, title):
    
    all_text = ' '.join(series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Plot word cloud for general comment_text
plot_word_cloud(data['comment_text'], 'Word Cloud for All Comments')

# Plot word cloud for each category
columns_to_analyze = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for column in columns_to_analyze:
    filtered_comments = data[data[column] == 1]['comment_text']
    plot_word_cloud(filtered_comments, f'Word Cloud for {column.title()} Comments')

# Function to plot top 10 words for a given series of text
def plot_top_words(series, title, color):
    # Tokenize the comments and count the frequency of each word
    all_words = ' '.join(series).split()
    word_freq = Counter(all_words)

    # Get the top 10 most common words and their counts
    top_words = word_freq.most_common(10)
    words, freqs = zip(*top_words)

    # Create a bar chart with Matplotlib
    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs, color=color)
    plt.title(title)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# List of columns to analyze and their associated colors
columns_to_analyze = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

# Apply the function to each column with the associated color
for column, color in zip(columns_to_analyze, colors):
    filtered_comments = data[data[column] == 1]['comment_text']
    plot_top_words(filtered_comments, f'Top 10 Words in {column.replace("_", " ").title()} Comments', color)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Vectorize the text using the TF-IDF vectorizer
X_tfidf = tfidf_vectorizer.fit_transform(data['comment_text'])

# List of y variables
y_variables = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Split the dataset into train (80%) and test (20%)
X_train_test, X_test, y_train_test, y_test = train_test_split(
    X_tfidf, data[y_variables], test_size=0.2, random_state=42
)

# Further split the train data into train (65% of original data) and validation (15% of original data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_test, y_train_test, test_size=0.1875, random_state=42  # 0.1875 is 15% of 80%
)

# Define the number of splits for StratifiedKFold
n_splits = 5  # Updated to 5 splits
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Train and evaluate the Logistic Regression model with TF-IDF vectorizer for each label
models = {}
for label in y_variables:
    print(f"Evaluating label: {label}")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    models[label] = lr

    # Perform stratified K-fold cross-validation
    auc_scores = []
    for train_index, val_index in skf.split(X_train, y_train[label]):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[label].iloc[train_index], y_train[label].iloc[val_index]

        lr.fit(X_fold_train, y_fold_train)
        y_fold_pred_prob = lr.predict_proba(X_fold_val)[:, 1]
        auc_scores.append(roc_auc_score(y_fold_val, y_fold_pred_prob))

    # Average AUC score across folds
    avg_auc_score = np.mean(auc_scores)
    print(f"Average AUC score across folds: {avg_auc_score:.4f}")

    # Evaluate on the test set
    y_test_pred = lr.predict(X_test)
    y_test_pred_prob = lr.predict_proba(X_test)[:, 1]
    test_auc_score = roc_auc_score(y_test[label], y_test_pred_prob)

    # Print classification report
    print("Classification report:")
    print(classification_report(y_test[label], y_test_pred))
    
    # Plot confusion matrix 
    ConfusionMatrixDisplay.from_predictions(y_test[label], y_test_pred, display_labels=[f'Not {label}', label]) 
    plt.title(f'Confusion Matrix for {label}') 
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test[label], y_test_pred_prob)
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {test_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label}')
    plt.legend(loc='lower right')
    plt.show()

# Save the models and vectorizer to a pickle file
with open('text_classification_models.pkl', 'wb') as file:
    pickle.dump({'models': models, 'vectorizer': tfidf_vectorizer}, file)

print("Models and vectorizer saved to 'text_classification_models.pkl'")