import os
import pandas as pd
import joblib
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#CUSTOM TRANSFORMERS

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.func)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame()
        df['CONTAINS_URL'] = X.str.contains(r'http[s]?://\S+', regex=True).astype(int)
        df['COMMENT_LENGTH'] = X.apply(len)
        df['NUMBER_SPECIAL_CHARACTERS'] = X.apply(lambda x: sum([1 for char in x if char in '$%&@!']))
        return df

#CUSTOM TEXT CLEANING FUNCTION

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming_remove_url(text):
    text = re.sub(r"http[s]?://\S+", "", text)  # remove URLs
    words = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in words if word.lower() not in stop_words])

#LOAD DATA

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "Youtube02-KatyPerry.csv")
df = pd.read_csv(csv_path)
df = df[['CONTENT', 'CLASS']]  # keep only relevant columns

# Shuffle
df = df.sample(frac=1, random_state=62).reset_index(drop=True)

# Train/test split (75/25)
split_index = int(len(df) * 0.75)
X_train = df['CONTENT'][:split_index]
y_train = df['CLASS'][:split_index]
X_test = df['CONTENT'][split_index:]
y_test = df['CLASS'][split_index:]

#BUILD PIPELINE

text_pipeline = Pipeline([
    ('stem', TextPreprocessor(stemming_remove_url)),
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

numeric_pipeline = Pipeline([
    ('features', FeatureEngineer()),
    ('scale', MinMaxScaler())
])

combined_features = FeatureUnion([
    ('text', text_pipeline),
    ('numeric', numeric_pipeline)
])

model_pipeline = Pipeline([
    ('features', combined_features),
    ('clf', MultinomialNB())
])

#TRAIN & EVALUATE

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#SAVE MODEL

joblib.dump(model_pipeline, 'spam_detector_pipeline.joblib')
print("\nModel saved to 'spam_detector_pipeline.joblib'")
