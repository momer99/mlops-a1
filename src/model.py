import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset (IMDB reviews)
df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k/master/train.csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

df['review'] = df['review'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())  # Naïve Bayes classifier
])

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "src/sentiment_model.pkl")
print("Model saved successfully!!")
