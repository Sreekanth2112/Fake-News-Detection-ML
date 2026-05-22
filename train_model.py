import pandas as pd
import re
import string
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine title + text
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

# Combine datasets
data = pd.concat([fake, true])

# Shuffle
data = data.sample(frac=1, random_state=42)

# Keep required columns
data = data[["content", "label"]]

# Text cleaning
def clean_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'\w*\d\w*', '', text)

    stop_words = stopwords.words('english')

    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    return text

# Clean data
data["content"] = data["content"].apply(clean_text)

# Input and output
x = data["content"]
y = data["label"]

# Better vectorizer
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2)
)

x_vectorized = vectorizer.fit_transform(x)

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Better model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")