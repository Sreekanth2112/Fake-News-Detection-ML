import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.markdown(
    """
    <h1 style='text-align:center; color:#ff4b4b;'>
    📰 Fake News Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("About Project")

st.sidebar.info(
    """
    This project uses Machine Learning and NLP
    to detect whether news is REAL or FAKE.
    """
)

# Description
st.write(
    "Enter a news article or headline below to check whether it is REAL or FAKE."
)

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # Remove newlines
    text = re.sub(r'\n', ' ', text)

    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove stopwords
    stop_words = stopwords.words('english')

    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    return text

# Text input box
news = st.text_area(
    "📝 Enter News Content",
    height=200
)

# Prediction button
if st.button("🔍 Check News"):

    # Empty input check
    if news.strip() == "":

        st.warning("Please enter some news text.")

    else:

        # Clean text
        cleaned_news = clean_text(news)

        # Convert text into vector
        vectorized_news = vectorizer.transform([cleaned_news])

        # Prediction
        prediction = model.predict(vectorized_news)

        # Probability score
        probability = model.predict_proba(vectorized_news)

        confidence_score = max(probability[0]) * 100

        # Result
        if prediction[0] == 1:

            st.balloons()

            st.success(
                "✅ This looks like REAL NEWS"
            )

        else:

            st.error(
                "🚨 Warning! This looks like FAKE NEWS"
            )

        # Confidence score
        st.info(
            f"Confidence Score: {confidence_score:.2f}%"
        )

# Footer
st.markdown("---")

st.write(
    "Developed by Sreekanth Chennoju"
)