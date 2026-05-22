# 📰 Fake News Detection Using Machine Learning

This project detects whether a news article is REAL or FAKE using Machine Learning and Natural Language Processing (NLP).

The system is built using Python, Scikit-learn, TF-IDF Vectorization, Logistic Regression, and Streamlit.

---

# 🚀 Features

✅ Fake News Detection  
✅ Real News Detection  
✅ Confidence Score  
✅ Beautiful Streamlit Web Application  
✅ NLP Text Cleaning  
✅ Machine Learning Prediction System  

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- Streamlit
- TF-IDF Vectorizer
- Logistic Regression

---

# 📂 Project Structure

```text
Fake_News_Detection/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

Dataset used:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Download Dataset

1. Download ZIP file from Kaggle
2. Extract ZIP
3. Copy these files into project folder:

```text
Fake.csv
True.csv
```

---

# ⚙️ Installation

## Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fake-News-Detection-ML.git
```

## Step 2 — Open Project Folder

```bash
cd Fake-News-Detection-ML
```

## Step 3 — Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python train_model.py
```

This will create:

```text
model.pkl
vectorizer.pkl
```

---

# ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Model
   ↓
Prediction
   ↓
Streamlit Web App
```

---

# 🖥️ Project Output

The system predicts:

- ✅ REAL NEWS
- 🚨 FAKE NEWS

along with a confidence score.

---

# 📸 Example

## REAL NEWS Example

```text
The Indian Space Research Organisation successfully launched a communication satellite to improve weather forecasting and internet services in rural areas.
```

## FAKE NEWS Example

```text
Aliens secretly landed in Hyderabad and gifted invisible flying bikes to students.
```

---

# 📈 Future Improvements

- Deep Learning (LSTM/BERT)
- Live News API Integration
- Multilingual Support
- Voice Input
- Online Deployment
- News Category Detection

---

# 👨‍💻 Author

Sreekanth Chennoju

Artificial Intelligence and Machine Learning Student

---

# ⭐ GitHub

If you like this project, give it a ⭐ on GitHub.
