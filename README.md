# PBEL-Project
AI-BASED EMOTION RECOGNITION FROM TEXT

# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Dataset (Emotion-Labeled Text) – expanded for demo
data = {
    "text": [
        "I am so happy today!",
        "Feeling lonely and sad...",
        "Why don’t you listen!",
        "This is scary.",
        "I’m excited for tomorrow!",
        "I feel frustrated with this work.",
        "I am worried about my exams.",
        "That joke made me laugh!",
        "I am angry at the delay.",
        "I feel calm and relaxed."
    ],
    "emotion": [
        "Joy", "Sadness", "Anger", "Fear", "Joy",
        "Anger", "Fear", "Joy", "Anger", "Calm"
    ]
}
df = pd.DataFrame(data)

# 3. Preprocessing + Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["text"])
y = df["emotion"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5. Logistic Regression Model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))

# 6. Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))

# 7. Training + Evaluation: Confusion Matrix (Naive Bayes example)
cm = confusion_matrix(y_test, y_pred_nb, labels=nb.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()
