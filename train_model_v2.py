import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("../dataset/disease_data.csv")

# Keep columns
data = data[
    ["disease", "symptoms", "cures", "doctor", "risk level"]
]

# Combine features into one text
data["full_text"] = (
    data["symptoms"] + " " +
    data["cures"] + " " +
    data["doctor"] + " " +
    data["risk level"].astype(str)
)

X = data["full_text"]
y = data["disease"]

# Encode labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english",
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_enc, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Model Accuracy:", acc)

# Save
joblib.dump(model, "model_v2.pkl")
joblib.dump(vectorizer, "vectorizer_v2.pkl")
joblib.dump(encoder, "encoder_v2.pkl")

print("✅ New Model Trained & Saved")
