import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# Load data
data = pd.read_csv("../dataset/disease_data.csv")

data = data[
    ["disease", "symptoms", "cures", "doctor", "risk level"]
]

# Combine text
data["text"] = (
    data["symptoms"] + " " +
    data["cures"] + " " +
    data["doctor"] + " " +
    data["risk level"].astype(str)
)

# Train vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(data["text"])

# Save everything
joblib.dump(X, "tfidf_matrix.pkl")
joblib.dump(vectorizer, "vectorizer_v3.pkl")
joblib.dump(data, "disease_data.pkl")

print("✅ Similarity Model Ready")
