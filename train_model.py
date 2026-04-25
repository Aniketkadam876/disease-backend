import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# Load dataset
data = pd.read_csv('../dataset/disease_data.csv')

# Keep only needed columns
data = data[['disease', 'symptoms']]

# Drop empty rows
data.dropna(inplace=True)

# Features & Target
X = data['symptoms']
y = data['disease']

# Convert disease names to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save files
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model Trained Successfully!")
