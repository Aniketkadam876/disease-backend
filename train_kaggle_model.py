import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Load data
train = pd.read_csv("dataset/training_data.csv")
test = pd.read_csv("dataset/test_data.csv")
# Remove unwanted columns
train = train.loc[:, ~train.columns.str.contains("^Unnamed")]
test = test.loc[:, ~test.columns.str.contains("^Unnamed")]

# Separate X & y
X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]

X_test = test.drop("prognosis", axis=1)
y_test = test["prognosis"]


# Encode labels
encoder = LabelEncoder()

y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)


# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=42
)

model.fit(X_train, y_train_enc)


# Evaluate
pred = model.predict(X_test)

acc = accuracy_score(y_test_enc, pred)

print("Model Accuracy:", acc)


# Save
joblib.dump(model, "kaggle_model.pkl")
joblib.dump(encoder, "kaggle_encoder.pkl")
joblib.dump(X_train.columns.tolist(), "symptom_list.pkl")

print("✅ Kaggle Model Saved")
