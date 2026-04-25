from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from db import get_connection
import hashlib
from chatbot import HealthChatBot
import numpy as np


# Initialize app
app = Flask(__name__)
CORS(app)


# Load ML Models
model = joblib.load("kaggle_model.pkl")
encoder = joblib.load("kaggle_encoder.pkl")
symptom_list = joblib.load("symptom_list.pkl")




bot = HealthChatBot()

model_diseases = set(encoder.classes_)
csv_diseases = set(bot.data["disease"])

missing = model_diseases - csv_diseases

print("Missing diseases:", missing)
# -----------------------------
# Helper: Convert Symptoms → Vector
# -----------------------------
def symptoms_to_vector(symptoms):

    user_syms = [s.strip().lower() for s in symptoms.split(",")]

    vector = []

    for s in symptom_list:
        s_lower = s.lower()

        # 🔥 Partial + flexible match
        found = any(
            us in s_lower or s_lower in us
            for us in user_syms
        )

        vector.append(1 if found else 0)

    return np.array(vector).reshape(1, -1)


# -----------------------------
# Home
# -----------------------------
@app.route('/')
def home():
    return jsonify({"message": "MediPredict AI API Running"})


# -----------------------------
# Register
# -----------------------------
@app.route('/register', methods=['POST'])
def register():

    data = request.get_json(force=True)

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    hashed = hashlib.sha256(password.encode()).hexdigest()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name,email,password) VALUES (%s,%s,%s)",
            (name, email, hashed)
        )
        conn.commit()

        return jsonify({"message": "User Registered Successfully"})

    except:
        return jsonify({"error": "Email already exists"}), 400

    finally:
        cursor.close()
        conn.close()


# -----------------------------
# Login
# -----------------------------
@app.route('/login', methods=['POST'])
def login():

    data = request.get_json(force=True)

    email = data.get("email")
    password = data.get("password")

    hashed = hashlib.sha256(password.encode()).hexdigest()

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM users WHERE email=%s AND password=%s",
        (email, hashed)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        return jsonify({
            "message": "Login Success",
            "user_id": user["id"],
            "name": user["name"]
        })
    else:
        return jsonify({"error": "Invalid credentials"}), 401


# -----------------------------
# ChatBot
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(force=True)

    message = data.get("message")
    print(message)

    if not message:
        return jsonify({"error": "Message required"}), 400

    response = bot.get_response(message)

    return jsonify(response)


# -----------------------------
# Predict (REAL ML)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(force=True)

    symptoms = data.get("symptoms")
    user_id = data.get("user_id")

    if not symptoms or len(symptoms.split(",")) < 2:
        return jsonify({
            "message":
            "Please enter at least 2-3 symptoms for accurate prediction."
        })

    # Convert to ML input
    X = symptoms_to_vector(symptoms)

    # Predict probabilities
    probs = model.predict_proba(X)[0]

    # Get Top 3
    top_idx = probs.argsort()[-3:][::-1]

    results = []

    for i in top_idx:
        disease_name = encoder.inverse_transform([i])[0]

        # Look up metadata in the chatbot's dataset
        match = bot.data[bot.data["disease"] == disease_name]

        if not match.empty:
            info = match.iloc[0]
            risk = info.get("risk level", "Unknown")
            cures = info.get("cures", "Unknown")
            doctor = info.get("doctor", "Unknown")
        else:
            risk, cures, doctor = "Unknown", "Consult a professional", "General Physician"

        results.append({
            "disease": disease_name,
            "confidence": round(probs[i] * 100, 2),
            "risk": risk,
            "cures": cures,
            "doctor": doctor
        })


    # Save best
    best = results[0]["disease"]
    print(results)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions (user_id,symptoms,disease) VALUES (%s,%s,%s)",
        (user_id, symptoms, best)
    )

    conn.commit()

    cursor.close()
    conn.close()


    return jsonify({
        "predictions": results
    })


# -----------------------------
# History
# -----------------------------
@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT symptoms, disease, created_at
        FROM predictions
        WHERE user_id=%s
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    history = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(history)


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
