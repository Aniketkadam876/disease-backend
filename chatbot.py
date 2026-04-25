import pandas as pd
import os
import re
from fuzzywuzzy import fuzz


class HealthChatBot:

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(BASE_DIR, "dataset", "disease_data.csv")
        self.data = pd.read_csv(DATA_PATH)

        # Pre-calculate unique symptoms
        all_syms = set()
        for col in self.data["symptoms"]:
            for s in str(col).split(","):
                all_syms.add(s.strip().lower())
        self.known_symptoms = sorted(list(all_syms), key=len, reverse=True)

        # ✅ Feature 4: Session-based memory (per user)
        self.sessions = {}  # { session_id: { symptoms, state } }

        # ✅ Feature 3: Synonym mapping
        self.synonyms = {
            "running nose": "runny nose",
            "throwing up": "vomiting",
            "can't sleep": "insomnia",
            "cannot sleep": "insomnia",
            "stomach pain": "abdominal pain",
            "tummy pain": "abdominal pain",
            "belly pain": "abdominal pain",
            "feeling cold": "chills",
            "feeling hot": "fever",
            "high temperature": "fever",
            "sore muscles": "muscle pain",
            "body ache": "body pain",
            "chest tightness": "chest pain",
            "shortness of breath": "breathlessness",
            "hard to breathe": "breathlessness",
            "throwing up": "vomiting",
            "puking": "vomiting",
            "loose motion": "diarrhea",
            "loose stools": "diarrhea",
            "watery eyes": "eye discharge",
            "itchy eyes": "eye irritation",
            "tired": "fatigue",
            "exhausted": "fatigue",
            "no energy": "fatigue",
            "spinning head": "dizziness",
            "feeling dizzy": "dizziness",
        }

    # -------------------------
    # Session Management
    # -------------------------
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "symptoms": [],
                "awaiting_confirm": None,  # symptom being confirmed via yes/no
            }
        return self.sessions[session_id]

    def reset_session(self, session_id):
        self.sessions[session_id] = {
            "symptoms": [],
            "awaiting_confirm": None,
        }

    # -------------------------
    # Response Formatter
    # -------------------------
    def format_response(self, message, disease="", risk="", cures="", doctor="", top3=None, confidence=0):
        return {
            "response": message,
            "disease": disease,
            "risk": risk,
            "cures": cures,
            "doctor": doctor,
            "top3": top3 or [],        # ✅ Feature 2: Top 3 diseases
            "confidence": confidence,  # ✅ Feature 5: Confidence score
        }

    # -------------------------
    # Main Response Handler
    # -------------------------
    def get_response(self, message, session_id="default"):
        # ✅ Feature 4: Use session memory
        session = self.get_session(session_id)
        msg = message.lower()

        # ✅ Feature 3: Normalize synonyms before processing
        msg = self.normalize_message(msg)

        # Greeting
        if any(re.search(rf"\b{word}\b", msg) for word in ["hi", "hello", "hey"]):
            return self.format_response("Hello! 👋 I'm your health assistant. Tell me your symptoms.")

        # Reset
        if any(re.search(rf"\b{word}\b", msg) for word in ["reset", "clear"]):
            self.reset_session(session_id)
            return self.format_response("Symptoms cleared. Please tell me your symptoms again.")

        # Help
        if "help" in msg:
            return self.format_response("Tell me symptoms like: fever, cough, headache, fatigue.")

        # ✅ Handle yes/no follow-up confirmation
        if session["awaiting_confirm"]:
            return self.handle_confirmation(msg, session, session_id)

        # Handle "no more symptoms"
        no_more_phrases = ["no", "nope", "done", "that's all", "that is all", "nothing", "no more", "nothing else"]
        if any(re.search(rf"\b{re.escape(phrase)}\b", msg) for phrase in no_more_phrases):
            if session["symptoms"]:
                return self.predict_disease(session, session_id)
            return self.format_response("Please tell me your symptoms first.")

        # ✅ Feature 1: Extract symptoms with fuzzy + synonym matching
        symptoms = self.extract_symptoms(msg)
        if symptoms:
            new_found = False
            for s in symptoms:
                if s not in session["symptoms"]:
                    session["symptoms"].append(s)
                    new_found = True

            if not new_found and len(session["symptoms"]) >= 2:
                return self.predict_disease(session, session_id)

            return self.ask_more(session, session_id)

        return self.format_response("Please describe your symptoms so I can help you.")

    # -------------------------
    # Synonym Normalization (Feature 3)
    # -------------------------
    def normalize_message(self, msg):
        for phrase, standard in self.synonyms.items():
            msg = re.sub(rf"\b{re.escape(phrase)}\b", standard, msg)
        return msg

    # -------------------------
    # Extract Symptoms (Feature 1: Fuzzy Matching)
    # -------------------------
    def extract_symptoms(self, msg):
        found = []

        # First pass: exact regex match (most reliable)
        for s in self.known_symptoms:
            if re.search(rf"\b{re.escape(s)}\b", msg):
                if s not in found:
                    found.append(s)

        # Second pass: fuzzy match on individual words for anything not yet matched
        msg_words = msg.split()
        for known in self.known_symptoms:
            if known in found:
                continue
            for word in msg_words:
                if len(word) > 3 and fuzz.ratio(word, known) > 82:
                    if known not in found:
                        found.append(known)
                    break

        return found

    # -------------------------
    # Ask More / Targeted Follow-up (Feature 3)
    # -------------------------
    def ask_more(self, session, session_id):
        symptoms = session["symptoms"]

        if len(symptoms) < 2:
            return self.format_response(
                f"I noted: {', '.join(symptoms)}. Do you have any other symptoms?"
            )

        # ✅ Feature 3: Ask targeted follow-up based on top candidate
        top_disease, top_symptoms = self.get_top_candidate(symptoms)
        missing = [s for s in top_symptoms if s not in symptoms]

        if missing:
            follow_up_symptom = missing[0]
            session["awaiting_confirm"] = follow_up_symptom
            return self.format_response(
                f"I noted: {', '.join(symptoms)}. "
                f"Do you also have **{follow_up_symptom}**? (yes / no)"
            )

        return self.predict_disease(session, session_id)

    # -------------------------
    # Handle yes/no confirmation
    # -------------------------
    def handle_confirmation(self, msg, session, session_id):
        symptom = session["awaiting_confirm"]
        session["awaiting_confirm"] = None

        if any(re.search(rf"\b{word}\b", msg) for word in ["yes", "yeah", "yep", "yup", "sure", "correct"]):
            if symptom not in session["symptoms"]:
                session["symptoms"].append(symptom)

        return self.predict_disease(session, session_id)

    # -------------------------
    # Get Top Candidate Disease
    # -------------------------
    def get_top_candidate(self, user_symptoms):
        best_row = None
        best_score = 0
        for _, row in self.data.iterrows():
            disease_symptoms = [s.strip().lower() for s in str(row["symptoms"]).split(",")]
            score = len(set(user_symptoms) & set(disease_symptoms)) / len(disease_symptoms)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            return "", []
        return best_row["disease"], [s.strip().lower() for s in str(best_row["symptoms"]).split(",")]

    # -------------------------
    # Predict Disease (Features 2, 5, 6)
    # -------------------------
    def predict_disease(self, session, session_id):
        user_symptoms = session["symptoms"]
        scores = []

        for _, row in self.data.iterrows():
            disease_symptoms = [s.strip().lower() for s in str(row["symptoms"]).split(",")]

            matched = set(user_symptoms) & set(disease_symptoms)
            unmatched = set(disease_symptoms) - set(user_symptoms)
            extra = set(user_symptoms) - set(disease_symptoms)

            # ✅ Feature 6: Weighted scoring
            score = (
                len(matched) * 2 -
                len(unmatched) * 0.5 -
                len(extra) * 0.3
            ) / max(len(disease_symptoms), 1)

            scores.append((score, row))

        scores.sort(key=lambda x: x[0], reverse=True)
        top3 = scores[:3]

        if top3[0][0] <= 0:
            return self.format_response(
                "I need more information. Please describe your symptoms in more detail."
            )

        # ✅ Feature 2: Top 3 results
        top3_list = []
        for score, row in top3:
            raw_confidence = len(set(user_symptoms) & set(
                [s.strip().lower() for s in str(row["symptoms"]).split(",")]
            )) / len([s.strip().lower() for s in str(row["symptoms"]).split(",")])
            confidence_pct = min(int(raw_confidence * 100), 99)
            top3_list.append({
                "disease": row["disease"],
                "confidence": confidence_pct,
                "risk": row["risk level"],
            })

        best = top3[0][1]
        best_confidence = top3_list[0]["confidence"]

        # Reset session after prediction
        self.reset_session(session_id)

        return self.format_response(
            f"Based on your symptoms, you most likely have **{best['disease']}**. See details below.",
            disease=best["disease"],
            risk=best["risk level"],
            cures=best["cures"],
            doctor=best["doctor"],
            top3=top3_list,
            confidence=best_confidence,
        )
