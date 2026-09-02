import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from backend.landmark_extractor import normalize_landmarks
from backend.dataset import generate_synthetic_dataset

class SignLanguageModel:
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.custom_samples_X = []
        self.custom_samples_y = []
        self.is_trained = False
        self.classes = []

    def train_initial_model(self):
        """Train classifier on base ASL benchmark synthetic dataset."""
        print("[AI Engine] Generating benchmark ASL dataset...")
        X, y = generate_synthetic_dataset(num_samples_per_class=100)
        
        self.classifier.fit(X, y)
        self.is_trained = True
        self.classes = list(np.unique(y))
        print(f"[AI Engine] Model successfully trained on {len(X)} samples across {len(self.classes)} sign classes.")

    def predict(self, landmarks):
        """
        Classifies hand landmarks.
        Returns dict: {'prediction': label, 'confidence': float, 'top_candidates': [...]}
        """
        if not self.is_trained:
            self.train_initial_model()

        features = normalize_landmarks(landmarks)
        if features is None:
            return {"prediction": "UNKNOWN", "confidence": 0.0, "top_candidates": []}

        features = features.reshape(1, -1)
        
        # The training data is made from gesture templates, so nearest-neighbor
        # matching preserves the shape similarity that the forest was losing.
        probs = self.classifier.predict_proba(features)[0]
        top_indices = np.argsort(probs)[::-1][:3]
        
        classes = self.classifier.classes_
        prediction = classes[top_indices[0]]
        confidence = float(probs[top_indices[0]])

        top_candidates = [
            {"label": str(classes[idx]), "confidence": float(probs[idx])}
            for idx in top_indices if probs[idx] > 0.05
        ]

        return {
            "prediction": str(prediction),
            "confidence": round(confidence, 4),
            "top_candidates": top_candidates
        }

    def add_custom_gesture(self, label, landmarks_list):
        """
        Dynamically retrains the model with new user-provided custom gesture landmarks.
        landmarks_list: List of 21-point landmark arrays
        """
        added_count = 0
        for lm in landmarks_list:
            feats = normalize_landmarks(lm)
            if feats is not None:
                self.custom_samples_X.append(feats)
                self.custom_samples_y.append(label.upper())
                added_count += 1

        if added_count > 0:
            # Re-train with base synthetic dataset + custom samples
            X_base, y_base = generate_synthetic_dataset(num_samples_per_class=60)
            X_combined = np.vstack([X_base, np.array(self.custom_samples_X)])
            y_combined = np.concatenate([y_base, np.array(self.custom_samples_y)])

            self.classifier.fit(X_combined, y_combined)
            self.classes = list(np.unique(y_combined))
            print(f"[AI Engine] Dynamic retrain complete! Added {added_count} samples for '{label}'. Total classes: {len(self.classes)}")
            return True, f"Successfully recorded '{label}' with {added_count} samples and updated AI model."

        return False, "Failed to extract valid landmark features from provided samples."

# Global singleton instance
sign_model = SignLanguageModel()
