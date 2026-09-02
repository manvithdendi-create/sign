import unittest
import numpy as np
from fastapi import HTTPException
from backend.landmark_extractor import normalize_landmarks
from backend.model import SignLanguageModel
from backend.server import validate_landmarks, LandmarkPoint
from backend.translator import TranslatorEngine

class TestSignLanguageAI(unittest.TestCase):
    def test_landmark_normalization(self):
        # 21 mock points
        dummy_landmarks = [{'x': float(i*0.01), 'y': float(i*0.02), 'z': 0.0} for i in range(21)]
        features = normalize_landmarks(dummy_landmarks)
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 63)

    def test_model_training_and_prediction(self):
        model = SignLanguageModel()
        model.train_initial_model()
        self.assertTrue(model.is_trained)
        self.assertGreater(len(model.classes), 5)

        # Test prediction on dummy input
        dummy_landmarks = [{'x': float(i*0.02), 'y': float(i*0.01), 'z': 0.0} for i in range(21)]
        result = model.predict(dummy_landmarks)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('top_candidates', result)
        print(f"\n[Test Result] Prediction: {result['prediction']}, Confidence: {result['confidence']}")

    def test_translator_requires_consecutive_stable_frames(self):
        translator = TranslatorEngine()

        translator.process_prediction("A", 0.9)
        translator.process_prediction("A", 0.9)
        self.assertEqual(translator.process_prediction("A", 0.9)["current_word"], "A")

        translator.process_prediction("A", 0.4)
        translator.process_prediction("A", 0.9)
        translator.process_prediction("A", 0.9)
        result = translator.process_prediction("A", 0.9)

        self.assertEqual(result["status"], "added")
        self.assertEqual(result["current_word"], "AA")

    def test_landmark_validation_requires_exact_finite_points(self):
        valid_points = [LandmarkPoint(x=0.0, y=0.0, z=0.0) for _ in range(21)]
        validate_landmarks(valid_points)

        with self.assertRaises(HTTPException):
            validate_landmarks(valid_points[:-1])

        invalid_points = valid_points.copy()
        invalid_points[0] = LandmarkPoint(x=float("nan"), y=0.0, z=0.0)
        with self.assertRaises(HTTPException):
            validate_landmarks(invalid_points)

    def test_translator_reset_does_not_clear_sentence(self):
        translator = TranslatorEngine()
        for _ in range(3):
            translator.process_prediction("A", 0.9)

        translator.reset_stability()

        self.assertEqual(translator.buffer, ["A"])
        self.assertEqual(translator.process_prediction("A", 0.9)["status"], "holding")

if __name__ == '__main__':
    unittest.main()
