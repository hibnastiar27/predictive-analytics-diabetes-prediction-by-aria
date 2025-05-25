"""
Script untuk prediksi diabetes menggunakan model XGBoost yang telah ditraining.
Input dari user akan diproses sesuai dengan encoding yang telah dilakukan.

Dibuat oleh : Nur Aria Hibnastiar 
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Path ke model
MODEL_PATH = Path(__file__).parent / "xgb_best_model.pkl"

# Kolom one-hot yang digunakan saat training
SMOKING_CATEGORIES = [
  "No Info", "current", "ever", "former", "never", "not current"
]

def encode_inputs(gender, age, hypertension, heart_disease, bmi, hba1c, glucose, smoking):
  # Label encoding untuk gender: Female=0, Male=1
  gender_encoded = 1 if gender.lower() == "male" else 0

  # One-hot encoding manual untuk smoking_history
  smoking_encoded = {f"smoking_history_{cat}": False for cat in SMOKING_CATEGORIES}
  if smoking in SMOKING_CATEGORIES:
      smoking_encoded[f"smoking_history_{smoking}"] = True

  # Gabungkan semua fitur menjadi satu dict
  features = {
    "gender": gender_encoded,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    **smoking_encoded
  }
  return pd.DataFrame([features])

def main():
  parser = argparse.ArgumentParser(description="Prediksi diabetes menggunakan model XGBoost")
  parser.add_argument("--gender", choices=["Male", "Female"], required=True)
  parser.add_argument("--age", type=float, required=True)
  parser.add_argument("--hypertension", type=int, choices=[0, 1], required=True)
  parser.add_argument("--heart_disease", type=int, choices=[0, 1], required=True)
  parser.add_argument("--bmi", type=float, required=True)
  parser.add_argument("--hba1c", type=float, required=True)
  parser.add_argument("--glucose", type=float, required=True)
  parser.add_argument("--smoking", choices=SMOKING_CATEGORIES, required=True)

  args = parser.parse_args()

  # Encode input
  input_df = encode_inputs(
      args.gender, args.age, args.hypertension, args.heart_disease,
      args.bmi, args.hba1c, args.glucose, args.smoking
  )

  # Load model
  if not MODEL_PATH.exists():
      raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")

  model = joblib.load(MODEL_PATH)

  # Prediksi
  prediction = model.predict(input_df)[0]
  result = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"

  print("\n=== Hasil Prediksi ===")
  print(f"Input: {args.__dict__}")
  print(f"Prediksi: {result}")

if __name__ == "__main__":
  main()
