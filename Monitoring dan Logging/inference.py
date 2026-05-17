"""
inference.py
Script untuk melakukan inferensi manual ke model Dreaddit.
Bisa dijalankan terpisah dari exporter untuk testing.
"""

import requests
import pandas as pd
import json
import random

MODEL_URL = "http://localhost:5001/v2/models/model/infer"

def load_sample(path='dreaddit_preprocessing/dreaddit-test-preprocessed.csv', n=5):
    df = pd.read_csv(path)
    X = df.drop(columns=['label'])
    y = df['label']
    indices = random.sample(range(len(df)), n)
    return X.iloc[indices].values, y.iloc[indices].values

def predict(row):
    payload = {
        "inputs": [
            {
                "name": "predict-prob",
                "shape": [1, len(row)],
                "datatype": "FP64",
                "data": row.tolist()
            }
        ]
    }
    resp = requests.post(MODEL_URL, json=payload, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    data = result['outputs'][0]['data']
    pred = data.index(max(data))
    confidence = max(data)
    return pred, confidence

if __name__ == '__main__':
    print(f"Mengirim request ke {MODEL_URL}\n")
    X, y_true = load_sample()

    for i, (row, label) in enumerate(zip(X, y_true)):
        try:
            pred, conf = predict(row)
            status = "BENAR" if pred == label else "SALAH"
            label_str = "Stress" if label == 1 else "Tidak Stress"
            pred_str  = "Stress" if pred == 1 else "Tidak Stress"
            print(f"[{i+1}] Aktual: {label_str} | Prediksi: {pred_str} | Confidence: {conf:.3f} | {status}")
        except Exception as e:
            print(f"[{i+1}] Error: {e}")
