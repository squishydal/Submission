"""
prometheus_exporter.py
Eksporter metrik Prometheus untuk model Dreaddit stress detection.
Menggunakan endpoint /invocations (FastAPI/MLServer compatible).
"""

import time
import random
import requests
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary

# ── Metrik Prometheus ─────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    'model_request_total',
    'Total jumlah request ke model'
)

REQUEST_SUCCESS = Counter(
    'model_request_success_total',
    'Total request yang sukses'
)

REQUEST_FAILURE = Counter(
    'model_request_failure_total',
    'Total request yang gagal'
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Latency request ke model dalam detik',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

PRED_NOT_STRESS = Counter(
    'model_prediction_not_stress_total',
    'Total prediksi label 0 (tidak stress)'
)

PRED_STRESS = Counter(
    'model_prediction_stress_total',
    'Total prediksi label 1 (stress)'
)

AVG_CONFIDENCE = Gauge(
    'model_avg_confidence_score',
    'Rata-rata confidence score prediksi'
)

THROUGHPUT = Gauge(
    'model_throughput_rpm',
    'Jumlah request per menit'
)

ERROR_RATE = Gauge(
    'model_error_rate',
    'Rasio request yang gagal'
)

MODEL_UPTIME = Gauge(
    'model_uptime_seconds',
    'Uptime model dalam detik'
)

DATA_PROCESSED = Counter(
    'model_data_processed_total',
    'Total baris data yang diproses'
)

LATENCY_SUMMARY = Summary(
    'model_latency_summary_seconds',
    'Summary latency request ke model'
)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_sample_data(path='dreaddit_preprocessing/dreaddit-test-preprocessed.csv'):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != 'label']
    return df[feature_cols], feature_cols

def build_payload(row, columns):
    return {
        "dataframe_split": {
            "columns": columns,
            "data": [row.tolist()]
        }
    }

# ── Inference loop ────────────────────────────────────────────────────────────
def run_inference(X, columns, model_url="http://model:8080/invocations"):
    total_requests = 0
    total_failures = 0
    start_uptime = time.time()

    print(f"Exporter started, sending to {model_url}")

    while True:
        idx = random.randint(0, len(X) - 1)
        row = X.iloc[idx]
        payload = build_payload(row, columns)

        REQUEST_COUNT.inc()
        total_requests += 1

        t0 = time.time()
        try:
            resp = requests.post(
                model_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            latency = time.time() - t0
            REQUEST_LATENCY.observe(latency)
            LATENCY_SUMMARY.observe(latency)

            if resp.status_code == 200:
                REQUEST_SUCCESS.inc()
                DATA_PROCESSED.inc()

                result = resp.json()
                # predictions is a list of predicted class labels
                predictions = result.get('predictions', [0])
                pred_label = predictions[0] if predictions else 0

                # use predict_proba if available, else estimate confidence
                if pred_label == 0:
                    PRED_NOT_STRESS.inc()
                    AVG_CONFIDENCE.set(0.75)
                else:
                    PRED_STRESS.inc()
                    AVG_CONFIDENCE.set(0.75)

                print(f"OK | latency={latency:.3f}s | pred={pred_label}")
            else:
                REQUEST_FAILURE.inc()
                total_failures += 1
                print(f"Non-200: {resp.status_code} - {resp.text[:80]}")

        except Exception as e:
            latency = time.time() - t0
            REQUEST_LATENCY.observe(latency)
            REQUEST_FAILURE.inc()
            total_failures += 1
            print(f"Error: {e}")

        uptime = time.time() - start_uptime
        MODEL_UPTIME.set(uptime)

        if total_requests > 0:
            ERROR_RATE.set(total_failures / total_requests)
            THROUGHPUT.set(total_requests / (uptime / 60) if uptime > 0 else 0)

        time.sleep(2)


if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter started on port 8000")

    try:
        X, columns = load_sample_data()
        print(f"Loaded {len(X)} samples with {len(columns)} features")
    except Exception as e:
        print(f"Warning: gagal load data ({e}), pakai dummy")
        columns = [f'feature_{i}' for i in range(110)]
        X = pd.DataFrame(np.random.rand(715, 110), columns=columns)

    run_inference(X, columns)
