from flask import Flask, request, jsonify

from model_logic import ModelLogic
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psutil, os, threading, time


# ──────────────────────────
# Metrics definitions
# ──────────────────────────
submit_click_total = Counter(
    "submit_click_total",
    "Total number of times the Submit button is clicked"
)

prediction_success_total = Counter(
    "prediction_success_total",
    "Total number of successful predictions"
)

prediction_error_total = Counter(
    "prediction_error_total",
    "Total number of failed predictions"
)

request_latency_seconds = Histogram(
    "request_latency_seconds",
    "Latency of the /predict endpoint in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)
model_cpu_percent = Gauge(
    "model_cpu_percent",
    "CPU usage of the model-service process (%)"
)
model_memory_rss_bytes = Gauge(
    "model_memory_rss_bytes",
    "Resident memory size (RSS) of the model-service process in bytes"
)
# ──────────────────────────
# Flask application
# ──────────────────────────

model = ModelLogic(
    'model/c2_Classifier_Sentiment_Model',
    'model/c1_BoW_Sentiment_Model.pkl')

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for making predictions.
    ---
    parameters:
      - name: review
        in: query
        type: string
        required: true
        description: The URL of the input data.
    responses:
      200:
        description: The predicted review type.
        schema:
          type: string
    """
    start = time.time()
    review = request.args['review']
    app.logger.debug(f'review={review}')
    prediction = model.predict(review)   
    app.logger.debug(f'prediction={prediction}')
    request_latency_seconds.observe(time.time() - start)  # record latency
    return jsonify(prediction=prediction)

EVENT_TO_COUNTER = {
    "frontend_submit_clicked": submit_click_total,
    "frontend_prediction_result": prediction_success_total,
    "frontend_prediction_error": prediction_error_total,
}

@app.route("/log-metric", methods=["POST"])
def log_metric():
    """
    Receive a frontend event and increment the corresponding counter.
    Expected JSON body: { "event": "<event_name>" }
    """
    data = request.get_json()
    event = data.get('event')
    if event in EVENT_TO_COUNTER:
        EVENT_TO_COUNTER[event].inc()
        return "", 204
    return jsonify({"error": "Unknown or missing event"}), 400

@app.route("/metrics")
def metrics():
    """
    Endpoint to expose Prometheus metrics
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

def _resource_monitor():
    proc = psutil.Process(os.getpid())
    # First call to cpu_percent “starts” the measurement window
    proc.cpu_percent(interval=None)
    while True:
        model_cpu_percent.set(proc.cpu_percent(interval=None))
        model_memory_rss_bytes.set(proc.memory_info().rss)
        time.sleep(5)           
        
threading.Thread(target=_resource_monitor, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)