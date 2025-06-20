from flask import Flask, request, jsonify

from model_logic import ModelLogic
from lib_ml import preprocessing
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psutil, os, threading, time, csv
import sys
import requests
import fcntl

MODEL_SERVICE_VERSION = os.getenv("MODEL_SERVICE_VERSION", "unknown")
USER_FEEDBACK_DIR = '/mnt/shared/user-feedback-data'
USER_FEEDBACK_PATH = f'{USER_FEEDBACK_DIR}/user_feedback.csv'

# ──────────────────────────
# Metrics definitions
# ──────────────────────────
submit_click_total = Counter(
    "submit_click_total",
    "Total number of times the Submit button is clicked",
    ['app_version', 'model_service_version']
)

reviews_started = Counter(
    'reviews_started_total',
    'Total number of users who started writing a review',
    ['app_version', 'model_service_version']
)

prediction_success_total = Counter(
    "prediction_success_total",
    "Total number of successful predictions",
    ['app_version', 'model_service_version']
)

prediction_error_total = Counter(
    "prediction_error_total",
    "Total number of failed predictions",
    ['app_version', 'model_service_version']
)

request_latency_seconds = Histogram(
    "request_latency_seconds",
    "Latency of the /predict endpoint in seconds",
    ['app_version', 'model_service_version'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)
model_cpu_percent = Gauge(
    "model_cpu_percent",
    "CPU usage of the model-service process (%)",
    ['app_version', 'model_service_version']
)
model_memory_rss_bytes = Gauge(
    "model_memory_rss_bytes",
    "Resident memory size (RSS) of the model-service process in bytes",
    ['app_version', 'model_service_version']
)
# ──────────────────────────
# Flask application
# ──────────────────────────

model = ModelLogic()

app = Flask(__name__)


def init_data(version):
    url_cv = f"https://github.com/remla25-team20/model-training/releases/download/{version}/Sentiment_Analysis_Preprocessor.joblib"
    url_model = f"https://github.com/remla25-team20/model-training/releases/download/{version}/Sentiment_Analysis_Model.joblib"
    
    target_dir = f"/mnt/shared/models/{version}/"
    fname_cv = "Sentiment_Analysis_Preprocessor.joblib"
    fname_model = "Sentiment_Analysis_Model.joblib"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    def _download_file(url, target):
        if os.path.isfile(target):
            return
        resp = requests.get(url)
        if resp.status_code != 200:
            raise FileNotFoundError(f"Could not download file at {url}")
        with open(target, "wb+") as f:
            f.write(resp.content)
    
    _download_file(url_cv, target_dir + fname_cv)
    _download_file(url_model, target_dir + fname_model)
    
    model.set_classifier_path(target_dir + fname_model)
    model.set_cv_path(target_dir + fname_cv)
    
    model.initialize_models()
    return


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

    app_version = request.cookies.get('version')
    app.logger.debug(f'version_cookie={app_version}')
    if app_version == None:
        app_version = 'unknown'

    start = time.time()
    review = request.args['review']
    print(f"The review is {review}")
    print(f"The review datatype is {type(review)}")
    app.logger.debug(f'review={review}')
    prediction = model.predict(review)   
    app.logger.debug(f'prediction={prediction}')
    request_latency_seconds.labels(
        model_service_version=MODEL_SERVICE_VERSION, 
        app_version=app_version
        ).observe(time.time() - start)  # record latency
    return jsonify(prediction=prediction)

@app.route("/feedback", methods=["POST"])
def store_user_feedback():
    """
    Write user feedback regarding the correctness of the prediction to a csv file.
    The data can be exported later to retrain/improve the model.
    Expected JSON body: {
      "reviewText": <String>
      "prediction": <1|0>
      "isPredictionCorrect": <True|False>
      }
    """
    data = request.get_json()
    review_text, prediction, isPredictionCorrect = data.get('reviewText'), data.get('prediction'), data.get('isPredictionCorrect')
    
    if not os.path.exists(USER_FEEDBACK_DIR):
        os.mkdir(USER_FEEDBACK_DIR)

    with open(USER_FEEDBACK_PATH, 'a', newline='') as feedback_file:
        # lock file while writing so multiple pods on same node don't overwrite each other
        fcntl.flock(feedback_file, fcntl.LOCK_EX)
        writer = csv.writer(feedback_file)
        # write header first time
        if (not os.path.exists(USER_FEEDBACK_PATH) or os.path.getsize(USER_FEEDBACK_PATH) == 0):
            writer.writerow(['review_text','prediction','isPredictionCorrect'])
        writer.writerow([review_text, prediction, isPredictionCorrect])
        fcntl.flock(feedback_file, fcntl.LOCK_UN)
    return "Feedback successfully collected.", 204

EVENT_TO_COUNTER = {
    "frontend_submit_clicked": submit_click_total,
    "frontend_prediction_result": prediction_success_total,
    "frontend_prediction_error": prediction_error_total,
    "frontend_review_started": reviews_started
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
        app_version = request.cookies.get('version')
        app.logger.debug(f'version_cookie={app_version}')
        if app_version == None:
            app_version = 'unknown'
        EVENT_TO_COUNTER[event].labels(
            model_service_version=MODEL_SERVICE_VERSION, 
            app_version=app_version
            ).inc()
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
        model_cpu_percent.labels(
            model_service_version=MODEL_SERVICE_VERSION, 
            app_version='undefined'
            ).set(proc.cpu_percent(interval=None))
        model_memory_rss_bytes.labels(
            model_service_version=MODEL_SERVICE_VERSION, 
            app_version='undefined'
            ).set(proc.memory_info().rss)
        time.sleep(5)
        
threading.Thread(target=_resource_monitor, daemon=True).start()

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v0.1.6-beta"
    init_data(version)
    app.run(host="0.0.0.0", port=8080, debug=True)

