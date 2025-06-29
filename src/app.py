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

word_count_raw = Counter(
    "word_count_raw",
    "Total number of non-stop words submitted",
    ['app_version', 'model_service_version']
)

word_count_encoded = Counter(
    "word_count_encoded",
    "Total number of words successfully encoded",
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

models = {}
app = Flask(__name__)

def init_data():
    fname_cv = "Sentiment_Analysis_Preprocessor.joblib"
    fname_model = "Sentiment_Analysis_Model.joblib"

    def fetch_releases():
        url = "https://api.github.com/repos/remla25-team20/model-training/releases"

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        resp = requests.get(url, headers=headers)

        if resp.status_code != 200:
            raise Exception(f"Failed to fetch releases: {resp.status_code}")
        
        releases = resp.json()
        version = [release['tag_name'] for release in releases]

        print(f'got model releases from GH: {version}')

        url_models = [release['assets'][0]['browser_download_url'] for release in releases if release['assets']]
        url_cvs = [release['assets'][1]['browser_download_url'] for release in releases if len(release['assets']) > 1]
        return zip(version, url_models, url_cvs)

    fetched_models = list(fetch_releases())
    for version, url_model, url_cv in fetched_models:

        target_dir = f"/mnt/shared/models/{version}/"

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

        model = ModelLogic()
        latest = f"/mnt/shared/models/{fetched_models[0][0]}/"
        model.set_classifier_path(latest + fname_model)
        model.set_cv_path(latest + fname_cv)
        model.initialize_models()
        models[version] = model
    return

@app.route("/model-versions", methods=["GET"])
def get_model_versions():
    """
    Get available model versions.

    Summary:
        Returns a list of model version identifiers currently available for inference.

    Parameters:
        None

    Responses:
        200:
            description: A JSON object containing the list of available model versions.
            schema:
                type: object
                properties:
                    modelVersions:
                        type: array
                        items:
                            type: string
                        description: List of model version identifiers.
    """
    return jsonify(modelVersions=list(models.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict sentiment based on the submitted review text.

    Summary:
        Accepts a review text and model version as input and returns a sentiment prediction.

    Parameters:
        - name: review
          in: query
          type: string
          required: true
          description: The user's review text to be analyzed.
        - name: modelVersion
          in: query
          type: string
          required: true
          description: The version of the model to be used for prediction.

    Responses:
        200:
            description: A JSON object with the prediction result.
            schema:
                type: object
                properties:
                    prediction:
                        type: string
                        description: The predicted label (e.g., '1' for positive, '0' for negative).
        400:
            description: Missing or invalid input.
    """

    app_version = request.cookies.get('version')
    app.logger.debug(f'version_cookie={app_version}')
    if app_version == None:
        app_version = 'unknown'

    start = time.time()
    review = request.args['review']
    model_version = request.args['modelVersion']
    app.logger.debug(f"/predict: MODEL_VERSION={model_version}")
    print(f"The review is {review}")
    print(f"The review datatype is {type(review)}")
    app.logger.debug(f'review={review}')

    wc_raw, wc_encoded, prediction = models[model_version].predict(review)   
 
    app.logger.debug(f'prediction={prediction}')
    
    # Record metrics
    request_latency_seconds.labels(
        model_service_version=MODEL_SERVICE_VERSION, 
        app_version=app_version
        ).observe(time.time() - start)  # record latency
    word_count_raw.labels(
        model_service_version=MODEL_SERVICE_VERSION,
        app_version=app_version
        ).inc(wc_raw)               # record non-stop word word count
    word_count_encoded.labels(
        model_service_version=MODEL_SERVICE_VERSION,
        app_version=app_version
        ).inc(wc_encoded)           # record encoded word count
    
    return jsonify(prediction=prediction)

@app.route("/feedback", methods=["POST"])
def store_user_feedback():
    """
    Submit user feedback about the correctness of the prediction.

    Summary:
        Stores feedback used for model improvement and retraining.

    Parameters:
        - in: body
          name: feedback
          required: true
          schema:
              type: object
              properties:
                  reviewText:
                      type: string
                      description: Original review text.
                  prediction:
                      type: integer
                      description: The predicted label (0 or 1).
                  isPredictionCorrect:
                      type: boolean
                      description: Whether the prediction was correct.

    Responses:
        204:
            description: Feedback successfully stored (no content).
        400:
            description: Invalid request body.
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
    Log frontend events for analytics and monitoring.

    Summary:
        Receives a frontend event and increments the appropriate Prometheus counter.

    Parameters:
        - in: body
          name: event
          required: true
          schema:
              type: object
              properties:
                  event:
                      type: string
                      enum: ["frontend_submit_clicked", "frontend_prediction_result", "frontend_prediction_error", "frontend_review_started"]
                      description: The name of the frontend event.

    Responses:
        204:
            description: Event successfully logged.
        400:
            description: Unknown or missing event name.
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
    Expose Prometheus metrics for scraping.

    Summary:
        Returns all service metrics in Prometheus text format.

    Parameters:
        None

    Responses:
        200:
            description: Prometheus-formatted metrics output.
            content:
                text/plain:
                    schema:
                        type: string
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
    init_data()
    app.run(host="0.0.0.0", port=8080, debug=True)

