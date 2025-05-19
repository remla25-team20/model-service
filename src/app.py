from flask import Flask, request, jsonify

from model_logic import ModelLogic
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Define Prometheus Counter
frontend_events_total = Counter(
    'frontend_events_total',
    'Counts frontend UI events',
    ['event']  # label: event type
)

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
    review = request.args['review']
    app.logger.debug(f'review={review}')
    prediction = model.predict(review)
    app.logger.debug(f'prediction={prediction}')
    return jsonify(prediction=prediction)

@app.route("/log-metric", methods=["POST"])
def log_metric():
    """
    Endpoint to receive frontend event metrics
    """
    data = request.get_json()
    event = data.get('event')
    if event:
        frontend_events_total.labels(event=event).inc()
        return "", 204
    return jsonify({"error": "Missing event field"}), 400

@app.route("/metrics")
def metrics():
    """
    Endpoint to expose Prometheus metrics
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)