from flask import Flask, request, jsonify

from model_logic import ModelLogic

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)