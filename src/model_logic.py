import joblib
import pickle
from lib_ml import preprocessing

class ModelLogic:
    """ Wrapper around the model """

    def __init__(self, model_path, cvFile_path, ):
        self.classifier = joblib.load(model_path)
        self.cv = joblib.load(cvFile_path)

    def predict(self, review) -> int:
        processed_review = preprocessing._text_process(review)
        processed_input = self.cv.transform([" ".join(processed_review)]).toarray()
        prediction = self.classifier.predict(processed_input)[0]
        return int(prediction)
