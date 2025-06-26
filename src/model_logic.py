import joblib
import pickle
from lib_ml import preprocessing

class ModelLogic:
    """ Wrapper around the model """

    def __init__(self, model_path = None, cvFile_path = None, ):
        self.classifier_path = model_path
        self.cv_path = cvFile_path
        self.initialized = False
        self.initialize_models()
    
    def set_classifier_path(self, path):
        self.classifier_path = path
    
    def set_cv_path(self, path):
        self.cv_path = path
        
    def initialize_models(self):
        if not self.initialized:
            try:
                self.classifier = joblib.load(self.classifier_path)
                self.cv = joblib.load(self.cv_path)
                self.initialized = True
            except Exception as e:
                return False
        return True

    def predict(self, review) -> int:
        self.initialize_models()
        if not self.initialized:
            return -1
        processed_review = preprocessing._text_process(review)
        processed_input = self.cv.transform([" ".join(processed_review)]).toarray()
        prediction = self.classifier.predict(processed_input)[0]
        return len(processed_review), int(sum([sum(row) for row in processed_input])), int(prediction)
