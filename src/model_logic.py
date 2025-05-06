import joblib
import pickle
from lib_ml import preprocessing

class ModelLogic:
    """ Wrapper around the model """

    def __init__(self, model_path, cvFile_path, ):
        self.classifier = joblib.load(model_path)
        self.cv = pickle.load(open(cvFile_path, "rb"))

    def predict(self, review) -> int:
        processed_review = preprocessing._text_process(review)
        print(f'processed_review = {processed_review}')
        processed_input = self.cv.transform([processed_review]).toarray()[0]
        print(f'processed_input = {processed_input}')
        prediction = self.classifier.predict([processed_input])[0]
        print(f'prediction = {prediction}')
        return int(prediction)

if __name__ == '__main__':
    model = ModelLogic(
        '../model/c2_Classifier_Sentiment_Model',
        '../model/c1_BoW_Sentiment_Model.pkl')
    prediction_result = model.predict('the food was not bad')
    print(prediction_result)
