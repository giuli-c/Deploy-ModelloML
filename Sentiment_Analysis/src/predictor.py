import fasttext

class SentimentPredictor:
    def __init___(self, model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model = fasttext.load_model(model_path)

    def predict(self, text):
        prediction = sel.model.predict(text)
        return prediction[0][0]