import fasttext

class Model:
    def __init__(self, model_path=None):
        self.model = fasttext.load_model(model_path) if model_path else None

    def train(self, data_path, output_model):
        self.model = fasttext.train_supervised(input=data_path)
        self.model.save_model(output_model)

    def predict(self, text):
        return self.model.predict(text)