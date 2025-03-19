import pandas as pd
from src.predictor import SentimentPredictor

class SentimentTester:
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.predictor = SentimentPredictor(tokenizer, model)
        self.df_test = self.test_dataset.to_pandas()

    def predict_sentiment(self):
        results = []
        for _, row in self.df_test.iterrows():
            text = row["text"]
            true_label = row["label"]
            pred_label = self.predictor.predict(text)
            results.append((text, pred_label, true_label))
        return results

    def get_results_dataframe(self):
        predictions = self.predict_sentiment()
        df_results = pd.DataFrame(predictions, columns=["Testo", "Predetto", "Reale"])
        return df_results

    def print_sample_results(self, n=5):
        df = self.get_results_dataframe()
        print(df.head(n))
