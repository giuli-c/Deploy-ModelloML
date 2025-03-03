from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name="cardiffnlp/tweet_sentiment_multilingual"):
        self.dataset_name = dataset_name

    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        return dataset["train"], dataset["test"]