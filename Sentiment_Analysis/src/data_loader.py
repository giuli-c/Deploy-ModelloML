from datasets import load_dataset

class DataLoader:
    """
    Classe usata per caricare il dataset.
    """

    def __init__(self, dataset_name="cardiffnlp/tweet_sentiment_multilingual"):
        self.dataset_name = dataset_name

    def load_data(self):
        """
        Carica il dataset di HuggingFace.
        Ritornano due datafram: train e test.
        """
        dataset = load_dataset(self.dataset_name, "italian")
        return dataset["train"].to_pandas(), dataset["test"].to_pandas()