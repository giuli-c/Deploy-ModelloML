from trasformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class LoadModel:
    def __init__(self, model_name=None):
        if model_name:
            self.model_name = model_name
        else:
            from src.load_config import LoadConfig  # importa qui per evitare cicli
            config_loader = LoadConfig()
            config = config_loader.load_config()
            self.model_name = config.get("model", {}).get("pre_trained", None)

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return tokenizer, model


