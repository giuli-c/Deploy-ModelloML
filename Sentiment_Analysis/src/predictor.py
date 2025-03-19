import torch

class SentimentPredictor:
    """
    Classe per la predizione del sentiment usando FastText.
    """
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        # Controllo dove eseguire il modello
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Sposto gli input (tokenizzati) sullo stesso device del modello
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # disattivo il calcolo del gradiente > per risparmiare memoria
        with torch.no_grad():
            # Passo gli input tokenizzati al modello e ottengo l’output
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Predizione della classe più alta
            pred = torch.argmax(logits, dim=1).item()
        return pred