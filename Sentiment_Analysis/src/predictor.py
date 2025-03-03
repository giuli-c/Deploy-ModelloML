from src.model import Model

class SentimentPredictor:
    """
    Classe per la predizione del sentiment usando FastText.
    """
    def __init___(self, model=None):
        """
        Inizializza il modello per la predizione.
        """
        self.model = Model().model

    def predict(self, text):
        """
        Esegue una predizione sul testo dato.
        """
        if not self.model:
            raise RuntimeError("Il modello non è caricato. Esegui addestramento o carica un modello esistente.")
        
        label, prob = self.model.predict(text)
        return {"sentiment": label[0], "confidence": prob[0]}