from src.model import Model
import torch

class SentimentPredictor:
    """
    Classe per la predizione del sentiment usando FastText.
    """
    # def __init__(self, model=None):
    #     """
    #     Inizializza il modello per la predizione.
    #     """
    #     self.model = Model().model

    # def predict(self, text):
    #     """
    #     Esegue una predizione sul testo dato.
    #     """
    #     label, prob = self.model.predict(text)

    #     # ERRORE: [*ValueError: Unable to avoid copy while creating an array as requested. If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)`*] 
    #     # Questo errore proviene da FastText, che sta tentando di usare np.array(obj, copy=False), ma NumPy 2.0 ha cambiato il comportamento di copy=False. 
    #     # PRIMA: Creava un array senza copiare obj - Se obj non era già un array NumPy, veniva fatta una conversione automatica - Se obj non poteva essere usato direttamente, NumPy faceva una copia senza generare errori
    #     # ORA: NumPy non copia più obj se non è già un array compatibile. Se è necessario copiare, invece di farlo automaticamente, genera un errore.
    #     # >>>> Rimuozione di newline e spazi extra
    #     #text = text.strip().replace("\n", " ")

    #     return {"sentiment": label[0], "confidence": prob[0]}
     def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
        return pred