#from src.model import Model
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