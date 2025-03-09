import fasttext
import mlflow
import mlflow.pyfunc
import logging
import os

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Model:
    """
    Classe per la gestione del modello: addestramento e caricamento.
    """
    def __init__(self, model_path=None):
        """
        Inizializza il modello. Se esiste un modello pre-addestrato, lo carica.
        """
        # Directory del progetto
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.model_path = model_path or os.path.join(self.root_dir, "models", "fasttext_sentiment.bin")
        self.model = None

        if os.path.exists(self.model_path):
            self.load_model()
        else:
            logging.warning("Modello non trovato. È necessario addestrarlo prima di usarlo.")


    def train(self, data_path):
        """
        Addestra il modello su un dataset e lo salvo.
        """
        if not os.path.exists(data_path):
            logging.warning(f"Il dataset {data_path} non esiste.")
        
        mlflow.set_experiment("SentimentAnalysis")
        with mlflow.start_run():
            logging.info("Avvio dell'addestramento del modello...")
            self.model = fasttext.train_supervised(input=data_path)

            if self.model:
                os.makedirs("models", exist_ok=True)
                self.model.save_model(self.model_path)
                logging.info(f"Modello addestrato e salvato in {self.model_path}")
                
                # Registrazione del modello in MLflow
                mlflow.log_param("dataset", data_path)
                mlflow.log_artifact(self.model_path)
                logging.info("Modello registrato in MLflow con successo!")
            else:
                logging.error("Errore nell'addestramento del modello.")

    def load_model(self):
        """
        Carica il modello FastText salvato.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = fasttext.load_model(self.model_path)
                logging.info(f"Modello caricato con successo da {self.model_path}")
            except Exception as e:
                logging.error(f"Errore nel caricamento del modello: {e}")
        else:
            logging.error(f"Il file {self.model_path} non esiste. Impossibile caricare il modello.")