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
    def __init__(self, F=None):
        """
        Inizializza il modello. Se esiste un modello pre-addestrato, lo carica.
        """
        # Directory del progetto
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.model_path = model_path or os.path.join(self.root_dir, "models", "fasttext_sentiment.bin")
        self.model = None

        if os.path.exists(self.model_path):
            print(f"------- Load del modello del path: {model_path}")
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
            print("----- Avvio dell'addestramento del modello...")
            self.model = fasttext.train_supervised(input=data_path)

            if self.model:
                model_dir = os.path.dirname(self.model_path)
                os.makedirs(model_dir, exist_ok=True)

                print(f"----- Provo a salvare il modello in {self.model_path}")
                if os.path.exists(self.model_path):
                    os.remove(self.model_path)
                    print(f"----- Eliminato quello esistente")
                
                self.model.save_model(self.model_path)
                logging.info(f"Modello addestrato e salvato in {self.model_path}")
                print(f"----- Salvato il nuovo modello in {self.model_path}")

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