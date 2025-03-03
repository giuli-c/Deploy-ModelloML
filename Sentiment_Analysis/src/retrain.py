from src.data_loader import DataLoader
from src.model import Model
from src.preprocess_data import PreprocessData

def retrain():
    """
    Ricarica i dati, aggiorna il dataset e ri-addestra il modello.
    """
    data_loader = DataLoader()
    train_df, _ = data_loader.load_data()
    
    # Preprocessinge dei dati
    train_df["text"] = train_df["text"].apply(PreprocessData.preprocess)
    label_map = PreprocessData.mapping_label()
    train_df["label"] = train_df["label"].map(label_map)

    # Salvo i dati nel formato FastText
    # Il formato richiesto da FastText per il training è un file di testo (.txt) con il seguente schema:
    # __label__sentiment Testo della recensione o del tweet
    train_df[["label", "text"]].to_csv("train_fasttext.txt", sep=" ", index=False, header=False, quoting=3)

    # Retrain del modello
    model = Model()
    train_model("train_fasttext.txt")

if __name__ == "__main__":
    retrain()