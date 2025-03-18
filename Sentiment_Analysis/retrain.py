# import os
# import fasttext
# import mlflow
# import pandas as pd
# from src.data_loader import DataLoader
# from src.model import Model
# from src.preprocess_data import PreprocessData

# def retrain():
#     """
#     Ricarica i dati, aggiorna il dataset e ri-addestra il modello.
#     """
#     # Directory del progetto
#     root_dir = os.path.abspath(os.path.dirname(__file__))
#     # Percorso assoluto per evitare problemi di percorso
#     retrain_file_path = os.path.join(root_dir, "train_fasttext_retrain.txt")

#     print("Inizio del retraining del modello...")

#      # Usa il dataset di monitoraggio per il retraining
#     train_df = pd.read_csv("sentiment_monitoring.csv", names=["text", "sentiment"])
    
#     # Preprocessinge dei dati
#     train_df["text"] = train_df["text"].apply(PreprocessData.preprocess)
#     label_map = PreprocessData.mapping_label()
#     train_df["label"] = train_df["label"].map(label_map)

#     # Salvo i dati nel formato FastText
#     train_df[["label", "text"]].to_csv(retrain_file_path, sep=" ", index=False, header=False)

#     # Retrain del modello
#     new_model_path = os.path.join(root_dir, "models", "train_fasttext_retrain.txt")
#     model = Model(new_model_path)
#     model.train(retrain_file_path)
