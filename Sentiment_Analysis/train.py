from src.data_loader import DataLoader
from src.model import Model
from src.preprocess_data import PreprocessData
import os

data_loader = DataLoader()
train_df, _ = data_loader.load_data()

# Preprocessinge dei dati
train_df["text"] = train_df["text"].apply(PreprocessData.preprocess)
label_map = PreprocessData.mapping_label()
train_df["label"] = train_df["label"].map(label_map)

# Salvo il dataset nel formato richiesto da FastText
fasttext_txt = train_df[["label", "text"]].to_csv("train_fasttext.txt", sep=" ", index=False, header=False)
assert os.path.exists(fasttext_txt), "Errore: train_fasttext.txt non è stato creato."

# Addestro il modello
model = Model()
result = model.train("train_fasttext.txt")
assert result == 0, "Errore nel train del modello."


