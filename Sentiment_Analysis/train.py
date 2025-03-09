from src.data_loader import DataLoader
from src.model import Model
from src.preprocess_data import PreprocessData
import os

# Directory del progetto
root_dir = os.path.abspath(os.path.dirname(__file__))
# Percorso assoluto per evitare problemi di percorso
train_file_path = os.path.join(root_dir, "train_fasttext.txt")

data_loader = DataLoader()
train_df, _ = data_loader.load_data()

# Preprocessinge dei dati
train_df["text"] = train_df["text"].apply(PreprocessData.preprocess)
label_map = PreprocessData.mapping_label()
train_df["label"] = train_df["label"].map(label_map)

# Salvo il dataset nel formato richiesto da FastText
train_df[["label", "text"]].to_csv(train_file_path, sep=" ", index=False, header=False)

# Verifico se il file è stato creato correttamente
if os.path.exists(train_file_path):
    print(f"File train_fasttext.txt creato correttamente in {train_file_path}")
else:
    print(f"ERRORE: Il file train_fasttext.txt non è stato creato!")

# Addestro il modello
model = Model()
result = model.train(train_file_path)

