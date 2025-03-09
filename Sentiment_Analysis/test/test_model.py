import pytest
import sys 
import os
import subprocess
import pandas as pd

# Aggiungo il percorso della cartella principale al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import Model
from src.data_loader import DataLoader
from src.preprocess_data import PreprocessData
from src.predictor import SentimentPredictor

@pytest.fixture(scope="session")
def prepare_training_data():
    """
    Esegue `train.py` una sola volta prima dei test.
    Assicura che il file train_fasttext.txt e il modello siano disponibili.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_script = os.path.join(root_dir, "train.py")

    # Percorsi dei file da verificare
    train_file = os.path.join(root_dir, "train_fasttext.txt")
    model_file = os.path.join(root_dir, "models/fasttext_sentiment.bin")

    # Se il modello non esiste, viene eseguito train.py
    if not os.path.exists(model_file):
        result = subprocess.run(["python", train_script], capture_output=True, text=True)
        assert result.returncode == 0, f"Train.py ha fallito: {result.stderr}"

    # Verifico che i file siano stati creati
    assert os.path.exists(train_file), "Errore: train_fasttext.txt non è stato creato!"
    assert os.path.exists(model_file), "Errore: Il modello non è stato creato!"

    return train_file, model_file

def test_data_loader():
    """
    Verifica che il DataLoader carichi correttamente i dati.
    """
    data_loader = DataLoader()
    train_df, test_df = data_loader.load_data()

    assert isinstance(train_df, pd.DataFrame), "Il train set non è un DataFrame!"
    assert isinstance(test_df, pd.DataFrame), "Il test set non è un DataFrame!"
    assert not train_df.empty, "Il train set è vuoto!"
    assert not test_df.empty, "Il test set è vuoto!"

def test_preprocess_data():
    """
    Verifica il preprocessing del testo.
    """
    text = "@mario Guarda questo link: http://example.com"
    expected_output = "@user Guarda questo link: http"
    
    processed_text = PreprocessData.preprocess(text)
    assert processed_text == expected_output, f"Preprocessing errato! Output: {processed_text}"

def test_load_model(prepare_training_data):
    """
    Verifica che il modello venga caricato correttamente dopo l'addestramento.
    """
    _, model_file = prepare_training_data
    model = Model(model_path=model_file)
    model.load_model()
    
    assert model.model is not None, "Errore: Il modello non è stato caricato correttamente."

def test_prediction(prepare_training_data):
    """
    Verifica che il modello riesca a fare una predizione dopo l'addestramento.
    """
    _, model_file = prepare_training_data
    predictor = SentimentPredictor()
    
    prediction = predictor.predict("Questo prodotto è fantastico!")
    
    assert prediction is not None, "Errore: La predizione non è stata eseguita."
    assert "sentiment" in prediction and "confidence" in prediction, "Errore: Il formato della predizione è errato."


