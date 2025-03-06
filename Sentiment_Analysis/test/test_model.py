import sys 
import os
import pytest
import subprocess

# Aggiungo il percorso della cartella principale al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import Model

def test_model_training():
    """
    Verifico che il file train.py venga eseguito correttamente e generi i file necessari per la predizione.
    """
    # Eseguo il training eseguendo train.py
    # definisco la cartella in cui si trova il processo
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_script = os.path.join(root_dir, "train.py")

    result = subprocess.run(["python", train_script], capture_output=True, text=True)
    assert result.returncode == 0, f"Train.py ha fallito la sua esecuzione"

    # Controllo che il file di training sia stato creato
    train_file = os.path.join(root_dir, "train_fasttext.txt")
    assert os.path.exists(train_file), "Errore: train_fasttext.txt non è stato creato."

    # Controllo che il modello sia stato salvato
    model_file = os.path.join(root_dir, "models/sentiment_model.ftz")
    assert os.path.exists(model_file), "Errore: Il modello non è stato salvato correttamente."

    
def test_model_prediction():
    """
    Verifico che il modello FastText riesca a fare predizioni.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root_dir, "models/sentiment_model.ftz")

    model = Model(model_path)
    prediction = model.model.predict("Covid cases are increasing fast!")
    assert prediction is not None, "Errore: la predizione non è stata eseguita."