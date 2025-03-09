import sys 
import os
import pytest
import subprocess

# Aggiungo il percorso della cartella principale al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import Model

@pytest.fixture(scope="session")
def train_model_file():
    """
    Eseguo train.py una volta prima di tutti i test.
    In tal modo viene assicurato che il modello sia disponibile e quindi creato.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_script = os.path.join(root_dir, "train.py")

    model_file = os.path.join(root_dir, "models/sentiment_model.ftz")
    if not os.path.exists(model_file):
        result = subprocess.run(["python", train_script], capture_output=True, text=True)
        assert result.returncode == 0, f"Train.py ha fallito: {result.stderr}"

    return model_file, root_dir  


def test_exist_model(train_model_file, _):
    """
    Verifico che il modello sia stato creato correttamente!
    """
    # Verifica che il modello sia stato creato
    assert os.path.exists(train_model_file), "Errore: Il modello NON esiste!"

    
def test_model_prediction(train_model_file, root_dir):
    """
    Verifico che il modello FastText riesca a fare predizioni.
    """
    model_path = os.path.join(root_dir, "models/sentiment_model.ftz")

    model = Model(model_path)
    prediction = model.model.predict("Covid cases are increasing fast!")
    assert prediction is not None, "Errore: la predizione non è stata eseguita."