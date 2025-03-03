import pytest
from src.model import Model

def test_model_training():
    """
    Verifico che il modello FastText venga addestrato correttamente.
    """
    model = Model()
    model.train("train_fasttext.txt")
    assert model.model is not None, "Errore: il modello non è stato addestrato correttamente."

def test_model_prediction():
    """
    Verifico che il modello FastText riesca a fare predizioni.
    """
    model = Model()
    prediction = model.model.predict("Covid cases are increasing fast!")
    assert prediction is not None, "Errore: la predizione non è stata eseguita."