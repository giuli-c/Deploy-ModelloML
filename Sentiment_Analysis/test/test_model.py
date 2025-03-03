import pytest
from src.model import Model

@pytest.fixture
def model():
    return Model("cardiffnlp/twitter-roberta-base-sentiment-latest")

def test_predict(model):
    prediction = model.predict("Covid cases are increasing fast!")
    assert prediction in ["negative", "neutral", "positive"]