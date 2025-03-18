from Sentiment_Analysis.src.load_model import LoadModel

def test_model_loading():
    loader = LoadModel()
    tokenizer, model = loader.load_model()
    assert tokenizer is not None and model is not None