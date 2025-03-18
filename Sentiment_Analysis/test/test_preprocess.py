from Sentiment_Analysis.src.preprocess_data import PreprocessData

def test_preprocess():
    text = "Questo è un TEST! http://link.com"
    cleaned = PreprocessData.preprocess(text)
    assert "http" not in cleaned and "!" not in cleaned