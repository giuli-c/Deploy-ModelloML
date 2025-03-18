from Sentiment_Analysis.src.load_config import LoadConfig

def test_load_config():
    config = LoadConfig().load_config()
    assert "model" in config
