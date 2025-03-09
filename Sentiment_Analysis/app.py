from fastapi import FastAPI, Request
from prometheus_client import Counter, Summary, generate_latest
import time
from src.model import Model
from src.predictor import SentimentPredictor

app = FastAPI()

# Crea le metriche Prometheus
REQUESTS = Counter("total_requests", "Numero totale di richieste ricevute")
LATENCY = Summary("request_latency_seconds", "Tempo di risposta in secondi")
SENTIMENT_COUNT = Counter("sentiment_predictions", "Numero di predizioni fatte", ["sentiment"])


class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(input: TextInput):
    """
    Endpoint per ottenere il sentiment di un testo.
    """
    return predict(input.text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)