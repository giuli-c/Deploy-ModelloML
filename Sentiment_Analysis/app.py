from fastapi import FastAPI, Request
from prometheus_client import Counter, Summary, generate_latest
import time
from src.model import Model
from src.predictor import SentimentPredictor

app = FastAPI()

# Creazione delle metriche Prometheus
REQUESTS = Counter("total_requests", "Numero totale di richieste ricevute")
SENTIMENT_COUNT = Counter("sentiment_predictions", "Numero di predizioni fatte", ["sentiment"])

# Creo un'istanza della classe SentimentPredictor
predictor = SentimentPredictor()

# Endpoint di test
@app.get("/")
async def home():
    return {"message": "API Sentiment Analysis con FastAPI"}

# Endpoint per la predizione
@app.post("/predict")
async def predict(request: Request):
    REQUESTS.inc()  # Incremento il contatore di richieste

    data = await request.json()
    text = data.get("text", "")

    if not text:
        return {"error": "Nessun testo fornito"}

    sentiment = predictor.predict(text)
    SENTIMENT_COUNT.labels(sentiment=sentiment["sentiment"]).inc()  # Registro il sentiment
    
    return sentiment

# Questo esporta le metriche su /metrics, che Prometheus raccoglierà ogni tot secondi.
@app.get("/metrics")
async def metrics():
    """Endpoint per Prometheus"""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)