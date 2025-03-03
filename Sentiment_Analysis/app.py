from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import SentimentPredictor

app = FastAPI()

# Caricamento del modello
predictor = SentimentPredictor()

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