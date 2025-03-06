FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia tutto il progetto dentro il container
COPY . . 

# Espone la porta per MLflow UI
EXPOSE 5000

#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] >> utile quando uso FASTAPI
# Comando di default: avvia il training e registra il modello su MLflow
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0", "--port", "5000"]