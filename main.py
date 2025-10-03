# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.domain.FormData import FormData
from model.run_model import predict_from_frontend
import os

from model.train_model import train_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=origins,  # o ["*"] para todos los orígenes (solo desarrollo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(data: FormData):
    print(data)

    if not os.path.exists('model/weights/best_model.weights.h5') or not os.path.exists('model/weights/normalization_stats.json'):
        train_model()

    resultado = predict_from_frontend(data)
    predicciones = resultado['predictions']
    print(resultado)
    respuesta = {nombre: float(valor) for nombre, valor in predicciones.items()}
    return respuesta

