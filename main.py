# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.domain.FormData import FormData
from model.run_model import predict_from_frontend

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
    resultado = predict_from_frontend(data)
    predicciones = resultado['predictions']  # Solo las predicciones
    print(resultado)
    respuesta = {nombre: float(valor) for nombre, valor in predicciones.items()}
    return respuesta

