"""
Módulo API REST - Interfaz HTTP para Predicciones en Tiempo Real

Proporciona una API FastAPI con dos endpoints:
1. GET /: Verificar que la API esté activa
2. POST /predict: Realizar predicción y guardar en PostgreSQL

La API recibe 15 features, realiza predicción y persiste el resultado en la BD.
Está lista para ser desplegada en producción con uvicorn o similar.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model.predict import predict_sales
from src.storage.postgres import save_prediction_to_postgres


# Crear aplicación FastAPI
app = FastAPI(
    title="SuperFresh Sales Prediction API",
    description="API para predecir ventas de productos en supermercados",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    """
    Modelo Pydantic para validación de entrada de predicción.
    
    Valida automáticamente que:
    - Todos los campos sean presentes
    - Los tipos sean correctos
    - Genera documentación automática en /docs
    """
    id_tienda: int
    id_producto: int
    precio_unitario: float
    promocion_activa: int  # 0 o 1
    descuento_pct: float  # 0-100
    temperatura_media: float  # Grados Celsius
    lluvia_mm: float  # Milímetros
    anio: int
    mes: int  # 1-12
    dia: int  # 1-31
    dia_semana: int  # 1-7
    es_fin_de_semana: int  # 0 o 1
    lag_1: float  # Unidades vendidas ayer
    lag_7: float  # Unidades vendidas hace 7 días
    media_movil_7: float  # Promedio últimos 7 días


@app.get("/")
def root():
    """
    Endpoint de verificación de salud de la API.
    
    Returns:
        dict: Mensaje confirmando que la API está activa
    """
    return {"message": "API de predicción de ventas de SuperFresh activa"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint principal para realizar predicciones.
    
    Proceso:
    1. Valida entrada con Pydantic
    2. Llama a predict_sales() con los 15 features
    3. Guarda predicción en PostgreSQL para auditoría
    4. Retorna predicción y confirmación
    
    Args:
        request: PredictionRequest validado con los 15 features
    
    Returns:
        dict: Contiene:
            - prediccion_unidades_vendidas: Valor predicho
            - input_data: Los features utilizados
            - message: Confirmación de guardado
    
    Raises:
        HTTPException 500: Si ocurre error en predicción o BD
    """
    try:
        # Convertir Pydantic model a diccionario
        input_data = request.model_dump()
        
        # Realizar predicción
        prediction = predict_sales(input_data)

        # Guardar en PostgreSQL para auditoría
        save_prediction_to_postgres(input_data, prediction)

        return {
            "prediccion_unidades_vendidas": round(prediction, 2),
            "input_data": input_data,
            "message": "Predicción realizada y guardada en PostgreSQL"
        }
    except Exception as e:
        # Retornar error HTTP 500 con detalles
        raise HTTPException(status_code=500, detail=str(e))