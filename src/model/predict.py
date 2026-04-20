"""
Módulo de Predicción - Realiza inferencia con el modelo entrenado

Este módulo carga el modelo RandomForest previamente entrenado y realiza
predicciones de unidades vendidas dado un conjunto de features.

Utiliza joblib para deserializar el modelo y pandas para preparar los datos.
"""

import joblib
import pandas as pd

from src.config import MODEL_FILE


def load_model():
    """
    Carga el modelo RandomForest entrenado y sus features asociados.
    
    Returns:
        tuple: (modelo RandomForest, lista de nombres de features)
    
    Raises:
        FileNotFoundError: Si el archivo del modelo no existe
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {MODEL_FILE}")

    # joblib deserializa el modelo y features guardados en pickle
    payload = joblib.load(MODEL_FILE)

    model = payload["model"]  # Modelo RandomForestRegressor
    features = payload["features"]  # Lista de nombres de features esperados

    return model, features


def prepare_input_data(input_data: dict, feature_columns: list) -> pd.DataFrame:
    """
    Prepara los datos de entrada en formato DataFrame con validación.
    
    Args:
        input_data (dict): Diccionario con los features de predicción
        feature_columns (list): Lista de nombres de features que el modelo espera
    
    Returns:
        pd.DataFrame: DataFrame con una fila conteniendo los features en el orden correcto
    
    Raises:
        ValueError: Si faltan columnas requeridas en los datos de entrada
    """
    # Convertir diccionario a DataFrame (una fila)
    df = pd.DataFrame([input_data])

    # Validar que todos los features esperados estén presentes
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas para predecir: {missing_columns}")

    # Seleccionar solo los features necesarios en el orden correcto
    df = df[feature_columns]
    return df


def predict_sales(input_data: dict) -> float:
    """
    Realiza predicción de unidades vendidas.
    
    Args:
        input_data (dict): Diccionario con los 15 features de predicción:
            - id_tienda: ID de la tienda
            - id_producto: ID del producto
            - precio_unitario: Precio del producto en €
            - promocion_activa: 1 si hay promoción activa, 0 si no
            - descuento_pct: Porcentaje de descuento (0-100)
            - temperatura_media: Temperatura promedio en grados Celsius
            - lluvia_mm: Precipitación en milímetros
            - anio, mes, dia: Componentes de la fecha
            - dia_semana: Día de la semana (1=domingo, 7=sábado)
            - es_fin_de_semana: 1 si es sábado/domingo, 0 si no
            - lag_1: Unidades vendidas del día anterior
            - lag_7: Unidades vendidas hace 7 días
            - media_movil_7: Promedio de ventas de los últimos 7 días
    
    Returns:
        float: Predicción de unidades vendidas (puede ser decimal)
    """
    # Cargar modelo y obtener lista de features esperados
    model, feature_columns = load_model()
    
    # Preparar datos en formato compatible con el modelo
    input_df = prepare_input_data(input_data, feature_columns)

    # Realizar predicción (model.predict retorna array, [0] para obtener el valor)
    prediction = model.predict(input_df)[0]
    return float(prediction)


def main():
    """
    Función de ejemplo para pruebas locales del módulo.
    Realiza una predicción de prueba con datos de ejemplo.
    """
    sample_input = {
        "id_tienda": 1,
        "id_producto": 101,
        "precio_unitario": 2.5,
        "promocion_activa": 1,
        "descuento_pct": 10,
        "temperatura_media": 18.5,
        "lluvia_mm": 0.0,
        "anio": 2026,
        "mes": 4,
        "dia": 20,
        "dia_semana": 0,
        "es_fin_de_semana": 0,
        "lag_1": 35,
        "lag_7": 30,
        "media_movil_7": 32.4,
    }

    prediction = predict_sales(sample_input)
    print(f"Predicción de unidades vendidas: {prediction:.2f}")


if __name__ == "__main__":
    main()