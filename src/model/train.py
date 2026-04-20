"""
Módulo de Entrenamiento del Modelo - Machine Learning

Este módulo entrena un modelo RandomForestRegressor para predecir unidades vendidas.

Etapas:
1. Carga dataset_modelo.csv generado por feature engineering
2. Selecciona 15 features relevantes
3. Divide datos 80/20 (entrenamiento/prueba)
4. Entrena RandomForest con 100 árboles
5. Evalúa usando MAE, RMSE y R²
6. Guarda modelo en pickle y métricas en JSON y PostgreSQL
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.storage.postgres import save_metrics_to_postgres

from src.config import DATASET_MODELO_FILE, MODEL_FILE, REPORTS_DIR


METRICS_FILE = REPORTS_DIR / "metrics" / "model_metrics.json"


def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset procesado y listo para entrenar.
    
    Returns:
        pd.DataFrame: Dataset con todas las características
    
    Raises:
        FileNotFoundError: Si no existe dataset_modelo.csv
    """
    if not DATASET_MODELO_FILE.exists():
        raise FileNotFoundError(f"No se encontró el dataset del modelo: {DATASET_MODELO_FILE}")

    # parse_dates convierte la columna 'fecha' a tipo datetime
    df = pd.read_csv(DATASET_MODELO_FILE, parse_dates=["fecha"])
    return df


def select_features(df: pd.DataFrame):
    """
    Selecciona los 15 features más importantes para el modelo.
    
    Features de negocio:
    - id_tienda, id_producto, precio_unitario, promocion_activa, descuento_pct
    
    Features climáticos:
    - temperatura_media, lluvia_mm
    
    Features temporales:
    - anio, mes, dia, dia_semana, es_fin_de_semana
    
    Features de serie temporal:
    - lag_1, lag_7, media_movil_7
    
    Args:
        df: DataFrame con todas las columnas disponibles
    
    Returns:
        tuple: (X features, y target, lista de nombres de features)
    
    Raises:
        ValueError: Si falta la columna objetivo 'unidades_vendidas'
    """
    feature_columns = [
        "id_tienda",
        "id_producto",
        "precio_unitario",
        "promocion_activa",
        "descuento_pct",
        "temperatura_media",
        "lluvia_mm",
        "anio",
        "mes",
        "dia",
        "dia_semana",
        "es_fin_de_semana",
        "lag_1",
        "lag_7",
        "media_movil_7",
    ]

    # Filtrar solo los features que existen en el dataset
    existing_features = [col for col in feature_columns if col in df.columns]

    target_column = "unidades_vendidas"

    if target_column not in df.columns:
        raise ValueError("La columna objetivo 'unidades_vendidas' no existe en el dataset.")

    X = df[existing_features].copy()
    y = df[target_column].copy()

    return X, y, existing_features


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Divide datos en conjunto de entrenamiento (80%) y prueba (20%).
    
    Args:
        X: Features
        y: Target (variable a predecir)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Entrena un modelo RandomForestRegressor.
    
    Hiperparámetros:
    - n_estimators=100: 100 árboles en el bosque
    - max_depth=10: Profundidad máxima de cada árbol (evita sobreajuste)
    - random_state=42: Seed para reproducibilidad
    - n_jobs=-1: Usa todos los núcleos del procesador
    
    RandomForest es un ensemble de árboles de decisión que:
    - Reduce varianza mediante bootstrap y promediado
    - Maneja no linealidades bien
    - Es robusto a valores atípicos
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        RandomForestRegressor: Modelo entrenado
    """
    model = RandomForestRegressor(
        n_estimators=100,  # 100 árboles
        max_depth=10,  # Limitar profundidad para evitar sobreajuste
        random_state=42,  # Reproducibilidad
        n_jobs=-1  # Usar todos los cores disponibles
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evalúa el modelo en el conjunto de prueba usando 3 métricas.
    
    Métricas:
    - MAE (Mean Absolute Error): Error promedio en unidades vendidas
      * Interpretable: si MAE=5, el error promedio es ±5 unidades
    
    - RMSE (Root Mean Squared Error): Penaliza más los errores grandes
      * Valor similar a MAE = errores consistentes
      * RMSE >> MAE = hay algunos errores muy grandes
    
    - R² (Coeficiente de determinación): Proporción de varianza explicada
      * 1.0 = predicción perfecta
      * 0.0 = predecir media = explicación nula
      * Negativo = peor que predecir media
    
    Args:
        model: Modelo entrenado
        X_test: Features de prueba
        y_test: Target de prueba (valores reales)
    
    Returns:
        dict: Diccionario con MAE, RMSE, R² redondeados a 4 decimales
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }

    return metrics


def save_model(model, feature_columns):
    """
    Guarda el modelo entrenado y sus features en pickle (joblib).
    
    El archivo pickle contiene un diccionario con:
    - "model": El RandomForestRegressor entrenado
    - "features": Lista de nombres de features que el modelo espera
    
    Args:
        model: RandomForestRegressor entrenado
        feature_columns: Lista de nombres de features
    """
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "features": feature_columns
    }

    joblib.dump(payload, MODEL_FILE)
    print(f"Modelo guardado en: {MODEL_FILE}")


def save_metrics(metrics: dict):
    """
    Guarda las métricas en JSON para registro histórico.
    
    Args:
        metrics: Diccionario con MAE, RMSE, R²
    """
    metrics_dir = METRICS_FILE.parent
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Métricas guardadas en: {METRICS_FILE}")


def main():
    df = load_dataset()
    X, y, feature_columns = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Entrenando modelo...")
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    print("\n--- MÉTRICAS DEL MODELO ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    save_model(model, feature_columns)
    save_metrics(metrics)
    save_metrics_to_postgres(metrics)


if __name__ == "__main__":
    main()