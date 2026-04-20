"""
Módulo de Almacenamiento en PostgreSQL

Este módulo gestiona todas las operaciones de base de datos:
1. Crear tablas (dataset_modelo, predicciones, metricas_modelo)
2. Cargar dataset de entrenamiento
3. Guardar predicciones realizadas por la API
4. Guardar métricas de evaluación del modelo

PostgreSQL es la base de datos centralizada para auditoría y análisis histórico.
"""

from pathlib import Path
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DATASET_MODELO_FILE

# Configurar encoding UTF-8 para Windows (importante para caracteres especiales)
os.environ['PGCLIENTENCODING'] = 'UTF8'
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'


def get_engine():
    """
    Crea y retorna un engine SQLAlchemy conectado a PostgreSQL.
    
    Configuración:
    - pool_pre_ping=True: Verifica conexión antes de usarla (evita conexiones rotas)
    - pool_recycle=3600: Recicla conexiones cada hora (importante en producción)
    - echo=False: No mostrar SQL generado
    
    Returns:
        Engine SQLAlchemy configurado para PostgreSQL
    """
    connection_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    engine = create_engine(
        connection_url,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    
    return engine


def create_tables():
    """
    Crea las 3 tablas principales si no existen.
    
    Tablas:
    1. dataset_modelo: Almacena el dataset de entrenamiento con todas las características
    2. predicciones: Registra cada predicción realizada por la API (auditoría)
    3. metricas_modelo: Historial de métricas de evaluación del modelo
    """
    engine = get_engine()

    # Tabla 1: Dataset de entrenamiento
    create_dataset_table = """
    CREATE TABLE IF NOT EXISTS dataset_modelo (
        id SERIAL PRIMARY KEY,
        fecha DATE,
        id_tienda INTEGER,
        id_producto INTEGER,
        unidades_vendidas DOUBLE PRECISION,
        precio_unitario DOUBLE PRECISION,
        promocion_activa INTEGER,
        descuento_pct DOUBLE PRECISION,
        temperatura_media DOUBLE PRECISION,
        lluvia_mm DOUBLE PRECISION,
        nombre_producto TEXT,
        categoria TEXT,
        anio INTEGER,
        mes INTEGER,
        dia INTEGER,
        dia_semana INTEGER,
        es_fin_de_semana INTEGER,
        lag_1 DOUBLE PRECISION,
        lag_7 DOUBLE PRECISION,
        media_movil_7 DOUBLE PRECISION
    );
    """

    # Tabla 2: Registro de predicciones (para auditoría y análisis)
    create_predicciones_table = """
    CREATE TABLE IF NOT EXISTS predicciones (
        id SERIAL PRIMARY KEY,
        fecha_prediccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        fecha_objetivo DATE,
        id_tienda INTEGER,
        id_producto INTEGER,
        prediccion_unidades DOUBLE PRECISION,
        precio_unitario DOUBLE PRECISION,
        promocion_activa INTEGER,
        descuento_pct DOUBLE PRECISION,
        temperatura_media DOUBLE PRECISION,
        lluvia_mm DOUBLE PRECISION,
        anio INTEGER,
        mes INTEGER,
        dia INTEGER,
        dia_semana INTEGER,
        es_fin_de_semana INTEGER,
        lag_1 DOUBLE PRECISION,
        lag_7 DOUBLE PRECISION,
        media_movil_7 DOUBLE PRECISION
    );
    """

    # Tabla 3: Métricas de evaluación del modelo (histórico)
    create_metricas_table = """
    CREATE TABLE IF NOT EXISTS metricas_modelo (
        id SERIAL PRIMARY KEY,
        fecha_entrenamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        algoritmo TEXT,
        mae DOUBLE PRECISION,
        rmse DOUBLE PRECISION,
        r2 DOUBLE PRECISION
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_dataset_table))
        conn.execute(text(create_predicciones_table))
        conn.execute(text(create_metricas_table))

    print("Tablas creadas correctamente en PostgreSQL.")


def load_dataset_to_postgres():
    """
    Carga el dataset_modelo.csv completo a la tabla dataset_modelo.
    
    Este dataset contiene todos los datos de entrenamiento con características engineered.
    Útil para reentrenamiento, auditoría y análisis posteriores.
    
    Raises:
        FileNotFoundError: Si no existe dataset_modelo.csv
    """
    if not Path(DATASET_MODELO_FILE).exists():
        raise FileNotFoundError(f"No se encontró el dataset: {DATASET_MODELO_FILE}")

    engine = get_engine()
    df = pd.read_csv(DATASET_MODELO_FILE)

    # Usar pandas para insertar el dataframe (más simple que SQL)
    df.to_sql(
        "dataset_modelo",
        con=engine,
        if_exists="append",  # Añadir a la tabla existente
        index=False
    )

    print(f"Dataset cargado en PostgreSQL: {len(df)} filas insertadas.")

    
def save_prediction_to_postgres(input_data: dict, prediction: float):
    """
    Guarda cada predicción realizada por la API en la tabla predicciones.
    
    Esto proporciona auditoría completa de:
    - Qué se predijo
    - Cuándo se hizo la predicción
    - Con qué features
    - Cuál fue el resultado
    
    Args:
        input_data (dict): Los 15 features usados para la predicción
        prediction (float): El valor predicho de unidades_vendidas
    """
    engine = get_engine()

    insert_query = """
    INSERT INTO predicciones (
        fecha_objetivo,
        id_tienda,
        id_producto,
        prediccion_unidades,
        precio_unitario,
        promocion_activa,
        descuento_pct,
        temperatura_media,
        lluvia_mm,
        anio,
        mes,
        dia,
        dia_semana,
        es_fin_de_semana,
        lag_1,
        lag_7,
        media_movil_7
    )
    VALUES (
        :fecha_objetivo,
        :id_tienda,
        :id_producto,
        :prediccion_unidades,
        :precio_unitario,
        :promocion_activa,
        :descuento_pct,
        :temperatura_media,
        :lluvia_mm,
        :anio,
        :mes,
        :dia,
        :dia_semana,
        :es_fin_de_semana,
        :lag_1,
        :lag_7,
        :media_movil_7
    );
    """

    # Construir fecha en formato YYYY-MM-DD
    payload = {
        "fecha_objetivo": f"{input_data['anio']}-{input_data['mes']:02d}-{input_data['dia']:02d}",
        "id_tienda": input_data["id_tienda"],
        "id_producto": input_data["id_producto"],
        "prediccion_unidades": float(prediction),
        "precio_unitario": input_data["precio_unitario"],
        "promocion_activa": input_data["promocion_activa"],
        "descuento_pct": input_data["descuento_pct"],
        "temperatura_media": input_data["temperatura_media"],
        "lluvia_mm": input_data["lluvia_mm"],
        "anio": input_data["anio"],
        "mes": input_data["mes"],
        "dia": input_data["dia"],
        "dia_semana": input_data["dia_semana"],
        "es_fin_de_semana": input_data["es_fin_de_semana"],
        "lag_1": input_data["lag_1"],
        "lag_7": input_data["lag_7"],
        "media_movil_7": input_data["media_movil_7"],
    }

    with engine.begin() as conn:
        conn.execute(text(insert_query), payload)

    print("Predicción guardada en PostgreSQL.")


def save_metrics_to_postgres(metrics: dict, algorithm: str = "RandomForestRegressor"):
    """
    Guarda las métricas de evaluación del modelo en la tabla metricas_modelo.
    
    Permite mantener histórico de desempeño del modelo a lo largo del tiempo.
    Útil para detección de degradación del modelo.
    
    Args:
        metrics (dict): Diccionario con MAE, RMSE, R2
        algorithm (str): Nombre del algoritmo usado (default: RandomForestRegressor)
    """
    engine = get_engine()

    insert_query = """
    INSERT INTO metricas_modelo (
        algoritmo,
        mae,
        rmse,
        r2
    )
    VALUES (
        :algoritmo,
        :mae,
        :rmse,
        :r2
    );
    """

    payload = {
        "algoritmo": algorithm,
        "mae": float(metrics["MAE"]),
        "rmse": float(metrics["RMSE"]),
        "r2": float(metrics["R2"]),
    }

    with engine.begin() as conn:
        conn.execute(text(insert_query), payload)

    print("Métricas guardadas en PostgreSQL.")




def create_tables():
    engine = get_engine()

    create_dataset_table = """
    CREATE TABLE IF NOT EXISTS dataset_modelo (
        id SERIAL PRIMARY KEY,
        fecha DATE,
        id_tienda INTEGER,
        id_producto INTEGER,
        unidades_vendidas DOUBLE PRECISION,
        precio_unitario DOUBLE PRECISION,
        promocion_activa INTEGER,
        descuento_pct DOUBLE PRECISION,
        temperatura_media DOUBLE PRECISION,
        lluvia_mm DOUBLE PRECISION,
        nombre_producto TEXT,
        categoria TEXT,
        anio INTEGER,
        mes INTEGER,
        dia INTEGER,
        dia_semana INTEGER,
        es_fin_de_semana INTEGER,
        lag_1 DOUBLE PRECISION,
        lag_7 DOUBLE PRECISION,
        media_movil_7 DOUBLE PRECISION
    );
    """

    create_predicciones_table = """
    CREATE TABLE IF NOT EXISTS predicciones (
        id SERIAL PRIMARY KEY,
        fecha_prediccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        fecha_objetivo DATE,
        id_tienda INTEGER,
        id_producto INTEGER,
        prediccion_unidades DOUBLE PRECISION,
        precio_unitario DOUBLE PRECISION,
        promocion_activa INTEGER,
        descuento_pct DOUBLE PRECISION,
        temperatura_media DOUBLE PRECISION,
        lluvia_mm DOUBLE PRECISION,
        anio INTEGER,
        mes INTEGER,
        dia INTEGER,
        dia_semana INTEGER,
        es_fin_de_semana INTEGER,
        lag_1 DOUBLE PRECISION,
        lag_7 DOUBLE PRECISION,
        media_movil_7 DOUBLE PRECISION
    );
    """

    create_metricas_table = """
    CREATE TABLE IF NOT EXISTS metricas_modelo (
        id SERIAL PRIMARY KEY,
        fecha_entrenamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        algoritmo TEXT,
        mae DOUBLE PRECISION,
        rmse DOUBLE PRECISION,
        r2 DOUBLE PRECISION
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_dataset_table))
        conn.execute(text(create_predicciones_table))
        conn.execute(text(create_metricas_table))

    print("Tablas creadas correctamente.")


def load_dataset_to_postgres():
    if not Path(DATASET_MODELO_FILE).exists():
        raise FileNotFoundError(f"No se encontró el dataset: {DATASET_MODELO_FILE}")

    engine = get_engine()
    df = pd.read_csv(DATASET_MODELO_FILE)

    df.to_sql(
        "dataset_modelo",
        con=engine,
        if_exists="append",
        index=False
    )

    print(f"Dataset cargado en PostgreSQL: {len(df)} filas insertadas.")
    
def save_prediction_to_postgres(input_data: dict, prediction: float):
    engine = get_engine()

    insert_query = """
    INSERT INTO predicciones (
        fecha_objetivo,
        id_tienda,
        id_producto,
        prediccion_unidades,
        precio_unitario,
        promocion_activa,
        descuento_pct,
        temperatura_media,
        lluvia_mm,
        anio,
        mes,
        dia,
        dia_semana,
        es_fin_de_semana,
        lag_1,
        lag_7,
        media_movil_7
    )
    VALUES (
        :fecha_objetivo,
        :id_tienda,
        :id_producto,
        :prediccion_unidades,
        :precio_unitario,
        :promocion_activa,
        :descuento_pct,
        :temperatura_media,
        :lluvia_mm,
        :anio,
        :mes,
        :dia,
        :dia_semana,
        :es_fin_de_semana,
        :lag_1,
        :lag_7,
        :media_movil_7
    );
    """

    payload = {
        "fecha_objetivo": f"{input_data['anio']}-{input_data['mes']:02d}-{input_data['dia']:02d}",
        "id_tienda": input_data["id_tienda"],
        "id_producto": input_data["id_producto"],
        "prediccion_unidades": float(prediction),
        "precio_unitario": input_data["precio_unitario"],
        "promocion_activa": input_data["promocion_activa"],
        "descuento_pct": input_data["descuento_pct"],
        "temperatura_media": input_data["temperatura_media"],
        "lluvia_mm": input_data["lluvia_mm"],
        "anio": input_data["anio"],
        "mes": input_data["mes"],
        "dia": input_data["dia"],
        "dia_semana": input_data["dia_semana"],
        "es_fin_de_semana": input_data["es_fin_de_semana"],
        "lag_1": input_data["lag_1"],
        "lag_7": input_data["lag_7"],
        "media_movil_7": input_data["media_movil_7"],
    }

    with engine.begin() as conn:
        conn.execute(text(insert_query), payload)

    print("Predicción guardada en PostgreSQL.")

def save_metrics_to_postgres(metrics: dict, algorithm: str = "RandomForestRegressor"):
    engine = get_engine()

    insert_query = """
    INSERT INTO metricas_modelo (
        algoritmo,
        mae,
        rmse,
        r2
    )
    VALUES (
        :algoritmo,
        :mae,
        :rmse,
        :r2
    );
    """

    payload = {
        "algoritmo": algorithm,
        "mae": float(metrics["MAE"]),
        "rmse": float(metrics["RMSE"]),
        "r2": float(metrics["R2"]),
    }

    with engine.begin() as conn:
        conn.execute(text(insert_query), payload)

    print("Métricas guardadas en PostgreSQL.")


def main():
    create_tables()
    load_dataset_to_postgres()


if __name__ == "__main__":
    main()