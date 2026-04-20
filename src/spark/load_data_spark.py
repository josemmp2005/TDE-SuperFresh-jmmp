"""
Módulo de Carga de Datos con Spark - Lee datos CSV raw

Este módulo carga los 4 archivos CSV principales usando Apache Spark:
1. ventas.csv: Registros de ventas diarias
2. productos.csv: Catálogo de productos
3. promociones.csv: Datos de promociones
4. clima.csv: Datos climáticos

Utiliza inferencia automática de esquema para detectar tipos de datos.
"""

from src.config import (
    VENTAS_FILE,
    PROMOCIONES_FILE,
    CLIMA_FILE,
    PRODUCTOS_FILE,
)
from src.spark.spark_session import get_spark_session


def load_csv_spark(spark, path):
    """
    Carga un archivo CSV usando Spark con inferencia de esquema.
    
    Args:
        spark: Sesión Spark activa
        path: Ruta del archivo CSV
    
    Returns:
        DataFrame de Spark con los datos del CSV
    """
    return (
        spark.read
        .option("header", True)  # Primera fila contiene nombres de columnas
        .option("inferSchema", True)  # Detectar tipos de datos automáticamente
        .csv(str(path))
    )


def load_all_data_spark():
    """
    Carga todos los archivos CSV raw necesarios para el proyecto.
    
    Returns:
        dict: Diccionario con sesión Spark y DataFrames:
            - spark: Sesión Spark (para usarla después o cerrarla)
            - ventas: DataFrame con datos de ventas
            - promociones: DataFrame con datos de promociones
            - clima: DataFrame con datos climáticos
            - productos: DataFrame con catálogo de productos
    """
    spark = get_spark_session()

    # Cargar cada dataset usando Spark
    ventas = load_csv_spark(spark, VENTAS_FILE)
    promociones = load_csv_spark(spark, PROMOCIONES_FILE)
    clima = load_csv_spark(spark, CLIMA_FILE)
    productos = load_csv_spark(spark, PRODUCTOS_FILE)

    return {
        "spark": spark,
        "ventas": ventas,
        "promociones": promociones,
        "clima": clima,
        "productos": productos,
    }


def main():
    data = load_all_data_spark()

    for name in ["ventas", "promociones", "clima", "productos"]:
        print(f"\n--- {name.upper()} ---")
        data[name].printSchema()
        data[name].show(5, truncate=False)

    data["spark"].stop()


if __name__ == "__main__":
    main()