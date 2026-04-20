"""
Módulo de Sesión Spark - Configuración centralizada de Apache Spark

Este módulo proporciona una sesión Spark reutilizable configurada para:
- Procesar en modo local usando todos los núcleos disponibles
- Reducir verbosidad de logs (solo errores)
- Facilitar el procesamiento distribuido de datos masivos
"""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "SuperFreshBigData") -> SparkSession:
    """
    Obtiene o crea una sesión Spark configurada.
    
    Args:
        app_name (str): Nombre de la aplicación Spark. Por defecto "SuperFreshBigData".
    
    Returns:
        SparkSession: Sesión Spark configurada lista para usar.
    
    Configuración:
        - master("local[*]"): Ejecuta localmente usando TODOS los núcleos del sistema
        - Log Level ERROR: Solo muestra errores, no información ni warnings
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Usar todos los núcleos disponibles
        .getOrCreate()
    )

    # Reducir verbosidad del log a solo errores
    spark.sparkContext.setLogLevel("ERROR")
    return spark