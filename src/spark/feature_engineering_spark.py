"""
Módulo de Ingeniería de Características con Spark

Este es el módulo más complejo del pipeline. Realiza:
1. Carga de datos limpios desde CSV
2. Combinación (join) de 4 tablas (ventas, promociones, clima, productos)
3. Relleno de valores faltantes
4. Creación de características temporales (año, mes, día, día de semana)
5. Creación de características de serie temporal (lag_1, lag_7, media_movil_7)
6. Eliminación de filas sin historial (primeros 7 días de cada tienda-producto)
7. Exportación del dataset final listo para entrenamiento

El resultado es un CSV con 19 columnas que será usado para entrenar el modelo.
"""

from pyspark.sql import Window
from pyspark.sql import functions as F

from src.config import PROCESSED_DATA_DIR, DATASET_MODELO_FILE
from src.spark.spark_session import get_spark_session


# Rutas de archivos limpios (salida de clean_data_spark.py)
VENTAS_CLEAN_FILE = PROCESSED_DATA_DIR / "ventas_limpio.csv"
PROMOCIONES_CLEAN_FILE = PROCESSED_DATA_DIR / "promociones_limpio.csv"
CLIMA_CLEAN_FILE = PROCESSED_DATA_DIR / "clima_limpio.csv"
PRODUCTOS_CLEAN_FILE = PROCESSED_DATA_DIR / "productos_limpio.csv"


def load_clean_data_spark():
    """
    Carga los 4 archivos CSV limpios usando Spark.
    
    Returns:
        tuple: (spark_session, df_ventas, df_promociones, df_clima, df_productos)
    """
    spark = get_spark_session("SuperFreshFeatureEngineering")

    # Cargar y convertir fechas a formato DATE
    ventas = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(VENTAS_CLEAN_FILE))
        .withColumn("fecha", F.to_date("fecha"))
    )

    promociones = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(PROMOCIONES_CLEAN_FILE))
        .withColumn("fecha", F.to_date("fecha"))
    )

    clima = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(CLIMA_CLEAN_FILE))
        .withColumn("fecha", F.to_date("fecha"))
    )

    productos = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(PRODUCTOS_CLEAN_FILE))
    )

    return spark, ventas, promociones, clima, productos


def merge_data(ventas, promociones, clima, productos):
    """
    Combina las 4 tablas mediante LEFT JOINs sucesivos.
    
    Orden de combinación:
    1. ventas LEFT JOIN promociones (en: fecha, id_tienda, id_producto)
    2. resultado LEFT JOIN clima (en: fecha)
    3. resultado LEFT JOIN productos (en: id_producto)
    
    El LEFT JOIN asegura que todas las ventas se conserven,
    incluso si no hay promoción o datos climáticos disponibles.
    
    Args:
        ventas, promociones, clima, productos: DataFrames a combinar
    
    Returns:
        DataFrame con todas las columnas combinadas
    """
    # Combinar ventas con promociones por fecha, tienda y producto
    df = ventas.join(
        promociones,
        on=["fecha", "id_tienda", "id_producto"],
        how="left"  # Mantener todas las ventas, aunque no haya promoción
    )

    # Combinar resultado con clima por fecha
    df = df.join(
        clima,
        on="fecha",
        how="left"  # Mantener todos los registros, aunque no haya datos climáticos
    )

    # Combinar resultado con productos por id_producto
    df = df.join(
        productos,
        on="id_producto",
        how="left"  # Mantener todos los registros, aunque falta información del producto
    )

    return df


def fill_missing_values(df):
    """
    Rellena valores faltantes (NULL) con estrategias apropiadas.
    
    Estrategias:
    - promocion_activa, descuento_pct, lluvia_mm: Rellenar con 0 (sin promoción, sin lluvia)
    - temperatura_media: Rellenar con la mediana del dataset (valor central)
    
    Args:
        df: DataFrame con posibles valores nulos
    
    Returns:
        DataFrame sin valores nulos
    """
    # Rellenar 0 en promociones y lluvia
    df = (
        df.withColumn("promocion_activa", F.coalesce(F.col("promocion_activa"), F.lit(0)))
          .withColumn("descuento_pct", F.coalesce(F.col("descuento_pct"), F.lit(0)))
          .withColumn("lluvia_mm", F.coalesce(F.col("lluvia_mm"), F.lit(0)))
    )

    # Para temperatura, usar la mediana como valor de relleno (más robusto que la media)
    if "temperatura_media" in df.columns:
        median_temp = df.approxQuantile("temperatura_media", [0.5], 0.01)[0]
        df = df.withColumn(
            "temperatura_media",
            F.coalesce(F.col("temperatura_media"), F.lit(median_temp))
        )

    return df


def create_time_features(df):
    """
    Crea características temporales a partir de la fecha.
    
    Nuevas columnas:
    - anio: Año (ej: 2026)
    - mes: Mes del año (1-12)
    - dia: Día del mes (1-31)
    - dia_semana: Día de la semana (1=domingo, 7=sábado)
    - es_fin_de_semana: Binario (1 si sábado/domingo, 0 si no)
    
    Estas características capturan patrones estacionales y semanales en las ventas.
    
    Args:
        df: DataFrame con columna 'fecha'
    
    Returns:
        DataFrame con características temporales añadidas
    """
    df = (
        df.withColumn("anio", F.year("fecha"))
          .withColumn("mes", F.month("fecha"))
          .withColumn("dia", F.dayofmonth("fecha"))
          .withColumn("dia_semana", F.dayofweek("fecha"))  # 1=domingo, 7=sábado
          .withColumn(
              "es_fin_de_semana",
              F.when(F.col("dia_semana").isin([1, 7]), 1).otherwise(0)  # 1 si es fin de semana
          )
    )

    return df


def create_lag_features(df):
    """
    Crea características de serie temporal (lags) y media móvil.
    
    Estos features capturan la autocorrelación temporal en las ventas.
    Se calculan POR TIENDA Y PRODUCTO (Window partitionBy).
    
    Nuevas columnas:
    - lag_1: Unidades vendidas del día anterior (mismo tienda-producto)
    - lag_7: Unidades vendidas hace 7 días (mismo tienda-producto)
    - media_movil_7: Promedio de unidades de los últimos 7 días (excluye el actual)
    
    Estas características son CRÍTICAS para predicciones precisas porque:
    - lag_1 captura tendencias corto plazo
    - lag_7 captura patrones semanales (lunes vs viernes, etc)
    - media_movil_7 suaviza ruido y captura tendencias
    
    Args:
        df: DataFrame con 'unidades_vendidas', ordenado por (tienda, producto, fecha)
    
    Returns:
        DataFrame con características lag añadidas
    """
    # Ventana ordenada por fecha, particionada por tienda y producto
    window_spec = Window.partitionBy("id_tienda", "id_producto").orderBy("fecha")
    
    # Ventana para media móvil: últimos 7 días (excluye el actual)
    rolling_window = window_spec.rowsBetween(-7, -1)

    df = (
        # lag_1: valor del día anterior
        df.withColumn("lag_1", F.lag("unidades_vendidas", 1).over(window_spec))
        # lag_7: valor de hace 7 días
          .withColumn("lag_7", F.lag("unidades_vendidas", 7).over(window_spec))
        # media_movil_7: promedio de los 7 días anteriores
          .withColumn("media_movil_7", F.avg("unidades_vendidas").over(rolling_window))
    )

    return df


def drop_rows_without_history(df):
    """
    Elimina las filas que NO tienen historial completo.
    
    Las primeras 7 filas de cada (tienda, producto) no tienen lag_7 ni media_movil_7
    porque no hay suficientes días anteriores. Estas filas no pueden usarse para entrenar.
    
    Args:
        df: DataFrame con features lag que pueden tener NULLs
    
    Returns:
        DataFrame solo con filas que tienen lag_1, lag_7 y media_movil_7 válidos
    """
    return df.dropna(subset=["lag_1", "lag_7", "media_movil_7"])


def save_dataset(df):
    """
    Guarda el dataset final a CSV listo para entrenamiento.
    
    Args:
        df: DataFrame final con todas las características
    """
    output_path = DATASET_MODELO_FILE
    pandas_df = df.toPandas()
    pandas_df.to_csv(output_path, index=False)
    print(f"Dataset guardado en: {output_path}")


def save_dataset(df):
    output_path = DATASET_MODELO_FILE
    pandas_df = df.toPandas()
    pandas_df.to_csv(output_path, index=False)
    print(f"Dataset guardado en: {output_path}")


def main():
    spark, ventas, promociones, clima, productos = load_clean_data_spark()

    df = merge_data(ventas, promociones, clima, productos)
    df = fill_missing_values(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = drop_rows_without_history(df)

    print("\n--- DATASET FINAL ---")
    df.printSchema()
    df.show(10, truncate=False)

    save_dataset(df)

    spark.stop()


if __name__ == "__main__":
    main()