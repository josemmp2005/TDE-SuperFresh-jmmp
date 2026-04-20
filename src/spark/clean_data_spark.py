"""
Módulo de Limpieza de Datos con Spark

Este módulo implementa las funciones de limpieza y validación para cada tabla:
1. clean_ventas: Elimina duplicados, nulos, valores negativos
2. clean_promociones: Normaliza porcentajes, rellena nulos
3. clean_clima: Valida datos climáticos, rellena ausencias
4. clean_productos: Completa información faltante del catálogo

Los datos limpios se exportan a CSV en la carpeta processed/
"""

from pyspark.sql import functions as F

from src.config import PROCESSED_DATA_DIR
from src.spark.load_data_spark import load_all_data_spark


def ensure_processed_dir():
    """Crea el directorio processed/ si no existe."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_ventas(df):
    """
    Limpia el dataset de ventas.
    
    Operaciones:
    - Convierte fecha a formato DATE
    - Elimina duplicados completamente idénticos
    - Elimina nulos en columnas clave (fecha, tienda, producto, unidades)
    - Filtra unidades_vendidas >= 0
    - Filtra precio_unitario >= 0 (si existe)
    
    Args:
        df: DataFrame de ventas raw
    
    Returns:
        DataFrame de ventas limpiado
    """
    df = (
        df.withColumn("fecha", F.to_date(F.col("fecha")))
          .dropDuplicates()
          .dropna(subset=["fecha", "id_tienda", "id_producto", "unidades_vendidas"])
          .filter(F.col("unidades_vendidas") >= 0)
    )

    if "precio_unitario" in df.columns:
        df = df.filter(F.col("precio_unitario") >= 0)

    return df


def clean_promociones(df):
    """
    Limpia el dataset de promociones.
    
    Operaciones:
    - Convierte fecha a formato DATE
    - Elimina duplicados
    - Rellena descuento_pct nulo con 0
    - Valida que descuento_pct esté en rango [0, 100]
    - Elimina nulos en columnas clave
    
    Args:
        df: DataFrame de promociones raw
    
    Returns:
        DataFrame de promociones limpiado
    """
    df = (
        df.withColumn("fecha", F.to_date(F.col("fecha")))
          .dropDuplicates()
          .dropna(subset=["fecha", "id_tienda", "id_producto", "promocion_activa"])
    )

    if "descuento_pct" in df.columns:
        df = (
            df.withColumn(
                "descuento_pct",
                F.when(F.col("descuento_pct").isNull(), F.lit(0))
                 .otherwise(F.col("descuento_pct"))
            )
            .filter((F.col("descuento_pct") >= 0) & (F.col("descuento_pct") <= 100))
        )

    return df


def clean_clima(df):
    """
    Limpia el dataset de clima.
    
    Operaciones:
    - Convierte fecha a formato DATE
    - Elimina duplicados
    - Rellena lluvia_mm nulo con 0
    - Valida que lluvia_mm >= 0
    - Elimina nulos en fecha
    
    Args:
        df: DataFrame de clima raw
    
    Returns:
        DataFrame de clima limpiado
    """
    df = (
        df.withColumn("fecha", F.to_date(F.col("fecha")))
          .dropDuplicates()
          .dropna(subset=["fecha"])
    )

    if "lluvia_mm" in df.columns:
        df = (
            df.withColumn(
                "lluvia_mm",
                F.when(F.col("lluvia_mm").isNull(), F.lit(0))
                 .otherwise(F.col("lluvia_mm"))
            )
            .filter(F.col("lluvia_mm") >= 0)
        )

    return df


def clean_productos(df):
    """
    Limpia el dataset de productos (catálogo).
    
    Operaciones:
    - Elimina duplicados
    - Elimina nulos en id_producto
    - Rellena nombre_producto nulo con "Desconocido"
    - Rellena categoria nulo con "Sin categoria"
    
    Args:
        df: DataFrame de productos raw
    
    Returns:
        DataFrame de productos limpiado
    """
    df = df.dropDuplicates().dropna(subset=["id_producto"])

    if "nombre_producto" in df.columns:
        df = df.withColumn(
            "nombre_producto",
            F.when(F.col("nombre_producto").isNull(), F.lit("Desconocido"))
             .otherwise(F.col("nombre_producto"))
        )

    if "categoria" in df.columns:
        df = df.withColumn(
            "categoria",
            F.when(F.col("categoria").isNull(), F.lit("Sin categoria"))
             .otherwise(F.col("categoria"))
        )

    return df


def save_df(df, filename):
    """
    Guarda un DataFrame Spark a CSV en la carpeta processed/.
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo sin extensión (ej: "ventas_limpio")
    """
    output_path = PROCESSED_DATA_DIR / f"{filename}.csv"

    pandas_df = df.toPandas()
    pandas_df.to_csv(output_path, index=False)

    print(f"Archivo guardado en: {output_path}")

def main():
    data = load_all_data_spark()
    spark = data["spark"]

    ventas = clean_ventas(data["ventas"])
    promociones = clean_promociones(data["promociones"])
    clima = clean_clima(data["clima"])
    productos = clean_productos(data["productos"])

    print("\n--- VENTAS LIMPIO ---")
    ventas.printSchema()
    ventas.show(5, truncate=False)

    print("\n--- PROMOCIONES LIMPIO ---")
    promociones.printSchema()
    promociones.show(5, truncate=False)

    print("\n--- CLIMA LIMPIO ---")
    clima.printSchema()
    clima.show(5, truncate=False)

    print("\n--- PRODUCTOS LIMPIO ---")
    productos.printSchema()
    productos.show(5, truncate=False)

    ensure_processed_dir()

    save_df(ventas, "ventas_limpio")
    save_df(promociones, "promociones_limpio")
    save_df(clima, "clima_limpio")
    save_df(productos, "productos_limpio")

    spark.stop()


if __name__ == "__main__":
    main()