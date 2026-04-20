"""
Módulo de Configuración Central - SuperFresh Sales Prediction

Este módulo centraliza todas las rutas de directorios, ubicaciones de archivos y
parámetros de conexión a la base de datos para el proyecto completo.

Permite mantener la configuración en un único lugar y facilita cambios globales
sin modificar múltiples archivos.
"""

from pathlib import Path
import os

# ============================================================================
# DIRECTORIOS Y RUTAS DEL PROYECTO
# ============================================================================

# Directorio base del proyecto (raíz del src)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios principales
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # Datos originales sin procesar
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Datos limpios y features engineered

MODELS_DIR = BASE_DIR / "models"  # Modelos entrenados serializados
REPORTS_DIR = BASE_DIR / "reports"  # Reportes y métricas del modelo

# ============================================================================
# ARCHIVOS DE DATOS RAW (INPUT)
# ============================================================================

VENTAS_FILE = RAW_DATA_DIR / "ventas.csv"  # Datos de ventas diarias
PROMOCIONES_FILE = RAW_DATA_DIR / "promociones.csv"  # Datos de promociones
CLIMA_FILE = RAW_DATA_DIR / "clima.csv"  # Datos climáticos
PRODUCTOS_FILE = RAW_DATA_DIR / "productos.csv"  # Catálogo de productos

# ============================================================================
# ARCHIVOS PROCESADOS Y MODELOS
# ============================================================================

DATASET_MODELO_FILE = PROCESSED_DATA_DIR / "dataset_modelo.csv"  # Dataset final para entrenamiento
MODEL_FILE = MODELS_DIR / "random_forest_sales.pkl"  # Modelo RandomForest entrenado

# ============================================================================
# CONFIGURACIÓN DE BASE DE DATOS POSTGRESQL
# ============================================================================

# Parámetros de conexión con valores por defecto (pueden ser sobrescritos por variables de entorno)
DB_HOST = os.getenv("DB_HOST", "localhost")  # Servidor PostgreSQL
DB_PORT = os.getenv("DB_PORT", "5432")  # Puerto PostgreSQL
DB_NAME = os.getenv("DB_NAME", "superfresh")  # Nombre de la base de datos
DB_USER = os.getenv("DB_USER", "admin")  # Usuario de conexión
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")  # Contraseña de conexión

# URL de conexión para SQLAlchemy (utilizada por el ORM)
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)