# SuperFresh Sales Prediction - Sistema de Predicción de Ventas

## Descripción General del Proyecto

Este es un sistema integrado de **predicción de ventas para una cadena de supermercados (SuperFresh)** que combina:

- **Procesamiento de datos masivos** con Apache Spark
- **Machine Learning** con scikit-learn (RandomForest)
- **API REST** en tiempo real con FastAPI
- **Base de datos PostgreSQL** para auditoría y análisis

El objetivo es predecir con precisión cuántas unidades de cada producto se venderán en cada tienda, permitiendo optimizar inventario y operaciones.

---

## Flujo General del Proyecto

```
ETAPA 1: INGESTA Y LIMPIEZA DE DATOS
│
├─ Entrada: 4 archivos CSV (ventas, productos, promociones, clima)
│
├─ load_data_spark.py
│  └─ Carga los 4 CSVs con Apache Spark
│
├─ clean_data_spark.py
│  ├─ Elimina duplicados y nulos
│  ├─ Valida rangos de valores
│  ├─ Rellena ausencias con estrategias apropiadas
│  └─ Exporta 4 CSVs limpios
│
└─ Salida: data/processed/ (4 CSVs limpios)


ETAPA 2: INGENIERÍA DE CARACTERÍSTICAS
│
├─ Entrada: 4 CSVs limpios
│
├─ feature_engineering_spark.py
│  ├─ Combina 4 tablas mediante LEFT JOINs
│  ├─ Crea características temporales (año, mes, día, día_semana, fin_semana)
│  ├─ Crea características de serie temporal:
│  │  ├─ lag_1: Ventas del día anterior
│  │  ├─ lag_7: Ventas hace 7 días
│  │  └─ media_movil_7: Promedio últimos 7 días
│  ├─ Rellena valores faltantes (0 para promociones, mediana para temperatura)
│  └─ Elimina primeros 7 días (sin historial suficiente)
│
└─ Salida: data/processed/dataset_modelo.csv (19 columnas, listo para entrenar)


ETAPA 3: ENTRENAMIENTO DEL MODELO
│
├─ Entrada: dataset_modelo.csv
│
├─ train.py
│  ├─ Selecciona 15 features más relevantes
│  ├─ Divide 80% entrenamiento / 20% prueba
│  ├─ Entrena RandomForestRegressor (100 árboles)
│  ├─ Evalúa con MAE, RMSE, R²
│  └─ Guarda modelo en pickle + métricas en JSON y PostgreSQL
│
└─ Salida: 
    ├─ models/random_forest_sales.pkl (modelo entrenado)
    ├─ reports/metrics/model_metrics.json
    └─ PostgreSQL: tabla metricas_modelo


ETAPA 4: INFERENCIA EN PRODUCCIÓN
│
├─ Entrada: HTTP POST /predict (15 features)
│
├─ src/api/main.py (FastAPI)
│  └─ Recibe request JSON con features
│
├─ predict.py
│  ├─ Carga modelo pickle
│  ├─ Valida features
│  └─ Realiza predicción
│
├─ postgres.py
│  └─ Guarda predicción en tabla predicciones (auditoría)
│
└─ Salida: JSON con predicción + confirmación
```

---

## Estructura del Proyecto

```
prediccion/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── setup_postgres.sql                 # Script de inicialización BD
│
├── data/                              # Datos del proyecto
│   ├── raw/                           # Datos originales (input)
│   │   ├── ventas.csv
│   │   ├── productos.csv
│   │   ├── promociones.csv
│   │   └── clima.csv
│   │
│   └── processed/                     # Datos procesados
│       ├── ventas_limpio.csv
│       ├── productos_limpio.csv
│       ├── promociones_limpio.csv
│       ├── clima_limpio.csv
│       └── dataset_modelo.csv         # ⭐ Datos finales para ML
│
├── models/                            # Modelos entrenados
│   └── random_forest_sales.pkl        # ⭐ Modelo serializado
│
├── reports/                           # Reportes y métricas
│   └── metrics/
│       └── model_metrics.json         # MAE, RMSE, R²
│
└── src/                               # Código fuente
    ├── __init__.py
    ├── config.py                      # Configuración centralizada
    │
    ├── spark/                         # Pipeline de datos
    │   ├── spark_session.py           # Sesión Spark centralizada
    │   ├── load_data_spark.py         # Carga CSVs raw
    │   ├── clean_data_spark.py        # Limpieza de datos
    │   └── feature_engineering_spark.py # Ingeniería de features
    │
    ├── model/                         # Machine Learning
    │   ├── train.py                   # Entrenamiento del modelo
    │   └── predict.py                 # Predicción/Inferencia
    │
    ├── storage/                       # Base de datos
    │   └── postgres.py                # Gestión PostgreSQL
    │
    └── api/                           # API REST
        └── main.py                    # FastAPI endpoints
```

---

## Cómo Ejecutar el Proyecto

### 1. Preparación del Entorno

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración de PostgreSQL

```bash
# Crear usuario y base de datos (ejecutar en PostgreSQL)
psql -U postgres < setup_postgres.sql

# O ejecutar manualmente:
# CREATE USER admin WITH PASSWORD 'admin';
# CREATE DATABASE superfresh OWNER admin;
# GRANT ALL PRIVILEGES ON DATABASE superfresh TO admin;
```

### 3. Ejecutar Pipeline de Datos

```bash
# Etapa 1: Cargar y limpiar datos
python -m src.spark.clean_data_spark

# Etapa 2: Ingeniería de características
python -m src.spark.feature_engineering_spark

# Etapa 3: Entrenar modelo
python -m src.model.train

# Crear tablas en PostgreSQL (primera vez)
python -m src.storage.postgres
```

### 4. Lanzar API en Producción

```bash
# Opción 1: Desarrollo (con auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Opción 2: Producción
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Características del Modelo (15 Features)

El modelo RandomForest utiliza 15 características:

### Features de Negocio (5)
- `id_tienda`: ID de la tienda (1, 2, 3...)
- `id_producto`: ID del producto (101, 102...)
- `precio_unitario`: Precio en euros
- `promocion_activa`: 1 si hay promoción, 0 si no
- `descuento_pct`: Porcentaje de descuento (0-100)

### Features Climáticos (2)
- `temperatura_media`: Temperatura en °C
- `lluvia_mm`: Precipitación en milímetros

### Features Temporales (5)
- `anio`: Año (2026)
- `mes`: Mes (1-12)
- `dia`: Día del mes (1-31)
- `dia_semana`: Día de la semana (1=domingo, 7=sábado)
- `es_fin_de_semana`: 1 si es sábado/domingo, 0 si no

### Features de Serie Temporal (3) ⭐ **MÁS IMPORTANTES**
- `lag_1`: Unidades vendidas del **día anterior**
- `lag_7`: Unidades vendidas **hace 7 días**
- `media_movil_7`: Promedio de ventas de los **últimos 7 días**

> Los features de serie temporal son críticos porque capturan autocorrelación temporal y patrones cíclicos en las ventas.

---

## Métricas del Modelo

El modelo se evalúa con 3 métricas en el conjunto de prueba:

### MAE (Mean Absolute Error)
- Error promedio en unidades vendidas
- **Ejemplo**: MAE = 5 significa ±5 unidades en promedio
- **Interpretación**: Directa y comprensible en el dominio del negocio

### RMSE (Root Mean Squared Error)
- Penaliza más los errores grandes que los pequeños
- **Comparación con MAE**:
  - Si RMSE ≈ MAE → errores consistentes
  - Si RMSE >> MAE → hay algunos errores muy grandes

### R² (Coeficiente de Determinación)
- Proporción de varianza explicada por el modelo
- **Rango**: 0 a 1 (o negativo si es peor que predecir media)
- **R² = 1.0** → predicción perfecta
- **R² = 0.0** → no mejor que predecir la media

---

## 🔌 API REST - Endpoints

### 1. GET `/` - Verificación de Salud

```bash
curl http://localhost:8000/
```

Respuesta:
```json
{
  "message": "API de predicción de ventas de SuperFresh activa"
}
```

### 2. POST `/predict` - Realizar Predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "dia_semana": 3,
    "es_fin_de_semana": 0,
    "lag_1": 35,
    "lag_7": 30,
    "media_movil_7": 32.4
  }'
```

Respuesta:
```json
{
  "prediccion_unidades_vendidas": 42.5,
  "input_data": {...},
  "message": "Predicción realizada y guardada en PostgreSQL"
}
```

---

## 🗄️ Base de Datos PostgreSQL

### Tabla 1: dataset_modelo
Almacena el dataset de entrenamiento con todas las características (19 columnas).

```sql
SELECT * FROM dataset_modelo LIMIT 5;
-- Útil para auditoria y reentrenamiento
```

### Tabla 2: predicciones
Registra CADA predicción realizada por la API (auditoría completa).

```sql
SELECT * FROM predicciones 
  ORDER BY fecha_prediccion DESC 
  LIMIT 10;
-- Historial completo de predicciones
```

### Tabla 3: metricas_modelo
Historial de evaluación del modelo (para detectar degradación).

```sql
SELECT * FROM metricas_modelo 
  ORDER BY fecha_entrenamiento DESC;
-- MAE, RMSE, R² a lo largo del tiempo
```

---

## 🔧 Dependencias Clave

| Librería | Versión | Propósito |
|----------|---------|----------|
| **pandas** | - | Manipulación de datos (DataFrames) |
| **pyspark** | - | Procesamiento distribuido de datos |
| **scikit-learn** | - | Machine Learning (RandomForest, métricas) |
| **joblib** | - | Serialización de modelos |
| **sqlalchemy** | - | ORM y conexión a BD |
| **psycopg2-binary** | - | Driver PostgreSQL |
| **fastapi** | - | Framework API REST |
| **uvicorn** | - | Servidor ASGI para FastAPI |

---

## Ejemplos de Uso

### Ejemplo 1: Predicción Básica
```python
from src.model.predict import predict_sales

prediccion = predict_sales({
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
    "dia_semana": 3,
    "es_fin_de_semana": 0,
    "lag_1": 35,
    "lag_7": 30,
    "media_movil_7": 32.4,
})

print(f"Predicción: {prediccion:.2f} unidades")
```

### Ejemplo 2: Entrenar Nuevo Modelo
```python
from src.model.train import (
    load_dataset, select_features, split_data,
    train_model, evaluate_model, save_model, save_metrics
)

df = load_dataset()
X, y, features = select_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

print(f"MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, R²: {metrics['R2']}")

save_model(model, features)
save_metrics(metrics)
```

### Ejemplo 3: Análisis de Datos
```python
import pandas as pd

# Cargar dataset de entrenamiento
df = pd.read_csv("data/processed/dataset_modelo.csv", parse_dates=["fecha"])

# Ventas por tienda
print(df.groupby("id_tienda")["unidades_vendidas"].mean())

# Impacto de promociones
print(df[df["promocion_activa"] == 1]["unidades_vendidas"].mean())
print(df[df["promocion_activa"] == 0]["unidades_vendidas"].mean())
```

---

## Optimizaciones y Mejoras Futuras

1. **Tuning de Hiperparámetros**
   - Usar GridSearchCV o RandomizedSearchCV para optimizar RandomForest
   - Experimentar con otros algoritmos (XGBoost, LightGBM)

2. **Manejo de Estacionalidad**
   - Agregar features de eventos (vacaciones, promociones especiales)
   - Usar modelos ARIMA o Prophet para series temporales

3. **Monitoreo en Producción**
   - Alertas si MAE/RMSE se degrada
   - Reentrenamiento automático periódicamente

4. **Escalabilidad**
   - Distribuir Spark en cluster (no solo local)
   - Usar docker para containerizar la API

5. **Features Adicionales**
   - Competencia (precios de competidores)
   - Inventario disponible
   - Demografía de la tienda

---

## Troubleshooting

### Error: "No se encontró el modelo"
```
FileNotFoundError: No se encontró el modelo entrenado: models/random_forest_sales.pkl
```
**Solución**: Ejecutar `python -m src.model.train` primero

### Error: "Conexión a PostgreSQL rechazada"
```
psycopg2.OperationalError: could not connect to server
```
**Solución**: 
- Verificar que PostgreSQL esté corriendo
- Verificar credenciales en config.py
- Ejecutar setup_postgres.sql

### Error: "Faltan columnas para predecir"
```
ValueError: Faltan columnas para predecir: ['lag_1', 'lag_7', ...]
```
**Solución**: Asegurar que el JSON enviado a /predict incluya todos los 15 features

---

## Notas de Desarrollo

- El proyecto usa **Spark localmente** (`master=local[*]`) para desarrollo
- En producción, considerar Spark en cluster distribuido
- RandomForest es robusto pero puede sobreajustarse en datos pequeños
- Los lags (lag_1, lag_7) requieren al menos 8 días de historial por tienda-producto
- Temperatura faltante se rellena con **mediana** (más robusta que media)

---

## Contacto y Documentación

Para más información sobre los módulos individuales, consulta:
- [src/config.py](src/config.py) - Configuración centralizada
- [src/model/train.py](src/model/train.py) - Lógica de entrenamiento
- [src/api/main.py](src/api/main.py) - Documentación de API en `/docs`

---

## Checklist de Implementación

- [x] Módulos de carga y limpieza de datos (Spark)
- [x] Ingeniería de características (features de serie temporal)
- [x] Entrenamiento de modelo (RandomForest)
- [x] API REST (FastAPI)
- [x] Integración PostgreSQL
- [x] Documentación completa en español
- [ ] Tests unitarios
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Monitoreo en producción

---

#   T D E - S u p e r F r e s h - j m m p  
 