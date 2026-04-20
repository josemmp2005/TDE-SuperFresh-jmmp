"""
Generador de datos de prueba coherentes para el sistema de predicción.
Mantiene relaciones lógicas entre tablas y patrones realistas de negocio.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Configuración de rutas
DATA_RAW_PATH = os.path.join(os.path.dirname(__file__), 'data', 'raw')

# SEMILLA para reproducibilidad
np.random.seed(42)

def cargar_datos_actuales():
    """Carga los datos existentes para mantener coherencia."""
    clima = pd.read_csv(os.path.join(DATA_RAW_PATH, 'cliima.csv'))
    productos = pd.read_csv(os.path.join(DATA_RAW_PATH, 'productos.csv'))
    promociones = pd.read_csv(os.path.join(DATA_RAW_PATH, 'promociones.csv'))
    ventas = pd.read_csv(os.path.join(DATA_RAW_PATH, 'ventas.csv'))
    
    return clima, productos, promociones, ventas

def generar_clima(fecha_inicio, fecha_fin, num_tiendas=3):
    """
    Genera datos de clima coherentes.
    - Variación estacional realista
    - Temperatura y lluvia correlacionadas en ciertos períodos
    """
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    datos_clima = []
    
    for i, fecha in enumerate(fechas):
        # Patrón estacional: más frío en invierno (enero-febrero), más cálido en verano (julio-agosto)
        mes = fecha.month
        estacionalidad = 10 + 8 * np.sin(2 * np.pi * (mes - 1) / 12)
        
        # Ruido aleatorio
        temp_aleatoria = estacionalidad + np.random.normal(0, 2.5)
        
        # Lluvia: mayor en invierno y primavera
        lluvia_base = 5 if mes in [12, 1, 2, 3, 4, 5] else 2
        lluvia = max(0, lluvia_base + np.random.normal(0, 3))
        
        datos_clima.append({
            'fecha': fecha.strftime('%Y-%m-%d'),
            'temperatura_media': round(temp_aleatoria, 1),
            'lluvia_mm': round(lluvia, 1)
        })
    
    return pd.DataFrame(datos_clima)

def generar_productos():
    """
    Genera un catálogo más completo y coherente de productos.
    Mantiene diversidad de categorías y precios realistas.
    """
    productos_nuevos = [
        # Categoria: Alimentacion (Lácteos y productos frescos)
        {'id_producto': 103, 'nombre_producto': 'Yogur Natural', 'categoria': 'Alimentacion'},
        {'id_producto': 104, 'nombre_producto': 'Queso Cheddar', 'categoria': 'Alimentacion'},
        {'id_producto': 105, 'nombre_producto': 'Pasta Integral', 'categoria': 'Alimentacion'},
        {'id_producto': 106, 'nombre_producto': 'Aceite de Oliva', 'categoria': 'Alimentacion'},
        {'id_producto': 107, 'nombre_producto': 'Mantequilla Salada', 'categoria': 'Alimentacion'},
        {'id_producto': 108, 'nombre_producto': 'Huevos Frescos (Docena)', 'categoria': 'Alimentacion'},
        {'id_producto': 109, 'nombre_producto': 'Jamón Serrano', 'categoria': 'Alimentacion'},
        {'id_producto': 110, 'nombre_producto': 'Mozzarella Fresca', 'categoria': 'Alimentacion'},
        {'id_producto': 111, 'nombre_producto': 'Yogur Griego', 'categoria': 'Alimentacion'},
        {'id_producto': 112, 'nombre_producto': 'Atún en Lata', 'categoria': 'Alimentacion'},
        {'id_producto': 113, 'nombre_producto': 'Garbanzos Cocidos', 'categoria': 'Alimentacion'},
        {'id_producto': 114, 'nombre_producto': 'Arroz Blanco', 'categoria': 'Alimentacion'},
        {'id_producto': 115, 'nombre_producto': 'Lentejas Rojas', 'categoria': 'Alimentacion'},
        
        # Categoria: Panaderia
        {'id_producto': 201, 'nombre_producto': 'Croissant', 'categoria': 'Panaderia'},
        {'id_producto': 202, 'nombre_producto': 'Donut de Chocolate', 'categoria': 'Panaderia'},
        {'id_producto': 203, 'nombre_producto': 'Galletas Integrales', 'categoria': 'Panaderia'},
        {'id_producto': 204, 'nombre_producto': 'Baguette Francesa', 'categoria': 'Panaderia'},
        {'id_producto': 205, 'nombre_producto': 'Pan de Molde Blanco', 'categoria': 'Panaderia'},
        {'id_producto': 206, 'nombre_producto': 'Pan Integral 500g', 'categoria': 'Panaderia'},
        {'id_producto': 207, 'nombre_producto': 'Medialuna Rellena', 'categoria': 'Panaderia'},
        {'id_producto': 208, 'nombre_producto': 'Magdalena Casera', 'categoria': 'Panaderia'},
        {'id_producto': 209, 'nombre_producto': 'Arepa Precocida', 'categoria': 'Panaderia'},
        
        # Categoria: Bebidas
        {'id_producto': 301, 'nombre_producto': 'Zumo de Naranja', 'categoria': 'Bebidas'},
        {'id_producto': 302, 'nombre_producto': 'Café Premium', 'categoria': 'Bebidas'},
        {'id_producto': 303, 'nombre_producto': 'Té Verde', 'categoria': 'Bebidas'},
        {'id_producto': 304, 'nombre_producto': 'Agua Mineral 1.5L', 'categoria': 'Bebidas'},
        {'id_producto': 305, 'nombre_producto': 'Refresco de Cola', 'categoria': 'Bebidas'},
        {'id_producto': 306, 'nombre_producto': 'Bebida Energética', 'categoria': 'Bebidas'},
        {'id_producto': 307, 'nombre_producto': 'Leche Semidesnatada', 'categoria': 'Bebidas'},
        {'id_producto': 308, 'nombre_producto': 'Horchata de Almendra', 'categoria': 'Bebidas'},
        {'id_producto': 309, 'nombre_producto': 'Vino Tinto Reserva', 'categoria': 'Bebidas'},
        {'id_producto': 310, 'nombre_producto': 'Cerveza Artesana', 'categoria': 'Bebidas'},
        
        # Categoria: Confiteria
        {'id_producto': 401, 'nombre_producto': 'Chocolate Negro', 'categoria': 'Confiteria'},
        {'id_producto': 402, 'nombre_producto': 'Caramelos Variados', 'categoria': 'Confiteria'},
        {'id_producto': 403, 'nombre_producto': 'Turrón de Almendra', 'categoria': 'Confiteria'},
        {'id_producto': 404, 'nombre_producto': 'Bombones Surtidos', 'categoria': 'Confiteria'},
        {'id_producto': 405, 'nombre_producto': 'Chicle de Menta', 'categoria': 'Confiteria'},
        {'id_producto': 406, 'nombre_producto': 'Caramelo Sabor Fresa', 'categoria': 'Confiteria'},
        {'id_producto': 407, 'nombre_producto': 'Nougat Francés', 'categoria': 'Confiteria'},
        
        # Categoria: Higiene y Cuidado Personal
        {'id_producto': 501, 'nombre_producto': 'Detergente Líquido', 'categoria': 'Higiene'},
        {'id_producto': 502, 'nombre_producto': 'Jabón de Manos', 'categoria': 'Higiene'},
        {'id_producto': 503, 'nombre_producto': 'Pasta de Dientes', 'categoria': 'Higiene'},
        {'id_producto': 504, 'nombre_producto': 'Papel Higiénico Pack', 'categoria': 'Higiene'},
        {'id_producto': 505, 'nombre_producto': 'Champú Neutro', 'categoria': 'Higiene'},
    ]
    
    return pd.DataFrame(productos_nuevos)

def generar_promociones(fecha_inicio, fecha_fin, productos_ids, num_tiendas=3):
    """
    Genera promociones coherentes.
    - Promociones estacionales (más descuentos en épocas bajas)
    - Promociones por tienda diferentes
    - Descuentos realistas (5-30%)
    """
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    promociones = []
    
    for fecha in fechas:
        for tienda in range(1, num_tiendas + 1):
            for prod_id in productos_ids:
                # 40% de probabilidad de tener promoción
                if np.random.random() < 0.4:
                    # Descuentos coherentes: 5%, 10%, 15%, 20%, 25%
                    descuento = np.random.choice([5, 10, 15, 20, 25])
                else:
                    descuento = 0
                
                promociones.append({
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'id_tienda': tienda,
                    'id_producto': prod_id,
                    'promocion_activa': 1 if descuento > 0 else 0,
                    'descuento_pct': descuento
                })
    
    return pd.DataFrame(promociones)

def generar_ventas(fecha_inicio, fecha_fin, clima_df, productos_df, promociones_df, num_tiendas=3):
    """
    Genera ventas coherentes considerando:
    - Impacto del clima (lluvia -> menos salidas, compran online o menos)
    - Impacto de promociones (descuento -> más unidades)
    - Estacionalidad por producto y categoría
    - Variedad realista de precios unitarios
    """
    ventas = []
    
    # Precios base realistas por producto
    precio_base = {
        101: 1.50, 102: 2.30, 103: 1.80, 104: 3.50, 105: 1.25, 106: 7.50,
        107: 4.00, 108: 3.20, 109: 8.50, 110: 5.50, 111: 2.50, 112: 1.80, 113: 1.50, 114: 2.00, 115: 1.75,
        201: 2.00, 202: 1.50, 203: 1.75, 204: 3.50, 205: 1.80, 206: 2.20, 207: 2.30, 208: 1.90, 209: 1.60,
        301: 2.50, 302: 3.00, 303: 2.75, 304: 1.20, 305: 1.80, 306: 2.50, 307: 1.40, 308: 3.20, 309: 8.00, 310: 2.80,
        401: 3.25, 402: 1.00, 403: 4.50, 404: 5.00, 405: 0.75, 406: 0.90, 407: 6.00,
        501: 3.50, 502: 2.00, 503: 1.50, 504: 4.00, 505: 3.00
    }
    
    for idx, row_clima in clima_df.iterrows():
        fecha_clima = pd.to_datetime(row_clima['fecha'])
        temperatura = row_clima['temperatura_media']
        lluvia = row_clima['lluvia_mm']
        
        for tienda in range(1, num_tiendas + 1):
            for prod_id in productos_df['id_producto'].values:
                # Obtener promoción del día
                promo_row = promociones_df[
                    (promociones_df['fecha'] == row_clima['fecha']) &
                    (promociones_df['id_tienda'] == tienda) &
                    (promociones_df['id_producto'] == prod_id)
                ]
                
                descuento = promo_row['descuento_pct'].values[0] if len(promo_row) > 0 else 0
                
                # Calcular unidades vendidas con factores:
                unidades_base = np.random.randint(5, 35)
                
                # Factor clima: lluvia reduce ventas no esenciales
                factor_clima = 0.8 if lluvia > 10 else (0.9 if lluvia > 5 else 1.0)
                
                # Factor promoción: descuento aumenta ventas
                factor_promo = 1.0 + (descuento / 100) * 0.5  # 5% desc = 2.5% más unidades
                
                # Factor temperatura: bebidas frías venden más en calor, calientes en frío
                categoria = productos_df[productos_df['id_producto'] == prod_id]['categoria'].values[0]
                if categoria == 'Bebidas':
                    factor_temp = 1.2 if temperatura > 20 else 0.8
                else:
                    factor_temp = 1.0
                
                # Calcular unidades finales
                unidades_vendidas = max(0, int(unidades_base * factor_clima * factor_promo * factor_temp))
                
                # Calcular precio (aplica descuento si existe)
                precio_unitario = precio_base[prod_id]
                precio_con_descuento = precio_unitario * (1 - descuento / 100)
                
                ventas.append({
                    'fecha': row_clima['fecha'],
                    'id_tienda': tienda,
                    'id_producto': prod_id,
                    'unidades_vendidas': unidades_vendidas,
                    'precio_unitario': round(precio_con_descuento, 2)
                })
    
    return pd.DataFrame(ventas)

def main():
    """Ejecuta la generación completa de datos."""
    
    print("=" * 70)
    print("GENERADOR DE DATOS DE PRUEBA CON COHERENCIA")
    print("=" * 70)
    
    # 1. Cargar datos actuales
    print("\n[1] Cargando datos existentes...")
    clima_actual, productos_actual, promo_actual, ventas_actual = cargar_datos_actuales()
    print(f"    [OK] Clima: {len(clima_actual)} registros")
    print(f"    [OK] Productos: {len(productos_actual)} registros")
    print(f"    [OK] Promociones: {len(promo_actual)} registros")
    print(f"    [OK] Ventas: {len(ventas_actual)} registros")
    
    # 2. Generar nuevos datos
    print("\n[2] Generando nuevos datos coherentes...")
    fecha_inicio = '2025-01-03'
    fecha_fin = '2025-03-31'
    
    # Generar datos
    clima_nuevo = generar_clima(fecha_inicio, fecha_fin)
    productos_nuevo = generar_productos()
    
    # Combinar productos (mantener los actuales + agregar nuevos)
    productos_completo = pd.concat([productos_actual, productos_nuevo], ignore_index=True)
    productos_completo = productos_completo.drop_duplicates(subset=['id_producto'])
    
    # Generar promociones y ventas con 3 tiendas
    promo_nueva = generar_promociones(fecha_inicio, fecha_fin, productos_completo['id_producto'].tolist(), num_tiendas=3)
    ventas_nueva = generar_ventas(fecha_inicio, fecha_fin, clima_nuevo, productos_completo, promo_nueva, num_tiendas=3)
    
    print(f"    [OK] Clima: {len(clima_nuevo)} nuevos registros")
    print(f"    [OK] Productos: {len(productos_nuevo)} productos adicionales")
    print(f"    [OK] Promociones: {len(promo_nueva)} registros")
    print(f"    [OK] Ventas: {len(ventas_nueva)} registros")
    
    # 3. Combinar datos
    print("\n[3] Combinando datos...")
    clima_combined = pd.concat([clima_actual, clima_nuevo], ignore_index=True)
    promo_combined = pd.concat([promo_actual, promo_nueva], ignore_index=True)
    ventas_combined = pd.concat([ventas_actual, ventas_nueva], ignore_index=True)
    
    # 4. Guardar archivos actualizados
    print("\n[4] Guardando archivos actualizados...")
    
    clima_combined.to_csv(os.path.join(DATA_RAW_PATH, 'cliima.csv'), index=False)
    productos_completo.to_csv(os.path.join(DATA_RAW_PATH, 'productos.csv'), index=False)
    promo_combined.to_csv(os.path.join(DATA_RAW_PATH, 'promociones.csv'), index=False)
    ventas_combined.to_csv(os.path.join(DATA_RAW_PATH, 'ventas.csv'), index=False)
    
    print(f"    [OK] cliima.csv: {len(clima_combined)} registros")
    print(f"    [OK] productos.csv: {len(productos_completo)} registros")
    print(f"    [OK] promociones.csv: {len(promo_combined)} registros")
    print(f"    [OK] ventas.csv: {len(ventas_combined)} registros")
    
    # 5. Resumen estadístico
    print("\n[5] Resumen estadístico de datos nuevos:")
    print("\n    CLIMA:")
    print(f"      Temperatura: {clima_nuevo['temperatura_media'].min():.1f}C - {clima_nuevo['temperatura_media'].max():.1f}C")
    print(f"      Lluvia: {clima_nuevo['lluvia_mm'].min():.1f}mm - {clima_nuevo['lluvia_mm'].max():.1f}mm")
    
    print("\n    VENTAS:")
    print(f"      Unidades: {ventas_nueva['unidades_vendidas'].min()} - {ventas_nueva['unidades_vendidas'].max()} por registro")
    print(f"      Promedio de unidades: {ventas_nueva['unidades_vendidas'].mean():.1f}")
    print(f"      Precio unitario: ${ventas_nueva['precio_unitario'].min():.2f} - ${ventas_nueva['precio_unitario'].max():.2f}")
    
    print("\n    PROMOCIONES:")
    promo_activas = (promo_nueva['promocion_activa'] == 1).sum()
    print(f"      Promociones activas: {promo_activas} ({100*promo_activas/len(promo_nueva):.1f}%)")
    print(f"      Descuentos: {promo_nueva[promo_nueva['descuento_pct']>0]['descuento_pct'].unique()}")
    
    print("\n" + "=" * 70)
    print("[OK] GENERACION COMPLETADA EXITOSAMENTE")
    print("=" * 70)

if __name__ == "__main__":
    main()
