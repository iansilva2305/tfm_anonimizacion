# Guía de Despliegue: Sistema de Anonimización y Detección de Fraude

Esta guía proporciona instrucciones detalladas para implementar el sistema de anonimización y detección de fraude específicamente optimizado para el MacBook Pro de 14 pulgadas (2023) con chip M2 Pro, 16GB de RAM y macOS 15.4.1.

## Tabla de Contenidos
- [Configuración Optimizada para M2 Pro](#configuración-optimizada-para-m2-pro)
- [Instalación Local Optimizada](#instalación-local-optimizada)
- [Configuraciones de Rendimiento](#configuraciones-de-rendimiento)
- [Almacenamiento Optimizado](#almacenamiento-optimizado)
- [Visualización para Pantalla Retina](#visualización-para-pantalla-retina)
- [Consideraciones de Seguridad](#consideraciones-de-seguridad)
- [Problemas Comunes y Soluciones](#problemas-comunes-y-soluciones)

## Configuración Optimizada para M2 Pro

El MacBook Pro ofrece un excelente rendimiento para este sistema gracias a:

- **Procesador M2 Pro**: 10-12 núcleos de CPU que pueden acelerar procesamiento paralelo
- **16GB de memoria unificada**: Permite cargar datasets medianos-grandes completamente en memoria
- **Arquitectura ARM**: Optimizaciones específicas para mejor rendimiento y eficiencia energética
- **SSD de alta velocidad**: Ideal para operaciones intensivas de E/S

### Rendimiento Esperado

| Operación | Rendimiento en tu MacBook Pro |
|-----------|------------------------------|
| Carga dataset 1M filas (CSV) | ~3-5 segundos |
| Carga dataset 1M filas (Parquet) | ~1-2 segundos |
| Anonimización completa 1M registros | ~10-20 segundos |
| Entrenamiento Random Forest | ~30-60 segundos |
| Evaluación completa (k=10, l=2) | ~2-3 minutos |

## Instalación Local Optimizada

### 1. Configuración Inicial

Abre Terminal.app y ejecuta:

```bash
# Asegurar XCode Command Line Tools (necesario para compilaciones nativas)
xcode-select --install

# Crear directorio de proyecto
mkdir -p ~/Documents/fraud_detection
cd ~/Documents/fraud_detection

# Clonar repositorio
git clone ttps://github.com/iansilva2305/tfm_anonimizacion.git .
```

### 2. Entorno Virtual Optimizado para M2

```bash
# Crear entorno virtual optimizado para ARM
python3 -m venv venv_m2 --system-site-packages

# Activar entorno
source venv_m2/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias optimizadas
pip install -r requirements_apple_silicon.txt
```

### 3. Verificación de Optimizaciones ARM

```bash
# Verificar que NumPy detecta la arquitectura ARM
python -c "import numpy as np; print(f'Arquitectura: {np.show_config()}'); print('Optimizado para ARM: ', 'arm64' in np.__config__.get_info('cpu'))"

# Verificar aceleración Metal para PyTorch
python -c "import torch; print(f'MPS (Metal) disponible: {torch.backends.mps.is_available()}')"
```

### 4. Configurar Jupyter Optimizado

```bash
# Instalar extensiones adicionales para mejor rendimiento
pip install jupyter-resource-usage

# Registrar kernel optimizado para Apple Silicon
python -m ipykernel install --user --name=venv_m2 --display-name="Python (M2 Optimizado)"

# Iniciar JupyterLab con configuración optimizada
jupyter lab --NotebookApp.max_buffer_size=1000000000
```

## Configuraciones de Rendimiento

### 1. Optimizaciones de Paralelización para M2 Pro

Añade al inicio de tus scripts/notebooks:

```python
# Configuración óptima para M2 Pro
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurar paralelización óptima (ajustar según tu modelo específico)
os.environ["OMP_NUM_THREADS"] = "8"  # Ajustar según número de performance cores
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"

# Optimizaciones para pandas
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

# Optimizaciones para scikit-learn
from sklearn.utils import parallel_backend
from sklearn.ensemble import RandomForestClassifier

# Usar modelo Random Forest con todos los cores
def crear_modelo_optimizado(n_estimators=100):
    with parallel_backend('threading', n_jobs=-1):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,  # Usar todos los cores
            verbose=1
        )
```

### 2. Optimización para Lotes de Datos Grandes

Para datasets que excedan la memoria disponible:

```python
# Procesamiento por lotes para datasets grandes
import pandas as pd

# Tamaño de lote optimizado para 16GB RAM
CHUNK_SIZE = 1000000  # 1 millón de filas por lote

# Función para procesar grandes datasets
def procesar_dataset_grande(ruta_archivo, funcion_proceso):
    """
    Procesa un dataset grande en lotes.
    
    Args:
        ruta_archivo: Ruta al archivo CSV/Parquet
        funcion_proceso: Función que procesa cada lote
    """
    resultados = []
    
    # Determinar formato y abrir en modo chunk
    if ruta_archivo.endswith('.csv'):
        reader = pd.read_csv(ruta_archivo, chunksize=CHUNK_SIZE)
    elif ruta_archivo.endswith('.parquet'):
        # Para Parquet, cargar metadatos y filtrar en lotes
        total_rows = pd.read_parquet(ruta_archivo, engine='pyarrow').shape[0]
        for i in range(0, total_rows, CHUNK_SIZE):
            chunk = pd.read_parquet(
                ruta_archivo, 
                engine='pyarrow',
                filters=[('__index_level_0__', '>=', i), 
                        ('__index_level_0__', '<', i + CHUNK_SIZE)]
            )
            resultados.append(funcion_proceso(chunk))
        return resultados
    else:
        raise ValueError("Formato no soportado. Use CSV o Parquet.")
    
    # Procesar cada lote para CSV
    for chunk in reader:
        resultados.append(funcion_proceso(chunk))
    
    return resultados
```

## Almacenamiento Optimizado

### 1. Formato de Archivo Recomendado

Para el MacBook Pro con SSD rápida, Parquet ofrece mejores prestaciones:

```python
# Convertir desde CSV a Parquet optimizado
def optimizar_almacenamiento(csv_path, parquet_path=None):
    """Convierte CSV a Parquet optimizado para M2 Pro"""
    if parquet_path is None:
        parquet_path = csv_path.replace('.csv', '.parquet')
    
    # Leer CSV
    df = pd.read_csv(csv_path)
    
    # Guardar como Parquet con compresión Zstandard (mejor en M2)
    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression='zstd',
        compression_level=3,  # Balance entre velocidad y tamaño
        index=False
    )
    
    print(f"Archivo original: {os.path.getsize(csv_path) / 1_000_000:.2f} MB")
    print(f"Archivo optimizado: {os.path.getsize(parquet_path) / 1_000_000:.2f} MB")
    
    return parquet_path
```

### 2. Estructura de Directorio Recomendada para macOS

```
~/Documents/fraud_detection/
├── data/
│   ├── raw/                 # Datos de entrada sin procesar
│   ├── processed/           # Datos preprocesados
│   └── anonymized/          # Datos anonimizados
├── models/
│   ├── original/            # Modelos entrenados con datos originales
│   └── anonymized/          # Modelos entrenados con datos anonimizados
├── results/
│   ├── metrics/             # Métricas de rendimiento y privacidad
│   ├── visualizations/      # Gráficos y visualizaciones
│   └── reports/             # Informes generados
├── notebooks/               # Jupyter notebooks
└── venv_m2/                 # Entorno virtual optimizado
```

### 3. Backup y Versionado

Aprovecha las funcionalidades de macOS:

```bash
# Crear snapshot con Time Machine (recomendado para hitos importantes)
tmutil localsnapshot

# Alternativa: Usar Git LFS para archivos grandes
brew install git-lfs
git lfs install
git lfs track "*.parquet" "*.pkl" "*.joblib"
```

## Visualización para Pantalla Retina

Tu MacBook Pro cuenta con una pantalla Liquid Retina XDR de alta resolución. Optimiza las visualizaciones:

### 1. Configuración Matplotlib para Pantalla Retina

```python
# Configuración para visualizaciones de alta calidad
def configurar_visualizacion_retina():
    """Configura matplotlib para aprovechamiento óptimo de pantalla Retina"""
    import matplotlib.pyplot as plt
    
    # Alta resolución para pantalla Retina
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 300
    
    # Tamaño optimizado para pantalla de 14"
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Estilo moderno
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Fuentes nítidas
    plt.rcParams['font.family'] = 'SF Pro Display'  # Fuente nativa de macOS
    plt.rcParams['font.size'] = 11
    
    # Colores optimizados para Retina
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#0984e3', '#d63031', '#00b894', '#e84393', 
        '#fdcb6e', '#6c5ce7', '#00cec9', '#e17055'
    ])
    
    return plt
```

### 2. Dashboard Streamlit Optimizado

```python
# Configuración de Streamlit para Retina
# Guardar como .streamlit/config.toml
"""
[theme]
primaryColor="#0984e3"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f7f7f7"
textColor="#262730"
font="SF Pro Display"

[server]
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[client]
showErrorDetails = true
toolbarMode = "minimal"
"""

# Lanzar dashboard con optimizaciones para Retina
# streamlit run --theme.base "light" dashboard.py
```

## Consideraciones de Seguridad

macOS 15.4.1 incluye funcionalidades de seguridad avanzadas. Aprovéchalas:

### 1. Almacenamiento Seguro

```python
# Almacenamiento seguro aprovechando Keychain de macOS
def guardar_credencial_segura(servicio, usuario, clave):
    """Guarda credencial en Keychain de macOS"""
    import subprocess
    
    cmd = [
        'security', 'add-generic-password',
        '-s', servicio,
        '-a', usuario,
        '-w', clave,
        '-U'  # Actualizar si ya existe
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Credencial para {servicio} guardada en Keychain")

def obtener_credencial_segura(servicio, usuario):
    """Recupera credencial desde Keychain de macOS"""
    import subprocess
    
    cmd = [
        'security', 'find-generic-password',
        '-s', servicio,
        '-a', usuario,
        '-w'  # Mostrar solo password
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()
```

### 2. FileVault para Datos Sensibles

Si trabajas con datos realmente sensibles, asegúrate de tener FileVault activo:

1. Abre Preferencias del Sistema
2. Ve a "Privacidad y Seguridad" → "FileVault"
3. Activa FileVault si no está habilitado

## Problemas Comunes y Soluciones

### 1. Error: "Memoria insuficiente"

**Solución optimizada para M2 Pro**:

```python
# Reducir precisión para ahorrar memoria
def reducir_precision_numericas(df):
    """Optimiza uso de memoria reduciendo precisión de columnas numéricas"""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['int64']).columns:
        # Determinar rango y usar el tipo más pequeño posible
        col_min, col_max = df[col].min(), df[col].max()
        
        if col_min >= 0:
            if col_max < 256:
                df[col] = df[col].astype('uint8')
            elif col_max < 65536:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 128:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32768:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    
    return df

# Ejemplo de uso
# df = reducir_precision_numericas(df)
# print(f"Memoria usada: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
```

### 2. Error: "Procesamiento lento durante anonimización"

**Solución optimizada para M2**:

```python
# Paralelización de función de hash para M2 Pro
import hashlib
from joblib import Parallel, delayed

def anonimizar_ids_paralelo(df, columna, n_jobs=-1):
    """Anonimización paralela usando todos los núcleos del M2 Pro"""
    # Función para aplicar a cada valor
    def hash_sha256(value):
        if not isinstance(value, str):
            value = str(value)
        return hashlib.sha256(value.encode()).hexdigest()
    
    # Obtener valores únicos para reducir hashing redundante
    valores_unicos = df[columna].unique()
    
    # Crear diccionario de mapeo en paralelo
    hashed_values = Parallel(n_jobs=n_jobs)(
        delayed(hash_sha256)(val) for val in valores_unicos
    )
    
    # Crear diccionario de mapeo
    mapeo = dict(zip(valores_unicos, hashed_values))
    
    # Aplicar mapeo (mucho más rápido que apply)
    df[columna] = df[columna].map(mapeo)
    
    return df
```

### 3. Error: "Visualizaciones lentas o borrosas"

**Solución para pantalla Retina**:

```python
# Renderizado eficiente para pantalla Retina
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def crear_grafico_optimizado(ancho=12, alto=8, dpi=200):
    """Crea gráficos optimizados para pantalla Retina"""
    # Usar backend Agg para mejor rendimiento
    fig = Figure(figsize=(ancho, alto), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Optimizaciones para rendimiento
    fig.set_tight_layout(True)
    
    return fig, ax

# Ejemplo de uso:
# fig, ax = crear_grafico_optimizado()
# ax.plot(range(10), range(10))
# ax.set_title("Gráfico Optimizado para Retina")
# fig.savefig("grafico_hd.png")
```

### 4. Error: "Modelo demasiado lento al ejecutar predicciones"

**Solución optimizada para M2 Pro**:

```python
# Optimización de modelo Random Forest para inferencia
import joblib
from sklearn.ensemble import RandomForestClassifier

def optimizar_modelo_inferencia(modelo, archivo_salida=None):
    """Optimiza un modelo para inferencia rápida en M2 Pro"""
    # Si no es un Random Forest, devolver sin cambios
    if not isinstance(modelo, RandomForestClassifier):
        return modelo
    
    # 1. Reducir precisión de los árboles
    for estimator in modelo.estimators_:
        for tree in [estimator.tree_]:
            # Reducir precisión de los valores de umbral
            tree.threshold = tree.threshold.astype('float32')
            
            # Reducir precisión de los valores de hojas
            if hasattr(tree, 'value'):
                tree.value = tree.value.astype('float32')
    
    # 2. Guardar modelo optimizado (si se especifica ruta)
    if archivo_salida:
        # Usar compresión zstandard (rápida en ARM)
        joblib.dump(modelo, archivo_salida, compress=('zstd', 3))
        print(f"Modelo optimizado guardado en {archivo_salida}")
    
    return modelo

# Ejemplo de uso:
# modelo_optimizado = optimizar_modelo_inferencia(modelo, "modelo_optimizado_m2.joblib")
```

### 5. Error: "Valores k muy grandes reducen demasiado la precisión"

**Solución adaptada para mejor rendimiento**:

```python
# Anonimización adaptativa optimizada para M2
def anonimizar_adaptativo(df, columna, k_min=5, k_max=20, precision_min=0.75, n_jobs=-1):
    """
    Encuentra el valor óptimo de k que mantiene precisión aceptable.
    Aprovecha paralelización en MBP M2 Pro.
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Preprocesar datos para evaluación
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Modelo base sin anonimizar
    modelo_base = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=n_jobs)
    precision_base = np.mean(cross_val_score(
        modelo_base, X_train, y_train, cv=3, n_jobs=n_jobs
    ))
    
    # Probar diferentes valores de k
    resultados = []
    k_valores = range(k_min, k_max + 1, 5)
    
    for k in k_valores:
        # Aplicar k-anonimato
        df_anon = aplicar_k_anonimato(df.copy(), columna, k)
        
        # Preparar datos anonimizados
        X_anon = df_anon.drop(['isFraud'], axis=1)
        y_anon = df_anon['isFraud']
        
        # Dividir datos
        X_train_anon, X_test_anon, y_train_anon, y_test_anon = train_test_split(
            X_anon, y_anon, test_size=0.3, random_state=42
        )
        
        # Evaluar precisión
        modelo_anon = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=n_jobs)
        precision_anon = np.mean(cross_val_score(
            modelo_anon, X_train_anon, y_train_anon, cv=3, n_jobs=n_jobs
        ))
        
        # Calcular pérdida relativa
        perdida = (precision_base - precision_anon) / precision_base
        
        resultados.append({
            'k': k,
            'precision': precision_anon,
            'perdida_porcentaje': perdida * 100
        })
    
    # Encontrar el mayor k con pérdida aceptable
    resultados_aceptables = [r for r in resultados if r['perdida_porcentaje'] <= (1 - precision_min) * 100]
    
    if resultados_aceptables:
        # Ordenar por k (descendente) para obtener el mayor k aceptable
        mejor_k = sorted(resultados_aceptables, key=lambda x: x['k'], reverse=True)[0]
        return aplicar_k_anonimato(df.copy(), columna, mejor_k['k']), mejor_k['k']
    else:
        # Si ninguno cumple, usar el de menor pérdida
        mejor_resultado = min(resultados, key=lambda x: x['perdida_porcentaje'])
        return aplicar_k_anonimato(df.copy(), columna, mejor_resultado['k']), mejor_resultado['k']
```

## Optimizaciones para Batería

Tu MacBook Pro puede funcionar con batería durante sesiones de análisis. Optimiza para mejorar la autonomía:

```python
# Configuración para optimizar la duración de batería
def modo_ahorro_energia():
    """Configura opciones para reducir consumo energético"""
    import os
    
    # Reducir hilos para mejor eficiencia energética
    os.environ["OMP_NUM_THREADS"] = "4"  # Usar menos cores
    os.environ["MKL_NUM_THREADS"] = "4"
    
    # Configuración para scikit-learn
    from sklearn.utils import parallel_backend
    
    # Usar backend threading con menos hilos
    parallel_backend('threading', n_jobs=4)
    
    # Evitar cálculos en segundo plano
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 100  # Menor resolución
    
    print("Modo ahorro de energía activado. Rendimiento reducido para extender batería.")
```

---

Esta guía de despliegue está específicamente optimizada para el uso del entorno Apple Silicon, lo que permite un despliegue más eficiente y rápido. MacBook Pro con M2 Pro y macOS 15.4.1. Aprovecha las capacidades del hardware para obtener el mejor rendimiento en el sistema de anonimización y detección de fraude.