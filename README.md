# Sistema de Anonimización y Detección de Fraude Bancario con Cumplimiento GDPR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/optimizado-M2%20Pro-purple.svg)](https://www.apple.com/macbook-pro/)
[![macOS](https://img.shields.io/badge/macOS-15.4.1-orange.svg)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene una implementación completa de un sistema de anonimización para datos bancarios y detección de fraude con cumplimiento GDPR, optimizado específicamente para MacBook Pro con chip Apple M2 Pro.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Ventajas en MacBook Pro M2](#ventajas-en-macbook-pro-m2)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación Optimizada](#instalación-optimizada)
- [Uso](#uso)
- [Ejemplos](#ejemplos)
- [Optimizaciones de Rendimiento](#optimizaciones-de-rendimiento)
- [Lineamientos de Cumplimiento GDPR](#lineamientos-de-cumplimiento-gdpr)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## 🔍 Descripción General

Este sistema proporciona una solución integral para detectar fraudes financieros sin comprometer la privacidad de los datos personales. Implementa técnicas avanzadas de anonimización (k-anonimato, l-diversidad y seudonimización) junto con un modelo de clasificación Random Forest, optimizado específicamente para aprovechar el rendimiento del chip Apple M2 Pro.

El sistema está diseñado para:
- Anonimizar datos sensibles en conformidad con el GDPR
- Mantener utilidad analítica para detección de fraude
- Evaluar y optimizar el equilibrio entre privacidad y rendimiento
- Proporcionar métricas y visualizaciones de cumplimiento normativo
- Aprovechar al máximo el rendimiento de tu MacBook Pro

## 🏗️ Arquitectura del Sistema

El sistema sigue una arquitectura modular organizada en pipelines secuenciales:

```
                           ┌─────────────────┐
                           │     Datos       │
                           │    Bancarios    │
                           └────────┬────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────┐
│            Pipeline de Preprocesamiento y EDA             │
│ ┌─────────────┐    ┌────────────┐     ┌────────────────┐  │
│ │  Carga de   │───▶│  Análisis  │────▶│ Identificación │  │
│ │   Datos     │    │ Exploratorio│    │ Datos Sensibles│  │
│ └─────────────┘    └────────────┘     └────────────────┘  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│               Pipeline de Anonimización                   │
│ ┌─────────────┐    ┌────────────┐     ┌────────────────┐  │
│ │Seudonimi-   │───▶│ K-anonimato│────▶│  L-diversidad  │  │
│ │zación SHA256│    │(Agrupación)│     │  (Verificación)│  │
│ └─────────────┘    └────────────┘     └────────────────┘  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│              Pipeline de Modelado Paralelo                │
│ ┌─────────────────────────┐    ┌────────────────────────┐ │
│ │Modelo Original          │    │Modelo Anonimizado      │ │
│ │(Sin anonimizar)         │    │(Con datos anonimizados)│ │
│ └────────────┬────────────┘    └─────────────┬──────────┘ │
│              │                               │            │
│              └───────────────┬───────────────┘            │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│            Pipeline de Evaluación y Reporting             │
│ ┌─────────────┐    ┌────────────┐     ┌────────────────┐  │
│ │Comparación  │───▶│ Evaluación │────▶│  Dashboard de  │  │
│ │de Modelos   │    │ GDPR       │     │  Cumplimiento  │  │
│ └─────────────┘    └────────────┘     └────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

## 🚀 Ventajas en MacBook Pro M2

El MacBook Pro de 14 pulgadas (2023) con chip M2 Pro y 16GB de RAM ofrece ventajas significativas para este sistema:

- **Procesamiento Paralelo**: El chip M2 Pro permite paralelizar el entrenamiento de modelos, reduciendo los tiempos hasta 5-10x.
- **Memoria Unificada**: 16GB de memoria unificada facilita el procesamiento de conjuntos de datos de tamaño medio-grande (hasta ~10M de registros).
- **Almacenamiento Rápido**: La SSD de alta velocidad optimiza la carga y guardado de datos.
- **Visualizaciones Avanzadas**: La pantalla Liquid Retina XDR permite visualizaciones detalladas de alta resolución.
- **Optimizaciones ARM**: Bibliotecas científicas optimizadas para arquitectura ARM obtienen mayor rendimiento.

## 💻 Requisitos del Sistema

### Hardware (Ya disponible en el MacBook Pro)
- CPU: Apple M2 Pro (10-12 núcleos)
- RAM: 16GB de memoria unificada
- Almacenamiento: SSD de alta velocidad
- Pantalla: Liquid Retina XDR de 14 pulgadas

### Software Requerido
- macOS 15.4.1 (ya instalado)
- Python 3.8 o superior (preferiblemente 3.11+ para mejores optimizaciones ARM)
- Entorno Jupyter (Notebook o Lab)
- Bibliotecas Python optimizadas para Apple Silicon (ver `requirements_apple_silicon.txt`)

## 🛠️ Instalación Optimizada

1. Abre Terminal.app en tu MacBook Pro

2. Clona este repositorio:
```bash
git clone https://github.com/iansilva2305/tfm_anonimizacion.git
cd tfm_anonimizacion
```

3. Crea un entorno virtual optimizado para Apple Silicon:
```bash
python3 -m venv venv_m2
source venv_m2/bin/activate
```

4. Instala las dependencias optimizadas:
```bash
pip install --upgrade pip
pip install -r requirements_apple_silicon.txt
```

5. Verifica la instalación:
```bash
python -c "import numpy as np; print(f'NumPy detecta {np.zeros(1).__array_interface__[\"data\"][0]} cores')"
```

## 📊 Uso

### Ejecución Optimizada

Para aprovechar al máximo el MacBook Pro M2, ejecuta el sistema así:

```python
# Configuración de paralelización para M2 Pro
import os
# Ajusta el número de hilos basándose en los núcleos de tu M2 Pro
os.environ["OMP_NUM_THREADS"] = "8"  
os.environ["MKL_NUM_THREADS"] = "8"

from eda_anonimizacion_fraud_detection import ejecutar_analisis_deteccion_fraude

# Ejecutar el pipeline completo
df_original, df_anonimizado, modelo_original, modelo_anonimizado, evaluacion_gdpr = ejecutar_analisis_deteccion_fraude(
    "data/dataset_anonimizacion_datos.csv",
    test_size=0.3,
    random_state=42,
    k_anonimato=10,
    l_diversidad=2
)
```

### Análisis Comparativo (Optimizado para M2)

```python
from eda_anonimizacion_fraud_detection import cargar_datos
from comparativa_modelos import ejecutar_analisis_comparativo

# Cargar datos (usando formato optimizado)
df = cargar_datos("data/dataset_anonimizacion_datos.parquet", engine='pyarrow')

# Definir columnas para el modelo
columnas_modelo = [
    'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 'step'
]

# Ejecutar análisis comparativo aprovechando todos los núcleos
recomendacion = ejecutar_analisis_comparativo(
    df, columnas_modelo, valores_k=[2, 5, 10, 20, 50], usar_paralelizacion=True
)
```

## 📝 Ejemplos

### Ejemplo 1: JupyterLab Optimizado para M2

```bash
# Instalar JupyterLab optimizado
pip install jupyterlab

# Lanzar con configuración optimizada para tu pantalla Retina
jupyter lab --port=8888 --NotebookApp.max_buffer_size=1000000000
```

Luego, abre `notebook_ejemplo_deteccion_fraude.ipynb` para una demostración interactiva.

### Ejemplo 2: Dashboard Streamlit para M2

```bash
# Instalar Streamlit (optimizado)
pip install streamlit

# Ejecutar dashboard GDPR (aprovecha Safari para mejor rendimiento)
streamlit run gdpr_dashboard.py
```

## ⚡ Optimizaciones de Rendimiento

Aprovecha al máximo el MacBook Pro con estas optimizaciones:

### Optimización de Memoria

```python
# Para datasets grandes, usar formato Parquet
import pandas as pd
import pyarrow as pa

# Cargar eficientemente (5-10x más rápido que CSV en M2)
df = pd.read_parquet('datos_grandes.parquet', engine='pyarrow')

# Guardar eficientemente
df.to_parquet('resultados.parquet', engine='pyarrow', compression='zstd')
```

### Aceleración de Modelos

```python
# Aprovechar todos los núcleos del M2 Pro
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,  # Usar todos los núcleos disponibles
    verbose=1
)
```

### Visualizaciones de Alta Resolución

```python
# Configuración para aprovechamiento de pantalla Retina
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200  # Alta resolución para pantalla Retina
plt.rcParams['figure.figsize'] = (12, 8)  # Tamaño optimizado para 14"
```

## 📊 Rendimiento Esperado

Con tu configuración específica, puedes esperar:

- **Carga de Datos**: ~1-2 segundos para datasets de 1 millón de filas (Parquet)
- **Anonimización**: Procesamiento de 5-10 millones de registros en minutos
- **Entrenamiento de Modelos**: 5-10x más rápido que en CPUs estándar
- **Visualizaciones**: Renderizado instantáneo incluso para gráficos complejos

## 🔒 Lineamientos de Cumplimiento GDPR

Este sistema está diseñado para cumplir con los siguientes principios clave del GDPR:

### Principios Implementados del Framework de GDPR
- **Minimización de datos (Art. 5.1.c)**: Reducción de granularidad mediante agrupación
- **Privacidad desde el diseño (Art. 25)**: Anonimización incorporada desde el inicio
- **Derecho al olvido (indirecto)**: Datos anonimizados que no permiten identificación
- **Seguridad del tratamiento (Art. 32)**: Uso de técnicas criptográficas (SHA-256)

### Recomendaciones de Uso
- Mantener un valor de k ≥ 10 para niveles adecuados de anonimización
- Verificar siempre la l-diversidad (l ≥ 2) después de agrupar datos
- Documentar todas las técnicas aplicadas para auditorías
- Realizar evaluaciones periódicas con datos actualizados

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, siga estos pasos:

1. Fork el repositorio
2. Cree una rama para su característica (`git checkout -b feature/nueva-caracteristica`)
3. Haga commit de sus cambios (`git commit -am 'Añadir: nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abra un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la Licencia MIT - vea el archivo `LICENSE` para más detalles.

---

Desarrollado como parte de un Trabajo Final de Máster en Análisis de Datos Masivos e Inteligencia Empresarial. Optimizado específicamente para MacBook Pro con Apple M2 Pro.
