# EDA y Anonimización de Datos Bancarios para Detección de Fraude
# =============================================================
# Este módulo implementa técnicas de análisis exploratorio y anonimización
# para datos bancarios, enfocado en la detección de fraude según el TFM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import itertools
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('ggplot')
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# Módulo 1: Funciones de Carga y Análisis Inicial
# ----------------------------------------------
def cargar_datos(ruta_archivo):
    """
    Carga el dataset desde la ruta especificada.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    df = pd.read_csv(ruta_archivo)
    print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df

def mostrar_info_basica(df):
    """
    Muestra información básica del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== INFORMACIÓN BÁSICA DEL DATASET =====")
    print("\nPrimeras 5 filas:")
    display(df.head())
    
    print("\nInformación del dataset:")
    display(df.info())
    
    print("\nEstadísticas descriptivas:")
    display(df.describe(include='all').T)
    
    print("\nValores nulos por columna:")
    display(df.isnull().sum())

def visualizar_valores_nulos(df):
    """
    Visualiza los valores nulos en el DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Mapa de calor de valores nulos')
    plt.tight_layout()
    plt.show()

# Módulo 2: Funciones de Identificación de Datos Sensibles
# ------------------------------------------------------
def identificar_columnas_sensibles(df):
    """
    Identifica las columnas con información sensible.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        
    Returns:
        list: Lista de columnas sensibles
    """
    # Para este dataset específico
    columnas_sensibles = [
        'nameOrig',  # Nombre/ID de origen
        'nameDest'   # Nombre/ID de destino
    ]
    
    print("\n===== IDENTIFICACIÓN DE DATOS SENSIBLES =====")
    print("\nColumnas con información sensible identificadas:")
    print(columnas_sensibles)
    
    return columnas_sensibles

def analizar_categoricas(df):
    """
    Analiza las columnas categóricas del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("\nColumnas categóricas:")
    print(columnas_categoricas)
    
    for col in columnas_categoricas:
        print(f"\nValores únicos en columna '{col}': {df[col].nunique()}")
        if df[col].nunique() < 20:  # Solo mostrar si hay pocos valores únicos
            display(df[col].value_counts())
        else:
            display(df[col].value_counts().head(10))  # Mostrar solo los 10 más frecuentes

def analizar_transacciones_fraude(df):
    """
    Analiza la distribución de tipos de transacciones y fraude.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    if 'type' in df.columns and 'isFraud' in df.columns:
        print("\nDistribución de tipos de transacciones:")
        display(df['type'].value_counts())
        
        print("\nDistribución de transacciones fraudulentas:")
        display(df['isFraud'].value_counts())
        print("\nPorcentaje de transacciones fraudulentas:",
            round(df['isFraud'].mean() * 100, 2), "%")
        
        # Análisis de fraude por tipo de transacción
        fraude_por_tipo = df.groupby('type')['isFraud'].mean().reset_index()
        fraude_por_tipo['porcentaje_fraude'] = fraude_por_tipo['isFraud'] * 100
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='type', y='porcentaje_fraude', data=fraude_por_tipo)
        plt.title('Porcentaje de Fraude por Tipo de Transacción')
        plt.ylabel('Porcentaje de Fraude (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Las columnas 'type' o 'isFraud' no están presentes en el dataset.")

# Módulo 3: Funciones de Análisis Exploratorio Detallado
# ----------------------------------------------------
def analizar_variables_numericas(df):
    """
    Analiza las variables numéricas con histogramas y boxplots.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        
    Returns:
        list: Lista de columnas numéricas
    """
    print("\n===== ANÁLISIS DE VARIABLES NUMÉRICAS =====")
    
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_para_histograma = [col for col in columnas_numericas if col not in ['isFraud', 'isFlaggedFraud']]
    
    # Histogramas
    if columnas_para_histograma:
        plt.figure(figsize=(18, 15))
        for i, col in enumerate(columnas_para_histograma, 1):
            if i <= 9:  # Limitar a 9 subplots
                plt.subplot(3, 3, i)
                sns.histplot(df[col], kde=True)
                plt.title(f'Distribución de {col}')
        plt.tight_layout()
        plt.show()
        
        # Boxplots
        plt.figure(figsize=(18, 15))
        for i, col in enumerate(columnas_para_histograma, 1):
            if i <= 9:  # Limitar a 9 subplots
                plt.subplot(3, 3, i)
                sns.boxplot(y=df[col])
                plt.title(f'Boxplot de {col}')
        plt.tight_layout()
        plt.show()
    else:
        print("No hay columnas numéricas para analizar mediante histogramas.")
    
    return columnas_numericas

def analizar_montos_por_tipo(df):
    """
    Analiza la distribución de montos por tipo de transacción y fraude.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== ANÁLISIS DE MONTOS POR TIPO Y FRAUDE =====")
    
    if all(col in df.columns for col in ['type', 'amount']):
        # Distribución del monto según tipo de transacción
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='type', y='amount', data=df)
        plt.title('Distribución de Montos por Tipo de Transacción')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Las columnas 'type' o 'amount' no están presentes en el dataset.")
    
    if all(col in df.columns for col in ['isFraud', 'amount']):
        # Distribución del monto según si es fraude o no
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='isFraud', y='amount', data=df)
        plt.title('Distribución de Montos por Estado de Fraude')
        plt.tight_layout()
        plt.show()
    else:
        print("Las columnas 'isFraud' o 'amount' no están presentes en el dataset.")

def analizar_correlaciones(df, columnas_numericas):
    """
    Analiza las correlaciones entre variables numéricas.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        columnas_numericas (list): Lista de columnas numéricas
    """
    print("\n===== MATRIZ DE CORRELACIÓN =====")
    
    # Verificar que haya suficientes columnas numéricas para crear una matriz de correlación
    if len(columnas_numericas) > 1:
        plt.figure(figsize=(14, 10))
        correlation_matrix = df[columnas_numericas].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.show()
    else:
        print("No hay suficientes columnas numéricas para calcular correlaciones.")

def analizar_balances(df):
    """
    Analiza los balances antes y después de transacciones.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== ANÁLISIS DE BALANCES =====")
    
    # Verificar que existan las columnas necesarias
    if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        sns.scatterplot(x='oldbalanceOrg', y='newbalanceOrig', hue='isFraud', data=df)
        plt.title('Balance Origen: Antes vs Después (por Fraude)')
        
        plt.subplot(2, 1, 2)
        sns.scatterplot(x='oldbalanceDest', y='newbalanceDest', hue='isFraud', data=df)
        plt.title('Balance Destino: Antes vs Después (por Fraude)')
        plt.tight_layout()
        plt.show()
    else:
        print("No se puede analizar balances: faltan columnas necesarias.")

def analizar_patrones_temporales(df):
    """
    Analiza patrones temporales en los datos.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== ANÁLISIS DE PATRONES TEMPORALES =====")
    
    # Verificar que existan las columnas necesarias
    if 'step' in df.columns and 'amount' in df.columns:
        # Agrupar por 'step' y calcular el promedio solo de 'amount'
        df_agrupado = df.groupby('step')['amount'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='step', y='amount', data=df_agrupado)
        plt.title('Evolución Temporal del Monto Promedio')
        plt.tight_layout()
        plt.show()
        
        # Si existe la columna isFraud, analizar fraude por paso temporal
        if 'isFraud' in df.columns:
            fraude_por_paso = df.groupby('step')['isFraud'].mean().reset_index()
            fraude_por_paso['porcentaje_fraude'] = fraude_por_paso['isFraud'] * 100
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='step', y='porcentaje_fraude', data=fraude_por_paso)
            plt.title('Evolución Temporal del Porcentaje de Fraude')
            plt.ylabel('Porcentaje de Fraude (%)')
            plt.tight_layout()
            plt.show()
    else:
        print("No se puede analizar patrones temporales: columnas 'step' o 'amount' no encontradas.")

# Módulo 4: Funciones de Anonimización
# ----------------------------------
def iniciar_anonimizacion(df):
    """
    Inicia el proceso de anonimización creando una copia del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: Copia del DataFrame para anonimización
    """
    print("\n===== INICIANDO PROCESO DE ANONIMIZACIÓN =====")
    
    df_anonimizado = df.copy()
    
    return df_anonimizado

def analizar_formato_ids(df_anonimizado):
    """
    Analiza el formato de los IDs para preservar patrones.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
    """
    if 'nameOrig' in df_anonimizado.columns and 'nameDest' in df_anonimizado.columns:
        muestra_nameOrig = df_anonimizado['nameOrig'].iloc[0]
        muestra_nameDest = df_anonimizado['nameDest'].iloc[0]
        
        print(f"Formato de nameOrig: {muestra_nameOrig}")
        print(f"Formato de nameDest: {muestra_nameDest}")
        
        # Comprobar si los IDs tienen un prefijo común
        prefijos_orig = set([name[:1] for name in df_anonimizado['nameOrig'].astype(str).sample(min(100, len(df_anonimizado)))])
        print(f"Prefijos detectados en nameOrig: {prefijos_orig}")
    else:
        print("Las columnas de IDs 'nameOrig' o 'nameDest' no están presentes en el dataset.")

def hash_sha256(value):
    """
    Aplica función hash SHA-256 a un valor.
    
    Args:
        value: Valor a hashear
        
    Returns:
        str: Valor hasheado
    """
    # Asegurarse de que el valor es un string antes de codificar
    if not isinstance(value, str):
        value = str(value)
    
    return hashlib.sha256(value.encode()).hexdigest()

def anonimizar_ids(df_anonimizado):
    """
    Anonimiza los IDs de origen y destino usando hashing SHA-256.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
        
    Returns:
        pd.DataFrame: DataFrame con IDs anonimizados
    """
    print("\n===== ANONIMIZANDO IDs CON SHA-256 =====")
    
    if 'nameOrig' in df_anonimizado.columns:
        df_anonimizado['nameOrig'] = df_anonimizado['nameOrig'].apply(hash_sha256)
        print("Columna 'nameOrig' anonimizada con SHA-256")
    
    if 'nameDest' in df_anonimizado.columns:
        df_anonimizado['nameDest'] = df_anonimizado['nameDest'].apply(hash_sha256)
        print("Columna 'nameDest' anonimizada con SHA-256")
    
    return df_anonimizado

def agrupar_valores_numericos(df_anonimizado, columna, bins, labels):
    """
    Agrupa valores numéricos en rangos para k-anonimato.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
        columna (str): Nombre de la columna a agrupar
        bins (list): Lista de límites para los rangos
        labels (list): Etiquetas para los rangos
        
    Returns:
        pd.DataFrame: DataFrame con valores agrupados
    """
    if columna in df_anonimizado.columns:
        df_anonimizado[f'{columna}_group'] = pd.cut(
            df_anonimizado[columna], 
            bins=bins, 
            labels=labels, 
            right=False, 
            include_lowest=True
        )
        print(f"Columna '{columna}' agrupada en rangos para k-anonimato")
    else:
        print(f"La columna '{columna}' no está presente en el dataset.")
    
    return df_anonimizado

def anonimizar_mediante_agrupacion(df_anonimizado):
    """
    Anonimiza valores numéricos mediante agrupación (k-anonimato).
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
        
    Returns:
        pd.DataFrame: DataFrame con valores numéricos agrupados
    """
    print("\n===== ANONIMIZANDO MEDIANTE AGRUPACIÓN (K-ANONIMATO) =====")
    
    # Agrupar amount en rangos
    if 'amount' in df_anonimizado.columns:
        max_amount = df_anonimizado['amount'].max() if not df_anonimizado['amount'].empty else 100000
        bins = [0, 1000, 5000, 10000, 50000, 100000, max_amount + 1]  # +1 para incluir el max
        labels = ['0-1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K', '+100K']
        
        df_anonimizado = agrupar_valores_numericos(df_anonimizado, 'amount', bins, labels)
    
    # Agrupar step en bloques de tiempo (mañana, tarde, noche)
    if 'step' in df_anonimizado.columns:
        max_step = df_anonimizado['step'].max() if not df_anonimizado['step'].empty else 744  # Por ejemplo, 31 días * 24 horas
        bins_step = [0, 8, 16, 24, max_step + 1]
        labels_step = ['madrugada', 'mañana', 'tarde', 'noche']
        
        df_anonimizado = agrupar_valores_numericos(df_anonimizado, 'step', bins_step, labels_step)
    
    return df_anonimizado

def verificar_k_anonimato(df_anonimizado, k=10):
    """
    Verifica si se cumple el k-anonimato en los grupos formados.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
        k (int): Valor de k para k-anonimato (predeterminado: 10)
        
    Returns:
        bool: True si se cumple k-anonimato, False en caso contrario
    """
    print(f"\n===== VERIFICANDO K-ANONIMATO (k={k}) =====")
    
    # Identificar columnas de agrupación
    columnas_agrupacion = [col for col in df_anonimizado.columns if col.endswith('_group')]
    
    if not columnas_agrupacion:
        print("No se encontraron columnas de agrupación para verificar k-anonimato.")
        return False
    
    # Contar frecuencias por grupo
    frecuencias = df_anonimizado.groupby(columnas_agrupacion).size().reset_index(name='frecuencia')
    
    # Verificar si todos los grupos tienen al menos k registros
    cumple_k_anonimato = frecuencias['frecuencia'].min() >= k
    
    print(f"Tamaño mínimo de grupo: {frecuencias['frecuencia'].min()}")
    print(f"Número de grupos: {len(frecuencias)}")
    print(f"¿Cumple k-anonimato?: {'Sí' if cumple_k_anonimato else 'No'}")
    
    # Mostrar distribución de tamaños de grupo
    plt.figure(figsize=(10, 6))
    sns.histplot(frecuencias['frecuencia'], bins=20, kde=True)
    plt.axvline(x=k, color='red', linestyle='--', label=f'k={k}')
    plt.title('Distribución de Tamaños de Grupo')
    plt.xlabel('Tamaño del Grupo')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return cumple_k_anonimato

def verificar_l_diversidad(df_anonimizado, columnas_agrupacion, columna_sensible, l=2):
    """
    Verifica si se cumple la l-diversidad en los grupos formados.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
        columnas_agrupacion (list): Lista de columnas que definen los grupos
        columna_sensible (str): Columna sensible para verificar diversidad
        l (int): Valor de l para l-diversidad (predeterminado: 2)
        
    Returns:
        bool: True si se cumple l-diversidad, False en caso contrario
    """
    print(f"\n===== VERIFICANDO L-DIVERSIDAD (l={l}) PARA '{columna_sensible}' =====")
    
    if not all(col in df_anonimizado.columns for col in columnas_agrupacion) or columna_sensible not in df_anonimizado.columns:
        print("No se encontraron las columnas necesarias para verificar l-diversidad.")
        return False
    
    # Calcular la diversidad de cada grupo
    diversidad_por_grupo = df_anonimizado.groupby(columnas_agrupacion)[columna_sensible].nunique().reset_index(name='diversidad')
    
    # Verificar si todos los grupos tienen al menos l valores distintos
    cumple_l_diversidad = diversidad_por_grupo['diversidad'].min() >= l
    
    print(f"Diversidad mínima: {diversidad_por_grupo['diversidad'].min()}")
    print(f"Número de grupos: {len(diversidad_por_grupo)}")
    print(f"¿Cumple l-diversidad?: {'Sí' if cumple_l_diversidad else 'No'}")
    
    # Mostrar distribución de diversidad
    plt.figure(figsize=(10, 6))
    sns.countplot(data=diversidad_por_grupo, x='diversidad')
    plt.axvline(x=l-0.5, color='red', linestyle='--', label=f'l={l}')
    plt.title(f'Distribución de l-Diversidad para {columna_sensible}')
    plt.xlabel(f'Número de valores distintos de {columna_sensible}')
    plt.ylabel('Número de Grupos')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return cumple_l_diversidad

# Módulo 5: Funciones de Modelado y Evaluación
# ------------------------------------------
def preparar_datos_para_modelo(df, columnas_modelo=None, columna_objetivo='isFraud', test_size=0.3, random_state=42):
    """
    Prepara los datos para entrenar un modelo de clasificación.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columnas_modelo (list): Lista de columnas a usar para el modelo (None para usar todas excepto el objetivo)
        columna_objetivo (str): Nombre de la columna objetivo
        test_size (float): Proporción de datos para test (0.0-1.0)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, columnas_usadas)
    """
    print("\n===== PREPARANDO DATOS PARA MODELO =====")
    
    if columna_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no está presente en el dataset.")
    
    # Si no se especifican columnas, usar todas excepto la objetivo
    if columnas_modelo is None:
        columnas_modelo = [col for col in df.columns if col != columna_objetivo]
    
    # Verificar si hay variables categóricas y codificarlas
    X = df[columnas_modelo].copy()
    y = df[columna_objetivo].copy()
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        print(f"Codificando variable categórica: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
    print(f"Columnas utilizadas: {X.columns.tolist()}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def entrenar_y_evaluar_modelo(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    """
    Entrena un modelo Random Forest y evalúa su rendimiento.
    
    Args:
        X_train (pd.DataFrame): Datos de entrenamiento
        X_test (pd.DataFrame): Datos de prueba
        y_train (pd.Series): Etiquetas de entrenamiento
        y_test (pd.Series): Etiquetas de prueba
        n_estimators (int): Número de árboles en el Random Forest
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (modelo, precision, sensibilidad, f1)
    """
    print("\n===== ENTRENANDO Y EVALUANDO MODELO RANDOM FOREST =====")
    
    # Entrenar modelo
    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    precision = accuracy_score(y_test, y_pred)
    sensibilidad = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad (Recall): {sensibilidad:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Mostrar matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.show()
    
    # Mostrar importancia de variables
    plt.figure(figsize=(12, 6))
    importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
    importancias = importancias.sort_values(ascending=False)
    sns.barplot(x=importancias.index, y=importancias.values)
    plt.title('Importancia de Variables')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return modelo, precision, sensibilidad, f1

def comparar_modelos(metricas_original, metricas_anonimizado):
    """
    Compara los resultados de dos modelos (original vs anonimizado).
    
    Args:
        metricas_original (tuple): (precision, sensibilidad, f1) del modelo original
        metricas_anonimizado (tuple): (precision, sensibilidad, f1) del modelo anonimizado
    """
    print("\n===== COMPARACIÓN DE MODELOS: ORIGINAL VS ANONIMIZADO =====")
    
    precision_orig, sensibilidad_orig, f1_orig = metricas_original
    precision_anon, sensibilidad_anon, f1_anon = metricas_anonimizado
    
    # Crear DataFrame para la comparación
    comparacion = pd.DataFrame({
        'Original': [precision_orig, sensibilidad_orig, f1_orig],
        'Anonimizado': [precision_anon, sensibilidad_anon, f1_anon],
        'Diferencia': [precision_orig - precision_anon, sensibilidad_orig - sensibilidad_anon, f1_orig - f1_anon]
    }, index=['Precisión', 'Sensibilidad', 'F1-Score'])
    
    display(comparacion)
    
    # Visualizar comparación
    plt.figure(figsize=(10, 6))
    comparacion[['Original', 'Anonimizado']].plot(kind='bar')
    plt.title('Comparación de Métricas: Original vs Anonimizado')
    plt.ylabel('Valor')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v1, v2 in zip(range(len(comparacion)), comparacion['Original'], comparacion['Anonimizado']):
        plt.text(i-0.2, v1+0.02, f"{v1:.3f}", color='black', fontweight='bold')
        plt.text(i+0.05, v2+0.02, f"{v2:.3f}", color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar diferencia porcentual
    dif_precision = (precision_orig - precision_anon) / precision_orig * 100
    dif_sensibilidad = (sensibilidad_orig - sensibilidad_anon) / sensibilidad_orig * 100
    dif_f1 = (f1_orig - f1_anon) / f1_orig * 100
    
    print(f"Reducción en precisión: {dif_precision:.2f}%")
    print(f"Reducción en sensibilidad: {dif_sensibilidad:.2f}%")
    print(f"Reducción en F1-Score: {dif_f1:.2f}%")

# Módulo 6: Evaluación de Cumplimiento GDPR
# ---------------------------------------
def evaluar_cumplimiento_gdpr(k_anonimato, l_diversidad, dif_precision):
    """
    Evalúa el cumplimiento del GDPR según los parámetros de anonimización y pérdida de precisión.
    
    Args:
        k_anonimato (int): Valor de k para k-anonimato
        l_diversidad (int): Valor de l para l-diversidad
        dif_precision (float): Diferencia porcentual en precisión
        
    Returns:
        dict: Evaluación de cumplimiento GDPR
    """
    print("\n===== EVALUACIÓN DE CUMPLIMIENTO GDPR =====")
    
    evaluacion = {
        'k_anonimato': {
            'valor': k_anonimato,
            'estado': 'Alto' if k_anonimato >= 10 else 'Medio' if k_anonimato >= 5 else 'Bajo',
            'descripcion': f"Indistinguibilidad conforme a Recital 26 (k={k_anonimato})"
        },
        'l_diversidad': {
            'valor': l_diversidad,
            'estado': 'Alto' if l_diversidad >= 3 else 'Medio' if l_diversidad >= 2 else 'Bajo',
            'descripcion': f"Evita inferencia por homogeneidad (l={l_diversidad})"
        },
        'perdida_precision': {
            'valor': dif_precision,
            'estado': 'Aceptable' if dif_precision <= 10 else 'Significativa' if dif_precision <= 20 else 'Alta',
            'descripcion': f"Impacto en la utilidad del modelo: {dif_precision:.2f}%"
        },
        'riesgo_reidentificacion': {
            'estado': 'Bajo' if k_anonimato >= 10 and l_diversidad >= 2 else 'Medio' if k_anonimato >= 5 else 'Alto',
            'descripcion': "Estimación del riesgo de reidentificación"
        }
    }
    
    # Mostrar resultados
    print("\nEvaluación del cumplimiento del GDPR:")
    for criterio, datos in evaluacion.items():
        print(f"- {criterio.replace('_', ' ').title()}: {datos.get('estado', '')}")
        print(f"  {datos.get('descripcion', '')}")
        if 'valor' in datos:
            print(f"  Valor: {datos['valor']}")
    
    # Evaluación global
    if evaluacion['riesgo_reidentificacion']['estado'] == 'Bajo' and evaluacion['perdida_precision']['estado'] == 'Aceptable':
        estado_global = "CUMPLE"
    elif evaluacion['riesgo_reidentificacion']['estado'] == 'Medio':
        estado_global = "CUMPLE PARCIALMENTE"
    else:
        estado_global = "NO CUMPLE"
    
    print(f"\nEstado global: {estado_global}")
    
    return evaluacion

def generar_dashboard_cumplimiento(evaluacion, metricas_original, metricas_anonimizado):
    """
    Genera un dashboard simple de cumplimiento GDPR.
    
    Args:
        evaluacion (dict): Resultado de la evaluación de cumplimiento
        metricas_original (tuple): (precision, sensibilidad, f1) del modelo original
        metricas_anonimizado (tuple): (precision, sensibilidad, f1) del modelo anonimizado
    """
    print("\n===== DASHBOARD DE CUMPLIMIENTO GDPR =====")
    
    # Definir colores según estado
    colores = {'Alto': 'green', 'Medio': 'orange', 'Bajo': 'red',
               'Aceptable': 'green', 'Significativa': 'orange', 'Alta': 'red'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Indicadores de Cumplimiento
    indicadores = ['k_anonimato', 'l_diversidad', 'perdida_precision', 'riesgo_reidentificacion']
    estados = [evaluacion[ind]['estado'] if ind in evaluacion else 'N/A' for ind in indicadores]
    colores_ind = [colores.get(estado, 'gray') for estado in estados]
    
    axes[0, 0].bar(range(len(indicadores)), [1]*len(indicadores), color=colores_ind)
    axes[0, 0].set_xticks(range(len(indicadores)))
    axes[0, 0].set_xticklabels([ind.replace('_', ' ').title() for ind in indicadores], rotation=45)
    axes[0, 0].set_title('Indicadores de Cumplimiento')
    axes[0, 0].set_ylim(0, 1.2)
    axes[0, 0].set_yticks([])
    
    for i, estado in enumerate(estados):
        axes[0, 0].text(i, 0.5, estado, ha='center', va='center', fontweight='bold', color='white')
    
    # Gráfico 2: Comparación de Métricas
    precision_orig, sensibilidad_orig, f1_orig = metricas_original
    precision_anon, sensibilidad_anon, f1_anon = metricas_anonimizado
    
    metricas = ['Precisión', 'Sensibilidad', 'F1-Score']
    orig_vals = [precision_orig, sensibilidad_orig, f1_orig]
    anon_vals = [precision_anon, sensibilidad_anon, f1_anon]
    
    x = np.arange(len(metricas))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, orig_vals, width, label='Original', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, anon_vals, width, label='Anonimizado', color='green', alpha=0.7)
    axes[0, 1].set_title('Comparación de Métricas')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metricas)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico 3: K-Anonimato
    k = evaluacion['k_anonimato']['valor']
    axes[1, 0].pie([k, 100-k], colors=['green', 'lightgray'], 
                  labels=[f'k={k}', ''], autopct='%1.1f%%')
    axes[1, 0].set_title('K-Anonimato')
    
    # Gráfico 4: Estado Global
    if 'riesgo_reidentificacion' in evaluacion:
        riesgo = evaluacion['riesgo_reidentificacion']['estado']
        color_riesgo = colores.get(riesgo, 'gray')
    else:
        riesgo = "No evaluado"
        color_riesgo = 'gray'
    
    axes[1, 1].text(0.5, 0.5, f"Riesgo de\nReidentificación:\n{riesgo}", 
                   ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 1].set_facecolor(color_riesgo)
    axes[1, 1].set_title('Evaluación de Riesgo')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Módulo 7: Flujo de Trabajo Completo
# ---------------------------------
def ejecutar_analisis_deteccion_fraude(ruta_entrada, test_size=0.3, random_state=42, k_anonimato=10, l_diversidad=2):
    """
    Ejecuta el flujo completo de análisis y detección de fraude con anonimización.
    
    Args:
        ruta_entrada (str): Ruta al archivo CSV de entrada
        test_size (float): Proporción de datos para test (0.0-1.0)
        random_state (int): Semilla para reproducibilidad
        k_anonimato (int): Valor de k para k-anonimato
        l_diversidad (int): Valor de l para l-diversidad
        
    Returns:
        tuple: (df_original, df_anonimizado, modelo_original, modelo_anonimizado, evaluacion_gdpr)
    """
    try:
        # 1. Cargar y analizar datos
        df = cargar_datos(ruta_entrada)
        mostrar_info_basica(df)
        visualizar_valores_nulos(df)
        
        # 2. Identificar datos sensibles
        columnas_sensibles = identificar_columnas_sensibles(df)
        analizar_categoricas(df)
        analizar_transacciones_fraude(df)
        
        # 3. Análisis exploratorio detallado
        columnas_numericas = analizar_variables_numericas(df)
        analizar_montos_por_tipo(df)
        analizar_correlaciones(df, columnas_numericas)
        analizar_balances(df)
        analizar_patrones_temporales(df)
        
        # 4. Preparar datos y entrenar modelo original
        print("\n===== ENTRENAMIENTO DE MODELO ORIGINAL (SIN ANONIMIZACIÓN) =====")
        
        # Seleccionar columnas relevantes para el modelo
        columnas_modelo = [
            'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest', 'step'
        ]
        
        # Codificar variables categóricas
        df_procesado = df.copy()
        if 'type' in df_procesado.columns:
            le = LabelEncoder()
            df_procesado['type'] = le.fit_transform(df_procesado['type'])
        
        # Preparar datos para el modelo original
        X_train_orig, X_test_orig, y_train_orig, y_test_orig, _ = preparar_datos_para_modelo(
            df_procesado, 
            columnas_modelo=columnas_modelo, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Entrenar y evaluar modelo original
        modelo_original, precision_orig, sensibilidad_orig, f1_orig = entrenar_y_evaluar_modelo(
            X_train_orig, X_test_orig, y_train_orig, y_test_orig, 
            n_estimators=100, random_state=random_state
        )
        
        # 5. Anonimización
        print("\n===== PROCESO DE ANONIMIZACIÓN =====")
        df_anonimizado = iniciar_anonimizacion(df)
        analizar_formato_ids(df_anonimizado)
        df_anonimizado = anonimizar_ids(df_anonimizado)
        df_anonimizado = anonimizar_mediante_agrupacion(df_anonimizado)
        
        # Verificar cumplimiento de k-anonimato y l-diversidad
        cumple_k = verificar_k_anonimato(df_anonimizado, k=k_anonimato)
        
        # Columnas para verificar l-diversidad (asumiendo que ya tenemos columnas agrupadas)
        columnas_agrupacion = [col for col in df_anonimizado.columns if col.endswith('_group')]
        if 'type' in df_anonimizado.columns and columnas_agrupacion:
            cumple_l = verificar_l_diversidad(df_anonimizado, columnas_agrupacion, 'type', l=l_diversidad)
        else:
            cumple_l = False
            print("No se pudo verificar l-diversidad: falta 'type' o columnas de agrupación.")
        
        # 6. Entrenar modelo con datos anonimizados
        print("\n===== ENTRENAMIENTO DE MODELO CON DATOS ANONIMIZADOS =====")
        
        # Utilizamos las columnas agrupadas en lugar de las originales
        columnas_modelo_anon = [
            'type', 'amount_group', 'step_group', 
            'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest'
        ]
        
        # Codificar variables categóricas
        df_anonimizado_procesado = df_anonimizado.copy()
        
        for col in ['type', 'amount_group', 'step_group']:
            if col in df_anonimizado_procesado.columns:
                le = LabelEncoder()
                df_anonimizado_procesado[col] = le.fit_transform(df_anonimizado_procesado[col])
        
        # Preparar datos para el modelo anonimizado
        columnas_disponibles = [col for col in columnas_modelo_anon if col in df_anonimizado_procesado.columns]
        
        X_train_anon, X_test_anon, y_train_anon, y_test_anon, _ = preparar_datos_para_modelo(
            df_anonimizado_procesado, 
            columnas_modelo=columnas_disponibles, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Entrenar y evaluar modelo anonimizado
        modelo_anonimizado, precision_anon, sensibilidad_anon, f1_anon = entrenar_y_evaluar_modelo(
            X_train_anon, X_test_anon, y_train_anon, y_test_anon, 
            n_estimators=100, random_state=random_state
        )
        
        # 7. Comparar modelos
        comparar_modelos(
            (precision_orig, sensibilidad_orig, f1_orig),
            (precision_anon, sensibilidad_anon, f1_anon)
        )
        
        # 8. Evaluar cumplimiento GDPR
        dif_precision = (precision_orig - precision_anon) / precision_orig * 100
        evaluacion_gdpr = evaluar_cumplimiento_gdpr(k_anonimato, l_diversidad, dif_precision)
        
        # 9. Generar dashboard
        generar_dashboard_cumplimiento(
            evaluacion_gdpr,
            (precision_orig, sensibilidad_orig, f1_orig),
            (precision_anon, sensibilidad_anon, f1_anon)
        )
        
        return df, df_anonimizado, modelo_original, modelo_anonimizado, evaluacion_gdpr
        
    except Exception as e:
        print(f"\n¡ERROR! Se ha producido un error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Si ya hemos cargado el DataFrame, lo devolvemos para que el usuario pueda trabajar con él
        if 'df' in locals():
            print("\nA pesar del error, se devuelve el DataFrame original para que puedas trabajar con él.")
            return df, None, None, None, None
        else:
            return None, None, None, None, None

# Código principal para ejecutar si se usa como script independiente
if __name__ == "__main__":
    # Definir ruta de archivo (ajustar según sea necesario)
    ruta_entrada = "dataset_anonimizacion_datos.csv"
    
    # Ejecutar todo el proceso
    df_original, df_anonimizado, modelo_original, modelo_anonimizado, evaluacion_gdpr = ejecutar_analisis_deteccion_fraude(
        ruta_entrada,
        test_size=0.3,
        random_state=42,
        k_anonimato=10,
        l_diversidad=2
    )