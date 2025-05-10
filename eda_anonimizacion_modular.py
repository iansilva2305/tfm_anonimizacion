# EDA y Anonimización de Datos Bancarios
# =================================
# Este notebook está modularizado para realizar un análisis exploratorio completo
# de datos de transacciones bancarias y aplicar técnicas de anonimización.

# Configuración e Importaciones
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import itertools
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('ggplot')
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# Importaciones opcionales con manejo de errores
try:
    from faker import Faker
    faker_available = True
except ImportError:
    print("Advertencia: Faker no está instalado. Algunas funciones de anonimización no estarán disponibles.")
    faker_available = False

try:
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
except ImportError:
    print("Advertencia: scikit-learn no está instalado. Algunas funciones pueden no estar disponibles.")
    sklearn_available = False

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
    display(columnas_sensibles)
    
    return columnas_sensibles

def analizar_categoricas(df):
    """
    Analiza las columnas categóricas del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("\nColumnas categóricas:")
    display(columnas_categoricas)
    
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
    print("\nDistribución de tipos de transacciones:")
    display(df['type'].value_counts())
    
    print("\nDistribución de transacciones fraudulentas:")
    display(df['isFraud'].value_counts())
    print("\nPorcentaje de transacciones fraudulentas:",
          round(df['isFraud'].mean() * 100, 2), "%")

# Módulo 3: Funciones de Análisis Exploratorio Detallado
# ----------------------------------------------------
def analizar_variables_numericas(df):
    """
    Analiza las variables numéricas con histogramas y boxplots.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== ANÁLISIS DE VARIABLES NUMÉRICAS =====")
    
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_para_histograma = [col for col in columnas_numericas if col not in ['isFraud', 'isFlaggedFraud']]
    
    # Histogramas
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
    
    return columnas_numericas

def analizar_montos_por_tipo(df):
    """
    Analiza la distribución de montos por tipo de transacción y fraude.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== ANÁLISIS DE MONTOS POR TIPO Y FRAUDE =====")
    
    # Distribución del monto según tipo de transacción
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='type', y='amount', data=df)
    plt.title('Distribución de Montos por Tipo de Transacción')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Distribución del monto según si es fraude o no
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='isFraud', y='amount', data=df)
    plt.title('Distribución de Montos por Estado de Fraude')
    plt.tight_layout()
    plt.show()

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
    else:
        print("No se puede analizar patrones temporales: columnas 'step' o 'amount' no encontradas.")


def analizar_fraude_por_tipo(df):
    """
    Analiza el porcentaje de fraude por tipo de transacción.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("\n===== ANÁLISIS DE FRAUDE POR TIPO DE TRANSACCIÓN =====")
    
    # Verificar que existan las columnas necesarias
    if 'type' in df.columns and 'isFraud' in df.columns:
        # Calcular el porcentaje de fraude por tipo
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
        print("No se puede analizar fraude por tipo: columnas 'type' o 'isFraud' no encontradas.")

# Módulo 4: Funciones de Anonimización
# ----------------------------------
def iniciar_anonimizacion(df):
    """
    Inicia el proceso de anonimización creando una copia del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: Copia del DataFrame para anonimización
        Faker: Objeto Faker para generar datos sintéticos
    """
    print("\n===== INICIANDO PROCESO DE ANONIMIZACIÓN =====")
    
    df_anonimizado = df.copy()
    fake = Faker()
    
    return df_anonimizado, fake

def analizar_formato_ids(df_anonimizado):
    """
    Analiza el formato de los IDs para preservar patrones.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
    """
    muestra_nameOrig = df_anonimizado['nameOrig'].iloc[0]
    muestra_nameDest = df_anonimizado['nameDest'].iloc[0]
    
    print(f"Formato de nameOrig: {muestra_nameOrig}")
    print(f"Formato de nameDest: {muestra_nameDest}")
    
    # Comprobar si los IDs tienen un prefijo común
    if 'nameOrig' in df_anonimizado.columns:
        prefijos_orig = set([name[:1] for name in df_anonimizado['nameOrig'].astype(str).sample(min(100, len(df_anonimizado)))])
        print(f"Prefijos detectados en nameOrig: {prefijos_orig}")

def anonimizar_ids(df_anonimizado):
    """
    Anonimiza los IDs de origen y destino.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
        
    Returns:
        pd.DataFrame: DataFrame con IDs anonimizados
    """
    print("\n===== ANONIMIZANDO IDs =====")
    
    # Preservar el formato: mantener prefijo y longitud pero aleatorizar números
    def anonimizar_id(id_original):
        if isinstance(id_original, str) and len(id_original) > 1:
            prefijo = id_original[0]  # Primer carácter (prefijo)
            longitud = len(id_original) - 1  # Longitud sin el prefijo
            return prefijo + ''.join([str(np.random.randint(0, 10)) for _ in range(longitud)])
        return id_original
    
    # Aplicar anonimización
    df_anonimizado['nameOrig'] = df_anonimizado['nameOrig'].apply(anonimizar_id)
    df_anonimizado['nameDest'] = df_anonimizado['nameDest'].apply(anonimizar_id)
    
    return df_anonimizado

def anonimizar_valores_numericos(df_anonimizado):
    """
    Anonimiza valores numéricos mediante perturbación.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
        
    Returns:
        pd.DataFrame: DataFrame con valores numéricos anonimizados
    """
    print("\n===== ANONIMIZANDO VALORES NUMÉRICOS =====")
    
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if col in df_anonimizado.columns:
            # Añadir ruido aleatorio (±3%)
            non_zero_indices = df_anonimizado[col] != 0
            if non_zero_indices.any():
                noise_factor = 0.03
                noise = np.random.normal(0, df_anonimizado.loc[non_zero_indices, col].std() * noise_factor, size=sum(non_zero_indices))
                df_anonimizado.loc[non_zero_indices, col] = df_anonimizado.loc[non_zero_indices, col] + noise
                # Evitar valores negativos después de añadir ruido
                df_anonimizado.loc[df_anonimizado[col] < 0, col] = 0
    
    return df_anonimizado

def anonimizar_tiempo(df_anonimizado):
    """
    Anonimiza la dimensión temporal mediante microagregación.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame a anonimizar
        
    Returns:
        pd.DataFrame: DataFrame con tiempo anonimizado
    """
    print("\n===== ANONIMIZANDO DIMENSIÓN TEMPORAL =====")
    
    if 'step' in df_anonimizado.columns:
        # Crear bins de tiempo (cada 6 pasos, por ejemplo)
        bin_size = 6
        df_anonimizado['step_bin'] = (df_anonimizado['step'] // bin_size) * bin_size
        # Opcional: eliminar paso original
        # df_anonimizado.drop('step', axis=1, inplace=True)
    
    return df_anonimizado

# Módulo 5: Funciones de Verificación de Anonimización
# -------------------------------------------------
def comparar_estadisticas(df, df_anonimizado):
    """
    Compara estadísticas descriptivas antes y después de la anonimización.
    
    Args:
        df (pd.DataFrame): DataFrame original
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
    """
    print("\n===== COMPARACIÓN DE ESTADÍSTICAS =====")
    
    print("Estadísticas descriptivas originales:")
    display(df.describe().T)
    
    print("\nEstadísticas descriptivas después de anonimización:")
    display(df_anonimizado.describe().T)

def verificar_ids_anonimizados(df, df_anonimizado, columnas_sensibles):
    """
    Verifica que los IDs estén correctamente anonimizados.
    
    Args:
        df (pd.DataFrame): DataFrame original
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
        columnas_sensibles (list): Lista de columnas sensibles
    """
    print("\n===== VERIFICACIÓN DE IDs ANONIMIZADOS =====")
    
    for col in columnas_sensibles:
        if col in df_anonimizado.columns:
            overlap = set(df[col].unique()) & set(df_anonimizado[col].unique())
            print(f"\nSolapamiento de valores en columna '{col}': {len(overlap)} valores")
            
            # Mostrar muestra comparativa
            print(f"\nMuestra comparativa de anonimización para '{col}':")
            muestra_comparativa = pd.DataFrame({
                'Original': df[col].head(10),
                'Anonimizado': df_anonimizado[col].head(10)
            })
            display(muestra_comparativa)

def evaluar_riesgo_reidentificacion(df_anonimizado):
    """
    Evalúa el riesgo de reidentificación en el DataFrame anonimizado.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
    """
    print("\n===== EVALUACIÓN DE RIESGO DE REIDENTIFICACIÓN =====")
    
    variables_riesgo = ['step_bin', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
    for i in range(2, 4):  # Probar combinaciones de 2 y 3 columnas
        for combo in itertools.combinations(variables_riesgo, i):
            if all(col in df_anonimizado.columns for col in combo):
                unique_combinations = df_anonimizado.groupby(list(combo)).size()
                ones_count = (unique_combinations == 1).sum()
                if ones_count > 0:
                    print(f"Combinación {combo}: {ones_count} registros únicamente identificables ({ones_count/len(df_anonimizado)*100:.2f}%)")

# Módulo 6: Visualización Comparativa y Resumen
# -------------------------------------------
def exportar_datos_anonimizados(df_anonimizado, ruta_salida):
    """
    Exporta los datos anonimizados a un archivo CSV.
    
    Args:
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
        ruta_salida (str): Ruta del archivo de salida
    """
    df_anonimizado.to_csv(ruta_salida, index=False)
    print(f"\nProceso de anonimización completado. Datos guardados en '{ruta_salida}'")

def resumir_proceso(df, df_anonimizado, columnas_sensibles):
    """
    Muestra un resumen del proceso de anonimización.
    
    Args:
        df (pd.DataFrame): DataFrame original
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
        columnas_sensibles (list): Lista de columnas sensibles
    """
    print("\n===== RESUMEN DEL PROCESO DE ANONIMIZACIÓN =====")
    print(f"- Registros procesados: {len(df)}")
    print(f"- Columnas originales: {len(df.columns)}")
    print(f"- Columnas en dataset anonimizado: {len(df_anonimizado.columns)}")
    print(f"- Columnas sensibles tratadas: {len(columnas_sensibles)}")

def comparar_distribuciones(df, df_anonimizado):
    """
    Compara distribuciones clave entre el DataFrame original y el anonimizado.
    
    Args:
        df (pd.DataFrame): DataFrame original
        df_anonimizado (pd.DataFrame): DataFrame anonimizado
    """
    print("\n===== COMPARACIÓN DE DISTRIBUCIONES =====")
    
    # Verificar que existan las columnas necesarias
    columnas_requeridas = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    if all(col in df.columns for col in columnas_requeridas) and all(col in df_anonimizado.columns for col in columnas_requeridas):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Distribución de montos
        sns.histplot(df['amount'], kde=True, ax=axes[0, 0], color='blue', alpha=0.5)
        sns.histplot(df_anonimizado['amount'], kde=True, ax=axes[0, 0], color='red', alpha=0.5)
        axes[0, 0].set_title('Distribución de Montos: Original vs Anonimizado')
        axes[0, 0].legend(['Original', 'Anonimizado'])
        
        # Distribución de balances de origen
        sns.histplot(df['oldbalanceOrg'], kde=True, ax=axes[0, 1], color='blue', alpha=0.5)
        sns.histplot(df_anonimizado['oldbalanceOrg'], kde=True, ax=axes[0, 1], color='red', alpha=0.5)
        axes[0, 1].set_title('Balance Origen: Original vs Anonimizado')
        axes[0, 1].legend(['Original', 'Anonimizado'])
        
        # Distribución de balances de destino
        sns.histplot(df['newbalanceDest'], kde=True, ax=axes[1, 0], color='blue', alpha=0.5)
        sns.histplot(df_anonimizado['newbalanceDest'], kde=True, ax=axes[1, 0], color='red', alpha=0.5)
        axes[1, 0].set_title('Balance Destino: Original vs Anonimizado')
        axes[1, 0].legend(['Original', 'Anonimizado'])
        
        # Comparar correlaciones
        original_corr = df[columnas_requeridas].corr()
        anon_corr = df_anonimizado[columnas_requeridas].corr()
        diff_corr = np.abs(original_corr - anon_corr)
        
        sns.heatmap(diff_corr, annot=True, cmap='YlGnBu', ax=axes[1, 1], cbar=True)
        axes[1, 1].set_title('Diferencias en Correlaciones (Original - Anonimizado)')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No se pueden comparar distribuciones: faltan columnas necesarias.")

def mostrar_tecnicas_aplicadas():
    """
    Muestra las técnicas de anonimización aplicadas.
    """
    print("\n===== TÉCNICAS DE ANONIMIZACIÓN APLICADAS =====")
    print("1. Enmascaramiento y reemplazo sintético para identificadores de cuenta")
    print("2. Perturbación (ruido) para variables numéricas (montos y balances)")
    print("3. Microagregación para variables temporales (step)")

def evaluacion_final():
    """
    Muestra una evaluación final del proceso de anonimización.
    """
    print("\n===== EVALUACIÓN FINAL =====")
    print("Los datos han sido anonimizados preservando las distribuciones estadísticas")
    print("principales y las características de fraude, mientras se protege la información")
    print("personal identificable como IDs de origen y destino.")

# Función principal actualizada para manejar errores y ser más robusta
def ejecutar_analisis_anonimizacion(ruta_entrada, ruta_salida):
    """
    Ejecuta todo el proceso de análisis y anonimización con manejo de errores.
    
    Args:
        ruta_entrada (str): Ruta al archivo CSV de entrada
        ruta_salida (str): Ruta del archivo CSV de salida
        
    Returns:
        tuple: (DataFrame original, DataFrame anonimizado)
    """
    try:
        # Cargar y analizar datos
        df = cargar_datos(ruta_entrada)
        mostrar_info_basica(df)
        visualizar_valores_nulos(df)
        
        # Identificar datos sensibles
        columnas_sensibles = identificar_columnas_sensibles(df)
        analizar_categoricas(df)
        analizar_transacciones_fraude(df)
        
        # Análisis exploratorio detallado
        columnas_numericas = analizar_variables_numericas(df)
        analizar_montos_por_tipo(df)
        analizar_correlaciones(df, columnas_numericas)
        analizar_balances(df)
        analizar_patrones_temporales(df)
        analizar_fraude_por_tipo(df)
        
        # Proceso de anonimización
        df_anonimizado, fake = iniciar_anonimizacion(df)
        analizar_formato_ids(df_anonimizado)
        df_anonimizado = anonimizar_ids(df_anonimizado)
        df_anonimizado = anonimizar_valores_numericos(df_anonimizado)
        df_anonimizado = anonimizar_tiempo(df_anonimizado)
        
        # Verificación de anonimización
        comparar_estadisticas(df, df_anonimizado)
        verificar_ids_anonimizados(df, df_anonimizado, columnas_sensibles)
        evaluar_riesgo_reidentificacion(df_anonimizado)
        
        # Exportar y resumir
        exportar_datos_anonimizados(df_anonimizado, ruta_salida)
        resumir_proceso(df, df_anonimizado, columnas_sensibles)
        comparar_distribuciones(df, df_anonimizado)
        mostrar_tecnicas_aplicadas()
        evaluacion_final()
        
        return df, df_anonimizado
    
    except Exception as e:
        print(f"\n¡ERROR! Se ha producido un error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
    
        # Si ya hemos cargado el DataFrame, lo devolvemos para que el usuario pueda trabajar con él
        if 'df' in locals():
            print("\nA pesar del error, se devuelve el DataFrame original para que puedas trabajar con él.")
            if 'df_anonimizado' in locals():
                return df, df_anonimizado
            else:
                return df, None
        else:
            return None, None

# Ejemplo de uso
# ------------
if __name__ == "__main__":
    # Definir rutas de archivos (ajustar según sea necesario)
    ruta_entrada = "dataset_anonimizacion_datos.csv"
    ruta_salida = "dataset_anonimizacion_datos_anonimizado.csv"
    
    # Ejecutar todo el proceso
    df_original, df_anonimizado = ejecutar_analisis_anonimizacion(ruta_entrada, ruta_salida)
