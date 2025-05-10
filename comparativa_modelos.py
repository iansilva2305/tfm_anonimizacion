# Análisis Comparativo entre Privacidad y Rendimiento para Detección de Fraude
# =========================================================================
# Este módulo expande el análisis para evaluar el impacto de diferentes niveles 
# de anonimización en el rendimiento de los modelos de detección de fraude

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def evaluar_impacto_k_anonimato(df, columnas_modelo, columna_objetivo='isFraud', 
                               valores_k=[2, 5, 10, 20, 50], 
                               random_state=42, test_size=0.3):
    """
    Evalúa el impacto de diferentes valores de k-anonimato en el rendimiento del modelo.
    
    Args:
        df (pd.DataFrame): DataFrame original
        columnas_modelo (list): Columnas para el modelo
        columna_objetivo (str): Columna objetivo
        valores_k (list): Lista de valores de k a evaluar
        random_state (int): Semilla para reproducibilidad
        test_size (float): Proporción de datos para test
        
    Returns:
        pd.DataFrame: Resultados comparativos
    """
    print("Evaluando impacto de diferentes valores de k-anonimato...")
    
    # Resultados para el modelo base (sin anonimización)
    # Entrenamos el modelo base con los datos originales
    X = df[columnas_modelo].copy()
    y = df[columna_objetivo].copy()
    
    # Preprocesamiento básico
    for col in X.select_dtypes(include=['object', 'category']).columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                       random_state=random_state)
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=random_state)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    precision_base = accuracy_score(y_test, y_pred)
    sensibilidad_base = recall_score(y_test, y_pred)
    f1_base = f1_score(y_test, y_pred)
    
    # Almacenar resultados
    resultados = {
        'k': ['Original'],
        'precisión': [precision_base],
        'sensibilidad': [sensibilidad_base],
        'f1_score': [f1_base],
        'pérdida_precisión': [0.0]
    }
    
    # Evaluar cada valor de k
    for k in valores_k:
        # Aplicar k-anonimato (simulado mediante agrupación)
        df_anon = df.copy()
        
        # Agrupar amount en rangos (simulando k-anonimato)
        if 'amount' in df_anon.columns:
            # El tamaño de los rangos variará según k
            # A mayor k, rangos más amplios para garantizar al menos k registros por grupo
            factor_amplitud = int(k / 2)  # Factor para ajustar el tamaño de los rangos
            max_amount = df_anon['amount'].max()
            
            # Rangos más pequeños para k pequeño, más grandes para k grande
            if k <= 5:
                bins = [0, 500, 1000, 2500, 5000, 10000, max_amount + 1]
            elif k <= 10:
                bins = [0, 1000, 5000, 10000, 50000, max_amount + 1]
            elif k <= 20:
                bins = [0, 2000, 10000, 50000, max_amount + 1]
            else:
                bins = [0, 5000, 20000, max_amount + 1]
                
            labels = [f'rango_{i}' for i in range(len(bins)-1)]
            df_anon['amount_group'] = pd.cut(df_anon['amount'], bins=bins, labels=labels, right=False)
            
            # Reemplazar amount por el valor medio del rango (simulando pérdida de precisión)
            # Creamos un mapeo de etiquetas a valores medios
            mid_values = {}
            for i in range(len(bins)-1):
                mid_values[labels[i]] = (bins[i] + bins[i+1]) / 2
                
            # Reemplazar amount por el valor medio del rango
            df_anon['amount'] = df_anon['amount_group'].map(mid_values)
            
            # Eliminar la columna de grupo
            df_anon.drop('amount_group', axis=1, inplace=True)
        
        # Agrupar step en bloques de tiempo (simulando k-anonimato)
        if 'step' in df_anon.columns:
            max_step = df_anon['step'].max()
            
            # Ajustar el tamaño de los bloques según k
            if k <= 5:
                block_size = 6  # Bloques de 6 horas
            elif k <= 10:
                block_size = 8  # Bloques de 8 horas
            elif k <= 20:
                block_size = 12  # Bloques de 12 horas
            else:
                block_size = 24  # Bloques de 24 horas
                
            # Agrupar step en bloques
            df_anon['step'] = (df_anon['step'] // block_size) * block_size
        
        # Entrenar modelo con datos anonimizados
        X_anon = df_anon[columnas_modelo].copy()
        y_anon = df_anon[columna_objetivo].copy()
        
        # Preprocesamiento
        for col in X_anon.select_dtypes(include=['object', 'category']).columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X_anon[col] = le.fit_transform(X_anon[col])
        
        X_train_anon, X_test_anon, y_train_anon, y_test_anon = train_test_split(
            X_anon, y_anon, test_size=test_size, random_state=random_state
        )
        
        modelo_anon = RandomForestClassifier(n_estimators=100, random_state=random_state)
        modelo_anon.fit(X_train_anon, y_train_anon)
        y_pred_anon = modelo_anon.predict(X_test_anon)
        
        precision_anon = accuracy_score(y_test_anon, y_pred_anon)
        sensibilidad_anon = recall_score(y_test_anon, y_pred_anon)
        f1_anon = f1_score(y_test_anon, y_pred_anon)
        
        # Calcular pérdida de precisión
        perdida_precision = ((precision_base - precision_anon) / precision_base) * 100
        
        # Almacenar resultados
        resultados['k'].append(f'k={k}')
        resultados['precisión'].append(precision_anon)
        resultados['sensibilidad'].append(sensibilidad_anon)
        resultados['f1_score'].append(f1_anon)
        resultados['pérdida_precisión'].append(perdida_precision)
    
    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar resultados
    print("\nResultados comparativos:")
    display(df_resultados)
    
    # Visualizar resultados
    plt.figure(figsize=(14, 8))
    
    # Gráfico 1: Precisión vs k
    plt.subplot(2, 2, 1)
    sns.barplot(x='k', y='precisión', data=df_resultados)
    plt.title('Precisión vs. Nivel de k-anonimato')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=45)
    
    # Gráfico 2: Sensibilidad vs k
    plt.subplot(2, 2, 2)
    sns.barplot(x='k', y='sensibilidad', data=df_resultados)
    plt.title('Sensibilidad vs. Nivel de k-anonimato')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=45)
    
    # Gráfico 3: F1-Score vs k
    plt.subplot(2, 2, 3)
    sns.barplot(x='k', y='f1_score', data=df_resultados)
    plt.title('F1-Score vs. Nivel de k-anonimato')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=45)
    
    # Gráfico 4: Pérdida de precisión vs k
    plt.subplot(2, 2, 4)
    sns.barplot(x='k', y='pérdida_precisión', data=df_resultados)
    plt.title('Pérdida de Precisión (%) vs. Nivel de k-anonimato')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df_resultados

def evaluar_tradeoff_privacidad_utilidad(df_resultados, umbral_cumplimiento=10.0):
    """
    Evalúa el equilibrio entre privacidad y utilidad basado en los resultados de k-anonimato.
    
    Args:
        df_resultados (pd.DataFrame): DataFrame con resultados de diferentes valores de k
        umbral_cumplimiento (float): Umbral de pérdida de precisión aceptable (%)
        
    Returns:
        dict: Recomendación de valor óptimo de k
    """
    print("\n===== EVALUACIÓN DEL EQUILIBRIO PRIVACIDAD-UTILIDAD =====")
    
    # Filtrar resultados para k > 1 (excluyendo el modelo original)
    resultados_k = df_resultados[df_resultados['k'] != 'Original'].copy()
    
    # Convertir columna k a numérico (extrayendo el número de 'k=X')
    resultados_k['k_valor'] = resultados_k['k'].apply(lambda x: int(x.split('=')[1]))
    
    # Identificar valores de k que cumplen el umbral de pérdida de precisión
    cumplen_umbral = resultados_k[resultados_k['pérdida_precisión'] <= umbral_cumplimiento]
    
    if cumplen_umbral.empty:
        print(f"No se encontraron valores de k que cumplan el umbral de pérdida de precisión ({umbral_cumplimiento}%).")
        # Si ninguno cumple, seleccionar el que tenga menor pérdida
        k_optimo = resultados_k.loc[resultados_k['pérdida_precisión'].idxmin()]
        recomendacion = {
            'k_optimo': int(k_optimo['k'].split('=')[1]),
            'precision': k_optimo['precisión'],
            'perdida': k_optimo['pérdida_precisión'],
            'cumple_umbral': False,
            'mensaje': "Ningún valor de k cumple el umbral de pérdida. Se recomienda el valor con menor pérdida."
        }
    else:
        # Seleccionar el mayor valor de k que cumpla el umbral
        k_optimo = cumplen_umbral.loc[cumplen_umbral['k_valor'].idxmax()]
        recomendacion = {
            'k_optimo': int(k_optimo['k_valor']),
            'precision': k_optimo['precisión'],
            'perdida': k_optimo['pérdida_precisión'],
            'cumple_umbral': True,
            'mensaje': f"Se recomienda k={int(k_optimo['k_valor'])} como valor óptimo balanceando privacidad y utilidad."
        }
    
    # Visualizar el equilibrio entre privacidad y utilidad
    plt.figure(figsize=(10, 6))
    plt.scatter(resultados_k['k_valor'], resultados_k['pérdida_precisión'], s=100, alpha=0.7)
    
    # Marcar el valor óptimo
    plt.scatter([recomendacion['k_optimo']], [recomendacion['perdida']], 
                s=200, c='red', marker='*', label='Valor óptimo')
    
    # Línea de umbral
    plt.axhline(y=umbral_cumplimiento, color='r', linestyle='--', alpha=0.5, 
                label=f'Umbral de pérdida ({umbral_cumplimiento}%)')
    
    plt.title('Equilibrio entre Privacidad (k) y Pérdida de Precisión')
    plt.xlabel('Valor de k (Nivel de Privacidad)')
    plt.ylabel('Pérdida de Precisión (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Mostrar recomendación
    print(f"\nRecomendación: {recomendacion['mensaje']}")
    print(f"  - Valor de k: {recomendacion['k_optimo']}")
    print(f"  - Precisión esperada: {recomendacion['precision']:.4f}")
    print(f"  - Pérdida de precisión: {recomendacion['perdida']:.2f}%")
    
    return recomendacion

def generar_informe_cumplimiento_gdpr(recomendacion, umbral_cumplimiento=10.0):
    """
    Genera un informe de cumplimiento del GDPR basado en las recomendaciones.
    
    Args:
        recomendacion (dict): Recomendación de valor óptimo de k
        umbral_cumplimiento (float): Umbral de pérdida de precisión aceptable (%)
    """
    print("\n===== INFORME DE CUMPLIMIENTO GDPR =====")
    
    # Evaluar el nivel de cumplimiento
    if recomendacion['k_optimo'] >= 10 and recomendacion['perdida'] <= umbral_cumplimiento:
        nivel_cumplimiento = "ALTO"
        color_cumplimiento = 'green'
    elif recomendacion['k_optimo'] >= 5 and recomendacion['perdida'] <= 15.0:
        nivel_cumplimiento = "MEDIO"
        color_cumplimiento = 'orange'
    else:
        nivel_cumplimiento = "BAJO"
        color_cumplimiento = 'red'
    
    # Crear un panel informativo
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Ocultar ejes
    ax.axis('off')
    
    # Título
    ax.text(0.5, 0.95, "INFORME DE CUMPLIMIENTO GDPR", 
            fontsize=18, ha='center', fontweight='bold')
    ax.text(0.5, 0.9, "Detección de Fraude Bancario", 
            fontsize=14, ha='center', style='italic')
    
    # Nivel de cumplimiento
    ax.text(0.5, 0.8, f"Nivel de Cumplimiento: {nivel_cumplimiento}", 
            fontsize=16, ha='center', fontweight='bold', color=color_cumplimiento,
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Detalles de la configuración
    ax.text(0.1, 0.7, "Configuración de Anonimización:", fontsize=14, fontweight='bold')
    ax.text(0.15, 0.65, f"• Valor óptimo de k-anonimato: {recomendacion['k_optimo']}", fontsize=12)
    ax.text(0.15, 0.61, f"• Pérdida de precisión: {recomendacion['perdida']:.2f}%", fontsize=12)
    ax.text(0.15, 0.57, f"• Precisión del modelo: {recomendacion['precision']:.4f}", fontsize=12)
    
    # Cumplimiento de principios GDPR
    ax.text(0.1, 0.5, "Principios GDPR:", fontsize=14, fontweight='bold')
    
    # Evaluar cada principio
    principios = {
        "Minimización de datos (Art. 5.1.c)": recomendacion['k_optimo'] >= 5,
        "Privacidad desde el diseño (Art. 25)": recomendacion['k_optimo'] >= 10,
        "Integridad y confidencialidad (Art. 32)": recomendacion['k_optimo'] >= 10,
        "Equilibrio utilidad-privacidad": recomendacion['perdida'] <= umbral_cumplimiento
    }
    
    y_pos = 0.45
    for principio, cumple in principios.items():
        color = 'green' if cumple else 'red'
        estado = "✓ CUMPLE" if cumple else "✗ NO CUMPLE"
        ax.text(0.15, y_pos, f"• {principio}: {estado}", fontsize=12, color=color)
        y_pos -= 0.04
    
    # Recomendaciones
    ax.text(0.1, 0.25, "Recomendaciones:", fontsize=14, fontweight='bold')
    
    if nivel_cumplimiento == "ALTO":
        ax.text(0.15, 0.2, "• El modelo cumple con los requisitos de privacidad establecidos por el GDPR.", fontsize=12)
        ax.text(0.15, 0.16, "• Se recomienda documentar las técnicas de anonimización aplicadas.", fontsize=12)
        ax.text(0.15, 0.12, "• Implementar mecanismos de monitorización continua del rendimiento.", fontsize=12)
    elif nivel_cumplimiento == "MEDIO":
        ax.text(0.15, 0.2, "• El modelo cumple parcialmente con los requisitos del GDPR.", fontsize=12)
        ax.text(0.15, 0.16, "• Considerar incrementar el valor de k a al menos 10 si es posible.", fontsize=12)
        ax.text(0.15, 0.12, "• Realizar una evaluación de impacto (DPIA) antes del despliegue.", fontsize=12)
    else:
        ax.text(0.15, 0.2, "• El modelo NO cumple con los requisitos mínimos del GDPR.", fontsize=12)
        ax.text(0.15, 0.16, "• Incrementar el valor de k y aplicar técnicas adicionales de anonimización.", fontsize=12)
        ax.text(0.15, 0.12, "• Evaluar la posibilidad de usar privacidad diferencial para mayor protección.", fontsize=12)
    
    # Fecha del informe
    from datetime import datetime
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    ax.text(0.5, 0.05, f"Informe generado el {fecha_actual}", 
            fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.show()

# Función de ejecución principal para análisis comparativo
def ejecutar_analisis_comparativo(df, columnas_modelo, columna_objetivo='isFraud',
                                valores_k=[2, 5, 10, 20, 50], umbral_cumplimiento=10.0):
    """
    Ejecuta un análisis comparativo completo de k-anonimato.
    
    Args:
        df (pd.DataFrame): DataFrame original
        columnas_modelo (list): Columnas para el modelo
        columna_objetivo (str): Columna objetivo
        valores_k (list): Lista de valores de k a evaluar
        umbral_cumplimiento (float): Umbral de pérdida de precisión aceptable (%)
        
    Returns:
        dict: Recomendación de configuración óptima
    """
    print("\n===== INICIANDO ANÁLISIS COMPARATIVO DE PRIVACIDAD-RENDIMIENTO =====")
    
    # Evaluar el impacto de diferentes valores de k
    df_resultados = evaluar_impacto_k_anonimato(
        df, columnas_modelo, columna_objetivo, valores_k
    )
    
    # Evaluar el equilibrio entre privacidad y utilidad
    recomendacion = evaluar_tradeoff_privacidad_utilidad(
        df_resultados, umbral_cumplimiento
    )
    
    # Generar informe de cumplimiento GDPR
    generar_informe_cumplimiento_gdpr(recomendacion, umbral_cumplimiento)
    
    return recomendacion

# Ejemplo de uso
if __name__ == "__main__":
    # Este código se ejecutará solo si este archivo se ejecuta como script
    # No cuando se importe como módulo
    
    print("Este módulo proporciona funciones para realizar análisis comparativos")
    print("de k-anonimato en modelos de detección de fraude.")
    print("\nEjemplo de uso:")
    print("from eda_anonimizacion_fraud_detection import cargar_datos")
    print("from comparativa_modelos import ejecutar_analisis_comparativo")
    print("\ndf = cargar_datos('dataset_anonimizacion_datos.csv')")
    print("columnas_modelo = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']")
    print("recomendacion = ejecutar_analisis_comparativo(df, columnas_modelo)")