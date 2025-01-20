import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Función para graficar la distribución de clases antes y después del balanceo
def graficar_distribucion_clases(y_train_original, y_train_balanceado, enfoque):
    # Contar las clases antes y después del balanceo
    distribucion_original = y_train_original.value_counts()
    distribucion_balanceado = y_train_balanceado.value_counts()

    # Crear una figura con dos gráficos de barras
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Gráfico para la distribución original
    ax[0].bar(distribucion_original.index, distribucion_original.values, color='skyblue')
    ax[0].set_title(f'Distribución Original de Clases ({enfoque})')
    ax[0].set_xlabel('Clase')
    ax[0].set_ylabel('Frecuencia')

    # Gráfico para la distribución después del balanceo
    ax[1].bar(distribucion_balanceado.index, distribucion_balanceado.values, color='lightcoral')
    ax[1].set_title(f'Distribución de Clases Balanceada ({enfoque})')
    ax[1].set_xlabel('Clase')
    ax[1].set_ylabel('Frecuencia')

    # Ajustar el espacio entre los gráficos
    plt.tight_layout()

    # Guardar la imagen
    plt.savefig(f"modelo_rf/distribucion_clases_{enfoque.replace(' ', '_').lower()}.png")
    plt.show()

# Función para obtener la distribución antes y después del balanceo para cada enfoque
def obtener_distribucion_clases(X_train, y_train, X_train_balanceado, y_train_balanceado, enfoque):
    graficar_distribucion_clases(y_train, y_train_balanceado, enfoque)

# Función para guardar las métricas
def guardar_archivo(dataframe, enfoque, tipo, carpeta="modelo_rf"):
    """
    Guarda un archivo (métricas o etiquetas) con un nombre basado en el enfoque y tipo.
    
    Args:
    - dataframe (pd.DataFrame): DataFrame a guardar.
    - enfoque (str): Nombre del enfoque ("sin balanceo", "balanceo manual", etc.).
    - tipo (str): Tipo de archivo ("metricas" o "resultados").
    - carpeta (str): Carpeta donde se guardará el archivo.
    """
    os.makedirs(carpeta, exist_ok=True)
    archivo = f"{carpeta}/{tipo}_{enfoque.replace(' ', '_').lower()}.csv"
    dataframe.to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"\nArchivo {tipo} guardado en: {archivo}")

# Función para entrenar y obtener métricas
def entrenar_y_registrar(X_train, y_train, X_test, y_test, enfoque, param_grid):
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    
    # Resultados de validación cruzada
    resultados_cv = pd.DataFrame(grid_search.cv_results_)
    metricas = resultados_cv[[ 
        'param_n_estimators', 
        'param_max_depth', 
        'param_min_samples_split', 
        'mean_test_score', 
        'std_test_score', 
        'mean_train_score'
    ]]
    metricas.rename(columns={ 
        'mean_test_score': 'F1_Score_Test', 
        'std_test_score': 'F1_Score_Std', 
        'mean_train_score': 'F1_Score_Train'
    }, inplace=True)
    
    # Predicciones del mejor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }

    # Guardar métricas y etiquetas
    guardar_archivo(metricas, enfoque, "metricas")
    guardar_archivo(pd.DataFrame({"Archivo": datos_prueba["Archivo"], "Predicción": y_pred}), enfoque, "resultados")

    return metricas, y_pred

# Configuración de rutas y lectura de los datos
transcripciones_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\audios_txt"
muestra_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\MuestraEntrenamiento.xlsx"
etiquetas_reales_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\Dataset_Global_Sentiment.xlsx"

# Leer lista de archivos seleccionados
muestra_entrenamiento = pd.read_excel(muestra_path)
nombres_archivos_entrenamiento = set(muestra_entrenamiento['Archivo'])

# Leer etiquetas reales
dataset_etiquetas = pd.read_excel(etiquetas_reales_path)

# Asegurar que la columna 'Archivo' incluye las extensiones si no están
def ajustar_nombres_archivos(df, columna):
    """Asegura que los nombres de archivo tengan la extensión .txt."""
    if not df[columna].str.endswith('.txt').all():
        df[columna] = df[columna] + '.txt'
    return df

dataset_etiquetas = ajustar_nombres_archivos(dataset_etiquetas, 'conversation_id')

# Función para leer transcripciones
def leer_transcripciones(folder_path):
    """Lee todos los archivos de texto en la carpeta especificada."""
    datos = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    texto = f.read()
                    datos.append({"Archivo": file, "Texto": texto})
    return pd.DataFrame(datos)

# Leer todas las transcripciones
todas_transcripciones = leer_transcripciones(transcripciones_path)

# Separar las transcripciones en entrenamiento y prueba
datos_entrenamiento = todas_transcripciones[todas_transcripciones['Archivo'].isin(nombres_archivos_entrenamiento)]
datos_prueba = todas_transcripciones[~todas_transcripciones['Archivo'].isin(nombres_archivos_entrenamiento)]

# Unir las etiquetas reales con las transcripciones
datos_entrenamiento = datos_entrenamiento.merge(dataset_etiquetas.rename(columns={'conversation_id': 'Archivo'}), on="Archivo", how="left").dropna(subset=['overall_sentiment'])

# Preprocesamiento básico
def preprocesar_texto(texto):
    """Limpia y transforma texto para análisis."""
    import re
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extraer_speaker_1(texto):
    """Extrae solo las líneas de Speaker 1 de una transcripción."""
    lineas = texto.split('\n')
    speaker_1 = [linea.split(':', 1)[1].strip() for linea in lineas if linea.startswith("Speaker 1:")]
    return ' '.join(speaker_1)

datos_entrenamiento["Texto"] = datos_entrenamiento["Texto"].apply(extraer_speaker_1).apply(preprocesar_texto)
datos_prueba["Texto"] = datos_prueba["Texto"].apply(extraer_speaker_1).apply(preprocesar_texto)

# Convertir texto a TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(datos_entrenamiento["Texto"])
y_train = datos_entrenamiento["overall_sentiment"]
X_test = vectorizer.transform(datos_prueba["Texto"])

# Modelo Sin Balanceo
obtener_distribucion_clases(X_train, y_train, X_train, y_train, "sin_balanceo")
metricas_sin_balanceo, y_pred_sin_balanceo = entrenar_y_registrar(X_train, y_train, X_test, datos_prueba["overall_sentiment"], "sin_balanceo", {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
})

# Modelo con Balanceo Manual
train_data = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
train_data['Etiqueta'] = y_train.values
majority = train_data[train_data['Etiqueta'] == 'Neutral']
minority = train_data[train_data['Etiqueta'] != 'Neutral']
minority_oversampled = minority.sample(len(majority), replace=True, random_state=42)
balanced_data = pd.concat([majority, minority_oversampled])
X_train_bal = balanced_data.drop('Etiqueta', axis=1)
y_train_bal = balanced_data['Etiqueta']

obtener_distribucion_clases(X_train, y_train, X_train_bal, y_train_bal, "balanceo_manual")
metricas_balanceo_manual, y_pred_balanceo_manual = entrenar_y_registrar(X_train_bal, y_train_bal, X_test, datos_prueba["overall_sentiment"], "balanceo_manual", {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
})

# Modelo con Balanceo Equitativo
target_size = min(train_data['Etiqueta'].value_counts())
balanced_data_eq = pd.concat([
    train_data[train_data['Etiqueta'] == label].sample(target_size, replace=True, random_state=42)
    for label in train_data['Etiqueta'].unique()
])
X_train_eq = balanced_data_eq.drop('Etiqueta', axis=1)
y_train_eq = balanced_data_eq['Etiqueta']

obtener_distribucion_clases(X_train, y_train, X_train_eq, y_train_eq, "balanceo_equitativo")
metricas_balanceo_eq, y_pred_balanceo_eq = entrenar_y_registrar(X_train_eq, y_train_eq, X_test, datos_prueba["overall_sentiment"], "balanceo_equitativo", {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
})

print("\nScript completado con éxito. Revisa la carpeta 'modelo_rf' para los archivos generados.")
