import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import joblib

# Configuración de rutas
transcripciones_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\audios_txt"
muestra_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\MuestraEntrenamiento.xlsx"
etiquetas_reales_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\Dataset_Global_Sentiment.xlsx"
model_base_path = "modelo_rf"

# Leer lista de archivos seleccionados
muestra_entrenamiento = pd.read_excel(muestra_path)
nombres_archivos_entrenamiento = set(muestra_entrenamiento['Archivo'])

# Leer etiquetas reales
dataset_etiquetas = pd.read_excel(etiquetas_reales_path)

# Asegurar que la columna 'Archivo' incluye las extensiones si no están
def ajustar_nombres_archivos(df, columna):
    if not df[columna].str.endswith('.txt').all():
        df[columna] = df[columna] + '.txt'
    return df

dataset_etiquetas = ajustar_nombres_archivos(dataset_etiquetas, 'conversation_id')

# Función para leer transcripciones
def leer_transcripciones(folder_path):
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
datos_entrenamiento = datos_entrenamiento.merge(
    dataset_etiquetas.rename(columns={'conversation_id': 'Archivo'}), on="Archivo", how="left"
).dropna(subset=['overall_sentiment'])

# Preprocesamiento básico
def preprocesar_texto(texto):
    import re
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extraer_speaker_1(texto):
    lineas = texto.split('\n')
    speaker_1 = [linea.split(':', 1)[1].strip() for linea in lineas if linea.startswith("Speaker 1:")]
    return ' '.join(speaker_1)

datos_entrenamiento["Texto"] = datos_entrenamiento["Texto"].apply(extraer_speaker_1).apply(preprocesar_texto)
datos_prueba["Texto"] = datos_prueba["Texto"].apply(extraer_speaker_1).apply(preprocesar_texto)

# Función para agregar etiquetas a las gráficas de barras
def agregar_etiquetas(ax):
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

# Visualizar distribución inicial de etiquetas
plt.figure(figsize=(8, 6))
ax = datos_entrenamiento['overall_sentiment'].value_counts().plot(
    kind='bar', color=sns.color_palette("Blues", n_colors=3), title="Distribución Inicial de Clases"
)
plt.xlabel("Clases")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
agregar_etiquetas(ax)
plt.tight_layout()
os.makedirs(model_base_path, exist_ok=True)
plt.savefig(os.path.join(model_base_path, "distribucion_inicial_clases.png"), dpi=300)
plt.close()

# Convertir texto a TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(datos_entrenamiento["Texto"])
y_train = datos_entrenamiento["overall_sentiment"]
X_test = vectorizer.transform(datos_prueba["Texto"])

# Función para guardar archivos
def guardar_archivo(dataframe, nombre_archivo, enfoque):
    enfoque_path = os.path.join(model_base_path, enfoque)
    os.makedirs(enfoque_path, exist_ok=True)
    archivo = os.path.join(enfoque_path, nombre_archivo)
    dataframe.to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"\nArchivo guardado en: {archivo}")

# Entrenar, registrar métricas y generar gráficos
def entrenar_y_registrar(X_train, y_train, X_test, enfoque, param_grid):
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
    etiquetas = pd.DataFrame({"Archivo": datos_prueba["Archivo"], "Predicción": y_pred})

    # Guardar métricas y etiquetas
    guardar_archivo(metricas, f"metricas_{enfoque}.csv", enfoque)
    guardar_archivo(etiquetas, f"resultados_{enfoque}.csv", enfoque)

    # Matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_train, best_model.predict(X_train), labels=y_train.unique())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y_train.unique(), yticklabels=y_train.unique())
    plt.title(f"Matriz de Confusión - {enfoque}")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    matriz_path = os.path.join(model_base_path, enfoque, f"matriz_confusion_{enfoque}.png")
    plt.savefig(matriz_path, dpi=300)
    plt.close()

    # Visualizar distribución de predicciones
    plt.figure(figsize=(8, 6))
    ax = etiquetas["Predicción"].value_counts().plot(
        kind='bar', color=sns.color_palette("Blues", n_colors=3), title=f"Distribución de Resultados Predichos - {enfoque}"
    )
    plt.xlabel("Clases")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    agregar_etiquetas(ax)
    plt.tight_layout()
    distribucion_path = os.path.join(model_base_path, enfoque, f"distribucion_predicciones_{enfoque}.png")
    plt.savefig(distribucion_path, dpi=300)
    plt.close()

    # Resumen del mejor modelo
    mejor_modelo_metricas = {
        "F1-Score (Validación Cruzada)": grid_search.best_score_,
        "F1-Score Std": resultados_cv.loc[grid_search.best_index_, 'std_test_score'],
        "Accuracy": accuracy_score(y_train, best_model.predict(X_train)),
        "Precision": precision_score(y_train, best_model.predict(X_train), average='weighted'),
        "Recall": recall_score(y_train, best_model.predict(X_train), average='weighted')
    }

    resumen_df = pd.DataFrame([mejor_modelo_metricas])
    guardar_archivo(resumen_df, f"resumen_mejor_modelo_{enfoque}.csv", enfoque)

    return metricas, etiquetas, resumen_df

# Definir hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Modelo Sin Balanceo
metricas_sin_balanceo, etiquetas_sin_balanceo, resumen_mejor_modelo = entrenar_y_registrar(X_train, y_train, X_test, "sin_balanceo", param_grid)

# Resumen en Consola
print("\nResumen del mejor modelo sin balanceo:")
print(resumen_mejor_modelo)

print("\nScript completado con éxito. Revisa la carpeta 'modelo_rf' para los archivos generados.")
