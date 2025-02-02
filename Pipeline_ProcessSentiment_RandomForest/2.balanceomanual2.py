import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
import joblib

# ----------------------------
# Configuración de rutas
# ----------------------------
transcripciones_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\audios_txt"  # Ajusta la ruta a la carpeta de transcripciones
muestra_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\MuestraEntrenamiento.xlsx"   # Ajusta la ruta a la muestra de archivos
etiquetas_reales_path = r"C:\Users\Andrea\OneDrive\Documentos\1.Maestria\ProyectoGrado\Dataset_Global_Sentiment.xlsx"  # Ajusta la ruta al archivo de etiquetas reales
model_base_path = "modelo_rf2"  # Carpeta donde se guardarán los resultados
os.makedirs(model_base_path, exist_ok=True)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def ajustar_nombres_archivos(df, columna):
    """Asegura que los nombres en la columna especificada terminen en '.txt'."""
    if not df[columna].str.endswith('.txt').all():
        df[columna] = df[columna] + '.txt'
    return df

def leer_transcripciones(folder_path):
    """Lee todos los archivos .txt de la carpeta y retorna un DataFrame con 'Archivo' y 'Texto'."""
    datos = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    texto = f.read()
                    datos.append({"Archivo": file, "Texto": texto})
    return pd.DataFrame(datos)

def preprocesar_texto(texto):
    """Limpia el texto: lo pasa a minúsculas, elimina caracteres no alfabéticos y espacios extra."""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extraer_speaker_1(texto):
    """Extrae el contenido de cada línea que comience con 'Speaker 1:' y lo une en un solo string."""
    lineas = texto.split('\n')
    speaker_1 = [linea.split(':', 1)[1].strip() for linea in lineas if linea.startswith("Speaker 1:")]
    return ' '.join(speaker_1)

def guardar_grafico(fig, filename):
    """Guarda la figura en la ruta indicada."""
    fig.tight_layout()
    fig.savefig(os.path.join(model_base_path, filename), dpi=300)
    plt.close(fig)

# ==========================================
# LECTURA DE DATOS
# ==========================================
# 1. Leer todas las transcripciones (dataset completo)
all_data = leer_transcripciones(transcripciones_path)
print(f"Total de conversaciones en el dataset completo: {len(all_data)}")

# 2. Cargar las etiquetas reales (se asume que el archivo contiene al menos 'conversation_id' y 'overall_sentiment')
dataset_etiquetas = pd.read_excel(etiquetas_reales_path)
dataset_etiquetas = ajustar_nombres_archivos(dataset_etiquetas, 'conversation_id')

# 3. Generar el subconjunto etiquetado (solo las conversaciones para las que hay etiqueta)
labeled_data = all_data.merge(
    dataset_etiquetas.rename(columns={'conversation_id': 'Archivo'}),
    on="Archivo", how="inner"
).dropna(subset=['overall_sentiment'])
print(f"Total de conversaciones etiquetadas: {len(labeled_data)}")

# ==========================================
# PREPROCESAMIENTO DEL TEXTO
# ==========================================
all_data["Texto"] = all_data["Texto"].apply(extraer_speaker_1).apply(preprocesar_texto)
labeled_data["Texto"] = labeled_data["Texto"].apply(extraer_speaker_1).apply(preprocesar_texto)

# Visualizar la distribución de clases en las 100 muestras etiquetadas (sin balancear)
fig = plt.figure(figsize=(8, 6))
ax = labeled_data['overall_sentiment'].value_counts().plot(
    kind='bar', color=sns.color_palette("Blues", n_colors=3),
    title="Distribución de Clases (100 Conversaciones Etiquetadas)"
)
ax.set_xlabel("Clases")
ax.set_ylabel("Cantidad")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='baseline')
guardar_grafico(fig, "distribucion_clasesetiq_sinbalanceo.png")

# ==========================================
# BALANCEO MANUAL DE CLASES
# ==========================================
print("\nDistribución antes del balanceo:")
print(labeled_data['overall_sentiment'].value_counts())

# Identificar la clase mayoritaria y las minoritarias
clase_mayoritaria = labeled_data['overall_sentiment'].value_counts().idxmax()
data_majority = labeled_data[labeled_data['overall_sentiment'] == clase_mayoritaria]
data_minority = labeled_data[labeled_data['overall_sentiment'] != clase_mayoritaria]
# Sobremuestreo: se muestrea con reemplazo las clases minoritarias hasta tener la misma cantidad que la mayoritaria
data_minority_oversampled = data_minority.sample(len(data_majority), replace=True, random_state=42)
labeled_data_balanced = pd.concat([data_majority, data_minority_oversampled])
print("\nDistribución después del balanceo (Manual):")
print(labeled_data_balanced['overall_sentiment'].value_counts())

# Graficar la distribución resultante del balanceo manual
fig = plt.figure(figsize=(8, 6))
ax = labeled_data_balanced['overall_sentiment'].value_counts().plot(
    kind='bar', color=sns.color_palette("Blues", n_colors=3),
    title="Distribución de Clases (Balanceo Manual)"
)
ax.set_xlabel("Clases")
ax.set_ylabel("Cantidad")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='baseline')
guardar_grafico(fig, "distribucion_clases_balanceadas_manual.png")

# ==========================================
# CONVERSIÓN A TF-IDF (USANDO LOS 100 ETIQUETADOS BALANCEADOS)
# ==========================================
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_labeled = vectorizer.fit_transform(labeled_data_balanced["Texto"])
y_labeled = labeled_data_balanced["overall_sentiment"]

# ==========================================
# ENTRENAMIENTO DEL MODELO CON GRID SEARCH
# ==========================================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    clf, param_grid, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1, return_train_score=True
)
grid_search.fit(X_labeled, y_labeled)
best_model = grid_search.best_estimator_
print("Mejor modelo encontrado:", best_model)

# ==========================================
# EVALUACIÓN DEL MODELO SOBRE LAS 100 MUESTRAS CON VALIDACIÓN CRUZADA
# ==========================================
# Utilizamos cross_val_predict para obtener predicciones para cada una de las 100 muestras etiquetadas
estimator = clone(best_model)
y_pred_cv = cross_val_predict(estimator, X_labeled, y_labeled, cv=5)
labels_ordenadas = sorted(y_labeled.unique())
cm_total = confusion_matrix(y_labeled, y_pred_cv, labels=labels_ordenadas)

fig = plt.figure(figsize=(8, 6))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_ordenadas, yticklabels=labels_ordenadas)
plt.title("Matriz de Confusión - 100 Datos Etiquetados (Validación Cruzada con balanceo manual)")
plt.xlabel("Predicción")
plt.ylabel("Real")
guardar_grafico(fig, "matriz_confusion_cv_100etiq_manual.png")

accuracy = accuracy_score(y_labeled, y_pred_cv)
precision = precision_score(y_labeled, y_pred_cv, average='weighted')
recall = recall_score(y_labeled, y_pred_cv, average='weighted')
f1 = f1_score(y_labeled, y_pred_cv, average='weighted')

print("Métricas en el conjunto de 100 datos etiquetados (validación cruzada con balanceo manual):")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# ==========================================
# GENERAR PREDICCIONES PARA TODO EL DATASET
# ==========================================
X_all = vectorizer.transform(all_data["Texto"])
all_data["Prediccion"] = best_model.predict(X_all)
archivo_predicciones = os.path.join(model_base_path, "predicciones_todo_balanceomanual.csv")
all_data.to_csv(archivo_predicciones, index=False, encoding="utf-8-sig")
print(f"Predicciones para todas las conversaciones guardadas en: {archivo_predicciones}")

print("Script completado con éxito.")
