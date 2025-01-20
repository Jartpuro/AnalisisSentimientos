import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

# Cargar modelo, vectorizador y resultados
model_path = "modelo_rf"
best_model = joblib.load(f"{model_path}/random_forest_model.pkl")
vectorizer = joblib.load(f"{model_path}/tfidf_vectorizer.pkl")
resultados_prueba = pd.read_csv(f"{model_path}/resultados_prueba.csv")

# Generar un DataFrame con métricas
def calcular_metricas(model, X_train, y_train):
    """Calcula métricas clave del modelo y retorna un DataFrame con los resultados."""
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    metrics = {
        "F1-Score (Validación Cruzada)": [scores.mean()],
        "F1-Score Std": [scores.std()],
        "Accuracy": [accuracy_score(y_train, model.predict(X_train))],
        "Precision": [precision_score(y_train, model.predict(X_train), average='weighted')],
        "Recall": [recall_score(y_train, model.predict(X_train), average='weighted')]
    }
    return pd.DataFrame(metrics)

# Preparar datos
resultados_prueba["Texto"].fillna("", inplace=True)
X_train = vectorizer.transform(resultados_prueba["Texto"])
y_train = resultados_prueba["Predicción"]  # Usar la columna Predicción como etiquetas reales

# Preparar datos
resultados_prueba["Texto"].fillna("", inplace=True)
X_train = vectorizer.transform(resultados_prueba["Texto"])
y_train = resultados_prueba["Predicción"]  # Usar la columna Predicción como etiquetas reales

# Verificar la distribución de etiquetas y la alineación de datos
print("Distribución de etiquetas en y_train:")
print(y_train.value_counts())

print("Tamaño de X_train:", X_train.shape)
print("Tamaño de y_train:", len(y_train))

# Calcular métricas
metricas = calcular_metricas(best_model, X_train, y_train)

# Mostrar métricas en una tabla bonita
import tabulate
print(tabulate.tabulate(metricas, headers='keys', tablefmt='grid', showindex=False))

# Mostrar distribución de los resultados predichos
def mostrar_distribucion_resultados(df, columna):
    """Genera un gráfico de barras para mostrar la distribución de resultados."""
    distribucion = df[columna].value_counts()
    distribucion.plot(kind='bar', color=['blue', 'green', 'red'], figsize=(8, 6))
    plt.title("Distribución de Resultados Predichos")
    plt.xlabel("Clases")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{model_path}/distribucion_resultados.png", dpi=300)
    plt.show()

# Llamar la función para mostrar la distribución de predicciones
mostrar_distribucion_resultados(resultados_prueba, "Predicción")

# Guardar resultados en un archivo CSV
metricas.to_csv(f"{model_path}/metricas_modelo.csv", index=False, encoding="utf-8-sig")
print("\nMétricas guardadas en modelo_rf/metricas_modelo.csv.")

# Guardar la tabla como una imagen
def guardar_tabla_como_imagen(df, filename):
    """Guarda un DataFrame como una imagen usando matplotlib."""
    if not df.empty:
        fig, ax = plt.subplots(figsize=(8, 2))  # Ajustar tamaño según contenido
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nLa tabla de métricas ha sido guardada como una imagen en {filename}.")
    else:
        print("\nNo se generaron métricas, no se puede guardar la tabla como imagen.")

# Guardar la tabla como imagen
guardar_tabla_como_imagen(metricas, f"{model_path}/metricas_modelo.png")