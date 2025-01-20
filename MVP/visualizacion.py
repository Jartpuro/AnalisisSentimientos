import os
import pathlib as pl
import streamlit as st
import pandas as pd
from utils import *
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import json

# ===============================
# Pydantic Models
# ===============================
class ConversationSegment(BaseModel):
    speaker_id: str
    text: str
    sentiment: str
    reasons: List[str]

class ConversationAnalysis(BaseModel):
    conversation_id: str
    overall_sentiment: str
    segments: List[ConversationSegment]
    academic_program: str
    notes: str

# ===============================
# Funciones para análisis
# ===============================
load_dotenv()


def analyze_conversation(contenido: str) -> ConversationAnalysis:
    client = OpenAI()  

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",  # Ajusta el modelo a tu necesidad
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente encargado de analizar la conversación y determinar "
                    "el interés del CLIENTE, es decir, la persona denominada como Speaker 1, "
                    "con respecto a la oferta académica o servicios de la universidad.\n\n"
                    
                    "Se deben revisar TODOS los turnos de la conversación (Speaker 0 y Speaker 1), "
                    "solo clasificarás en 'Positivo', 'Neutral' o 'Negativo' los turnos "
                    "de Speaker 0 y Speaker 1.\n\n"
                    
                    "Finalmente, generarás un 'overall_sentiment' que refleje la actitud "
                    "de Speaker 1 a lo largo de toda la conversación.\n\n"
                    
                    "== Criterios para clasificar al cliente (Speaker 1) ==\n"
                    "1) POSITIVO\n"
                    "   - Muestra entusiasmo por la oferta académica.\n"
                    "   - Hace preguntas detalladas o solicita más información.\n"
                    "   - Utiliza frases como '¡Qué interesante!', 'Me encantaría saber más', etc.\n"
                    "   - Acepta programar cita, se muestra dispuesto a inscribirse, compartir la info.\n"
                    "   - Tono de voz amable, entusiasta; ríe o muestra alegría.\n"
                    "   - Agradece la llamada o contacto.\n\n"
                    
                    "2) NEUTRAL\n"
                    "   - Escucha sin mostrar emociones fuertes.\n"
                    "   - Responde con afirmaciones simples ('Entiendo', 'De acuerdo').\n"
                    "   - Indecisión, necesita tiempo para pensar.\n"
                    "   - Tono de voz estable y profesional.\n"
                    "   - No utiliza lenguaje emocional ni muestra entusiasmo/descontento.\n"
                    "   - Interacción cordial pero limitada.\n\n"
                    
                    "3) NEGATIVO\n"
                    "   - Expresa desinterés o rechazo.\n"
                    "   - Frases como 'No me interesa', 'No, gracias'.\n"
                    "   - Critica la oferta o institución.\n"
                    "   - Muestra frustración, molestia, lenguaje cortante.\n"
                    "   - Tono de voz serio, molesto, impaciente; intenta finalizar rápido.\n\n"
                    
                    "=== Consideraciones Adicionales ===\n"
                    " - Observa palabras clave, modismos, sarcasmo, ironía.\n"
                    " - Nota si el sentimiento de Speaker 1 cambia a lo largo de la conversación.\n"
                    " - Anota cualquier evento significativo que influya en el sentimiento.\n\n"
                    
                    "=== Instrucciones de Formato ===\n"
                    "Devuelve la respuesta en formato JSON válido, usando la siguiente estructura:\n\n"
                    "ConversationAnalysis {\n"
                    "  conversation_id: str,\n"
                    "  overall_sentiment: str,  // 'Positivo', 'Neutral' o 'Negativo'\n"
                    "  segments: ConversationSegment[],\n"
                    "  academic_program: str,   // Nombre del programa académico que se le ofrece a Speaker 1\n"
                    "  notes: str\n"
                    "}\n\n"
                    "ConversationSegment {\n"
                    "  speaker_id: str,        // 'Speaker 0' o 'Speaker 1'\n"
                    "  text: str,              // Texto del turno\n"
                    "  sentiment: str,         // 'Positivo', 'Neutral', 'Negativo'\n"
                    "  reasons: str[]          // Lista de las razones de la clasificación.\n"
                    "}\n\n"
                    "Importante:\n"
                    " - No incluyas explicación adicional por fuera de la estructura. \n"
                    " - El 'overall_sentiment' se basa únicamente en la actitud de Speaker 1.\n"
                    " - Si no hay suficiente información para clasificar, usa 'Neutral'.\n"
                )
            },
            {
                "role": "user",
                "content": contenido
            }
        ],
        response_format=ConversationAnalysis
    )

    analysis = completion.choices[0].message.parsed
    return analysis

def read_txt_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
        return content
    except Exception as e:
        return f"Error al leer el archivo: {e}"

# ===============================
# Creación de directorios (utils)
# ===============================
current_path      = pl.Path.cwd()
output_folders    = create_output_directories(current_path)
data_folders_path = create_data_directories(output_folders[0])
imagen_path       = create_img_directories(current_path)

# ===============================
# Streamlit App
# ===============================
# Agregamos algo de CSS para embellecer la interfaz
st.markdown(
    """
    <style>
    /* Fondo del sidebar */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }
    /* Títulos principales */
    .big-title {
        font-size: 2em;
        font-weight: bold;
        color: #1f3b6f;
        margin-bottom: 0.5em;
    }
    /* Subtítulos */
    .sub-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #3f3f3f;
        margin-bottom: 0.2em;
    }
    /* Caja de resultado (por ejemplo para mostrar IDs) */
    .result-box {
        background-color: #eef6ff;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("Menú")
    st.write("Selecciona una opción:")
    model_openai = st.selectbox("", ["gpt-4o", "chatgpt-4o-latest", "gpt-4o-mini", "o1-preview"])
    uploaded_file = st.file_uploader("Cargar archivo de texto", type=["txt"])

st.markdown("<div class='big-title'>Clasificador de Sentimientos</div>", unsafe_allow_html=True)
st.markdown("---")

# Información de integrantes
st.markdown("<div class='sub-title'>Integrantes</div>", unsafe_allow_html=True)
st.markdown("1. **Nombre 1**<br>2. **Nombre 2**<br>3. **Nombre 3**", unsafe_allow_html=True)

# Función adicional para asignar colores e íconos por sentimiento
sentiment_styles = {
    "Positivo": {
        "icon": "✅",
        "color": "#2E8B57",  # verde
    },
    "Neutral": {
        "icon": "⚪",
        "color": "#808080",  # gris
    },
    "Negativo": {
        "icon": "❌",
        "color": "#B22222",  # rojo
    }
}

# Acción de análisis cuando el archivo ya se ha subido
if uploaded_file is not None:
    content = read_txt_file(uploaded_file)

    if st.button("Analizar conversación"):
        st.markdown("<div class='sub-title'>Contenido del archivo cargado</div>", unsafe_allow_html=True)
        st.text_area("", content, height=300)

        # Procesar el análisis
        analysis = analyze_conversation(content)

        # Mostrar resultados con estilo
        st.markdown("<div class='big-title'>Resultado del Análisis</div>", unsafe_allow_html=True)
        overall_sentiment = analysis.overall_sentiment
        
        # Encabezado de Sentimiento Global
        st.markdown(
            f"<h3 style='color:{sentiment_styles[overall_sentiment]['color']};'>"
            f"Sentimiento Global (Speaker 1): {sentiment_styles[overall_sentiment]['icon']} {overall_sentiment}"
            "</h3>", 
            unsafe_allow_html=True
        )

        # Información de la conversación en una "caja"
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.write(f"**Conversation ID**: {analysis.conversation_id}")
        st.write(f"**Programa Académico**: {analysis.academic_program}")
        st.write(f"**Notas**: {analysis.notes}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Expanders para ver detalles de cada segmento
        with st.expander("Ver Segmentos de la Conversación"):
            for idx, segment in enumerate(analysis.segments, start=1):
                seg_sent = segment.sentiment
                icon = sentiment_styles[seg_sent]["icon"]
                color = sentiment_styles[seg_sent]["color"]
                
                st.markdown(f"<div class='sub-title'>Turno #{idx}</div>", unsafe_allow_html=True)
                st.write(f"**Speaker**: {segment.speaker_id}")
                # Mostramos el texto en un área de texto con el color correspondiente al sentimiento
                st.markdown(
                    f"<p style='border-left: 4px solid {color}; padding-left:8px;'>"
                    f"{segment.text}</p>", 
                    unsafe_allow_html=True
                )
                # Etiqueta de sentimiento
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold;'>"
                    f"{icon} Sentimiento: {seg_sent}</span>", 
                    unsafe_allow_html=True
                )
                # Razones
                if segment.reasons:
                    st.write("**Razones**:")
                    for reason in segment.reasons:
                        st.write(f"- {reason}")
                st.markdown("---")

        # Sección final para mostrar el JSON parseado (opcional)
        with st.expander("Ver JSON Estructurado"):
            st.json(analysis.dict())

else:
    st.info("Por favor carga un archivo .txt en la barra lateral para iniciar el análisis.")
