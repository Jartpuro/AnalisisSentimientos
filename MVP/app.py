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

load_dotenv()



class ConversationSegment(BaseModel):
    """
    Representa cada turno de la conversación junto con su sentimiento y
    las posibles razones para esa clasificación.
    """
    speaker_id: str  # Ejemplo: "Speaker 0", "Speaker 1"
    text: str        # Texto del turno en la conversación
    sentiment: str   # Puede ser "Positivo", "Neutral" o "Negativo"
    reasons: List[str]  # Lista de razones (puede incluir indicios textuales o contexto)

class ConversationAnalysis(BaseModel):
    """
    Modelo principal que encapsula la información de la conversación.
    """
    conversation_id:    str           # Identificador único de la conversación
    overall_sentiment:  str         # Sentimiento global de la conversación
    segments:           List[ConversationSegment]  # Lista de turnos clasificados
    academic_program:   str         # Sentimiento global de la conversación
    notes:              str                     # Campo opcional para comentarios adicionales o contexto

def analyze_conversation(contenido: str) -> ConversationAnalysis:
    """
    Analiza la conversación dada (Speaker 0 y Speaker 1) y determina el sentimiento global
    de Speaker 1, así como la clasificación de cada turno (Positivo, Neutral o Negativo).
    También extrae información como el programa académico ofrecido (academic_program) y
    comentarios adicionales (notes), según el formato Pydantic definido.

    Parameters:
    -----------
    contenido : str
        Texto de la conversación transcrita, incluyendo los turnos de Speaker 0 y Speaker 1.

    Returns:
    --------
    ConversationAnalysis
        Objeto que contiene:
        - conversation_id (str): Identificador de la conversación.
        - overall_sentiment (str): 'Positivo', 'Neutral' o 'Negativo' (reflecta la actitud de Speaker 1).
        - segments (List[ConversationSegment]): Lista con los turnos clasificados.
        - academic_program (str): Nombre del programa académico ofrecido a Speaker 1.
        - notes (str): Comentarios adicionales o información relevante.
    """

    # Asegúrate de que tu cliente de OpenAI esté configurado correctamente con la API key.
    client = OpenAI()  # O tu propia inicialización de cliente, p. ej. openai.api_key = "..."

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
                    "de Speaker 0 y Speaker 1."
                    "\n\n"
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
        response_format=ConversationAnalysis  # Clase Pydantic que modela la respuesta
    )

    # Obtén la respuesta parseada en un objeto Pydantic
    analysis = completion.choices[0].message.parsed

    return analysis

def get_txt_files_by_folder(parent_path):
    """
    Obtiene un diccionario donde cada clave es el nombre de una carpeta en parent_path
    y cada valor es una lista de rutas completas de archivos .txt dentro de esa carpeta.

    :param parent_path: Ruta principal que contiene las subcarpetas.
    :return: Diccionario con rutas de archivos .txt organizadas por carpetas.
    """
    txt_files = {}
    for folder in parent_path.iterdir():
        if folder.is_dir():  # Solo procesar carpetas
            txt_files[folder.name] = []
            for txt_file in folder.glob("*.txt"):  # Buscar archivos .txt en la carpeta
                txt_files[folder.name].append(txt_file.resolve())
    return txt_files

def read_txt_file(file_path):
    """
    Lee un archivo de texto asegurándose de manejar caracteres especiales (UTF-8).
    
    :param file_path: Ruta completa al archivo de texto.
    :return: Contenido del archivo como cadena.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error al leer el archivo: {e}"


current_path        = pl.Path.cwd()
output_folders      = create_output_directories(current_path)
data_folders_path   = create_data_directories(output_folders[0])
imagen_path         = create_img_directories(current_path)
audio_path          = output_folders[0].joinpath('raw', 'audios_txt')
processed_folder    = output_folders[0].joinpath('processed')
output_file         = processed_folder.joinpath('sentiments_output.csv')

st.write(f'Current path: {current_path}')
st.write(f'audio_path: {audio_path}')
st.write(output_file)
# st.write('# Prueba con 1 solo archivo')
# audio negativo
# audio_prueba = audio_path.joinpath('2023-05-13a', '5db4b150a2a746f188fcd719951375cd_20230513t16_31_utc.txt')
# #audio_prueba = audio_path.joinpath('2023-05-13a', '4d1237f231144ff7ab3b0915ce55a261_20230513t15_48_utc.txt')
# # audio positivo
# #audio_prueba = audio_path.joinpath('2023-04-22a', '71c3a935-7b26-437e-921e-0578d6591052_20230422T16_17_UTC.txt')

# # Leer y mostrar el contenido del archivo
# st.write(f"Archivo seleccionado: {audio_prueba}")
# if audio_prueba.exists():
#     contenido = read_txt_file(audio_prueba)
#     st.text_area("Contenido del archivo", contenido, height=300)
# else:
#     st.write("El archivo no existe.")


# Buscar la forma de cambiar el role de system por developer que es un enfoque nuevo
# https://platform.openai.com/docs/guides/text-generation?lang=python




# # Convierte el modelo a diccionario Python:
# json_data = analysis.model_dump()

# st.write('## Resultado del análisis en formato estructurado')
# st.write("=== Conversación Analizada ===")
# st.write(f"Conversation ID: {analysis.conversation_id}")
# st.write(f"Overall Sentiment (Speaker 1): {analysis.overall_sentiment}")
# st.write(f"Academic Program: {analysis.academic_program}")
# st.write(f"Notes: {analysis.notes}")

# st.write("\n=== Segments ===")
# for segment in analysis.segments:
#     st.write(f"Speaker: {segment.speaker_id}")
#     st.write(f"Text: {segment.text}")
#     st.write(f"Sentiment: {segment.sentiment}")
#     st.write(f"Reasons: {segment.reasons}")
#     st.write("---")

# # Si deseas imprimir en formato JSON con indent y mantener tildes/caracteres:
# st.write("\n=== JSON Output ===")
# st.write(json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False))





# Definir las columnas que se usarán en el DataFrame
df_columns = [
    'folder', 
    'filename', 
    'overall_sentiment', 
    'academic_program', 
    'notes', 
    'segments'
]

# Verificar si existe un archivo en output_file
if output_file.exists():
    df = pd.read_csv(output_file, keep_default_na=True)
else:
    df = pd.DataFrame(columns=df_columns)

txt_files_by_folder = get_txt_files_by_folder(audio_path)

for folder, files in txt_files_by_folder.items():
    for file in files:
        # Verificar si ya existe un registro para este folder y filename
        mask = (df['folder'] == folder) & (df['filename'] == file.name)

        if mask.any():
            # Si el registro existe, verificar si 'overall_sentiment' está vacío o es NaN
            if pd.isnull(df.loc[mask, 'overall_sentiment']).all():
                contenido = read_txt_file(file)
                resultado = analyze_conversation(contenido)

                df.loc[mask, 'overall_sentiment'] = resultado.overall_sentiment
                df.loc[mask, 'academic_program']   = resultado.academic_program
                df.loc[mask, 'notes']              = resultado.notes
                df.loc[mask, 'segments']           = resultado.segments

                # Guardar el DataFrame (con la fila actualizada) en cada iteración
                #df.to_csv(output_file, index=False)
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
        else:
            # Si no existe registro para este archivo, se analiza y se agrega
            contenido = read_txt_file(file)
            resultado = analyze_conversation(contenido)

            new_row = {
                'folder': folder,
                'filename': file.name,
                'overall_sentiment': resultado.overall_sentiment,
                'academic_program': resultado.academic_program,
                'notes': resultado.notes,
                'segments': resultado.segments
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Guardar el DataFrame (con la fila agregada) en cada iteración
            # df.to_csv(output_file, index=False)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')

st.write("DataFrame resultante:")
st.write(df)