# LaMini PDF reader

Este script en Python está diseñado para extraer texto de un archivo PDF, detectar su idioma, y permitir al usuario hacer preguntas sobre el contenido del documento. El script utiliza modelos de lenguaje avanzados para generar respuestas precisas y evaluar la calidad de las respuestas en función de su similitud con la pregunta y su consistencia con el contenido del documento.

## Funcionalidades Principales

1. **Extracción de Texto**: Extrae texto de un archivo PDF utilizando la biblioteca `pdfplumber`.
2. **Detección de Idioma**: Detecta el idioma del texto extraído utilizando `langdetect`.
3. **Traducción de Texto**: Traduce texto entre idiomas utilizando `googletrans`.
4. **Generación de Respuestas**: Utiliza un modelo de lenguaje (`LaMini-T5-738M`) para generar respuestas a las preguntas del usuario.
5. **Búsqueda Semántica**: Crea un índice de búsqueda utilizando `FAISS` y `HuggingFaceEmbeddings` para encontrar fragmentos relevantes del texto.
6. **Evaluación de Respuestas**: Evalúa la calidad de las respuestas generadas utilizando métricas de similitud y consistencia.

## Requisitos

- Python 3.8 o superior
- Bibliotecas de Python: `pdfplumber`, `langdetect`, `googletrans`, `langchain`, `transformers`, `sentence-transformers`, `torch`, `asyncio`
