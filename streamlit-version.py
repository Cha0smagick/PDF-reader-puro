import os
import time
import logging
import pdfplumber
from langdetect import detect
from googletrans import Translator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
import asyncio
import streamlit as st

# Desactivar el warning de symlinks en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar el traductor
translator = Translator()

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF y maneja excepciones."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text.replace('\n', ' ').strip()  # Eliminar saltos de línea
    except Exception as e:
        logger.error(f"Error al extraer texto del PDF: {e}")
        return ""

def detect_language(text):
    """Detecta el idioma del texto."""
    try:
        return detect(text)
    except Exception as e:
        logger.error(f"Error al detectar el idioma: {e}")
        return "en"  # Default to English if detection fails

async def translate_text(text, src_lang, dest_lang):
    """Traduce texto de un idioma a otro."""
    try:
        translated = await translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        logger.error(f"Error al traducir el texto: {e}")
        return text

def load_qa_pipeline():
    """Carga el modelo de lenguaje y el pipeline de QA."""
    # Cargar el modelo y el tokenizador
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M", device_map="auto", torch_dtype=torch.float32)

    # Configurar el pipeline de generación de texto
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4000,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

def create_index(text):
    """Crea un índice de búsqueda usando FAISS con fragmentos optimizados."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Ajuste de tamaño
    documents = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(documents, embeddings)

def evaluate_answer(question, answer, context):
    """Evalúa la calidad de la respuesta generada con métricas adicionales."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    similarity_score = util.cos_sim(question_embedding, answer_embedding).item()
    context_embedding = model.encode(context, convert_to_tensor=True)
    consistency_score = util.cos_sim(answer_embedding, context_embedding).item()
    
    # Nueva métrica: longitud de la respuesta
    answer_length = len(answer.split())
    
    return similarity_score, consistency_score, answer_length

async def main(pdf_path):
    """Función principal que orquesta la extracción de texto y la respuesta a preguntas."""
    start_time = time.time()
    text = extract_text_from_pdf(pdf_path)
    logger.info(f"Texto extraído en {time.time() - start_time:.2f} segundos")

    # Detectar el idioma del texto
    text_language = detect_language(text)
    logger.info(f"Idioma detectado: {text_language}")

    start_time = time.time()
    vectorstore = create_index(text)
    logger.info(f"Índice creado en {time.time() - start_time:.2f} segundos")

    start_time = time.time()
    qa_pipeline = load_qa_pipeline()
    logger.info(f"Modelo de QA cargado en {time.time() - start_time:.2f} segundos")

    # Preguntar al usuario en qué idioma desea la respuesta (solo una vez)
    response_language = st.selectbox("¿Deseas la respuesta en español?", ["Sí", "No"])
    response_in_spanish = response_language == "Sí"

    while True:
        question = st.text_input("Ingresa tu pregunta (o 'shazam!' para salir): ")
        if question.lower() == "shazam!":
            logger.info("¡Programa finalizado!")
            break

        # Traducir la pregunta al idioma del texto si es necesario
        if detect_language(question) != text_language:
            question_translated = await translate_text(question, 'es', text_language)
        else:
            question_translated = question

        start_time = time.time()
        relevant_docs = vectorstore.similarity_search(question_translated, k=3)
        context = " ".join(doc.page_content for doc in relevant_docs)

        qa = RetrievalQA.from_chain_type(llm=qa_pipeline, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True)
        result = qa({"query": question_translated})
        answer = result["result"]
        latency = time.time() - start_time

        # Traducir la respuesta al español si es necesario
        if response_in_spanish:
            answer = await translate_text(answer, text_language, 'es')

        similarity_score, consistency_score, answer_length = evaluate_answer(question_translated, answer, context)

        st.write(f"Pregunta: {question}")
        st.write(f"Respuesta generada: {answer} (Longitud: {answer_length} palabras)")
        st.write(f"Similitud pregunta-respuesta: {similarity_score:.2f}")
        st.write(f"Consistencia con el documento: {consistency_score:.2f}")
        st.write(f"Tiempo de respuesta: {latency:.2f} segundos")
        st.write("-" * 50)

if __name__ == "__main__":
    st.title("Sistema de Preguntas y Respuestas sobre Documentos PDF")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        asyncio.run(main("temp.pdf"))
