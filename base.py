import time
import logging
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Función para extraer texto de un PDF usando pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Función para cargar el modelo de lenguaje y el pipeline de QA
def load_qa_pipeline():
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=0 if torch.cuda.is_available() else -1  # Usar GPU si está disponible
    )
    return qa_pipeline

# Función para crear el índice de búsqueda usando FAISS
def create_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(documents, embeddings)
    return vectorstore

# Función para evaluar la calidad de la respuesta
def evaluate_answer(question, answer, context):
    # Cargar modelo de embeddings para similitud semántica
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Calcular similitud entre pregunta y respuesta
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    similarity_score = util.cos_sim(question_embedding, answer_embedding).item()

    # Calcular similitud entre respuesta y contexto (fragmentos relevantes)
    context_embedding = model.encode(context, convert_to_tensor=True)
    consistency_score = util.cos_sim(answer_embedding, context_embedding).item()

    return similarity_score, consistency_score

# Función principal
def main(pdf_path):
    # Extraer texto del PDF
    start_time = time.time()
    text = extract_text_from_pdf(pdf_path)
    logger.info(f"Texto extraído en {time.time() - start_time:.2f} segundos")

    # Crear índice de búsqueda
    start_time = time.time()
    vectorstore = create_index(text)
    logger.info(f"Índice creado en {time.time() - start_time:.2f} segundos")

    # Cargar modelo de QA
    start_time = time.time()
    qa_pipeline = load_qa_pipeline()
    logger.info(f"Modelo de QA cargado en {time.time() - start_time:.2f} segundos")

    # Bucle para recibir preguntas del usuario
    while True:
        question = input("Ingresa tu pregunta (o 'shazam!' para salir): ")
        if question.lower() == "shazam!":
            logger.info("¡Programa finalizado!")
            break

        # Buscar los fragmentos de texto más relevantes para la pregunta
        start_time = time.time()
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Pasar la pregunta y el contexto al modelo de QA
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"]
        latency = time.time() - start_time

        # Evaluar la calidad de la respuesta
        similarity_score, consistency_score = evaluate_answer(question, answer, context)

        # Logs de seguimiento
        logger.info(f"Pregunta: {question}")
        logger.info(f"Respuesta generada: {answer}")
        logger.info(f"Similitud pregunta-respuesta: {similarity_score:.2f}")
        logger.info(f"Consistencia con el documento: {consistency_score:.2f}")
        logger.info(f"Tiempo de respuesta: {latency:.2f} segundos")
        logger.info("-" * 50)

if __name__ == "__main__":
    pdf_path = "documento.pdf"  # Ruta al archivo PDF
    main(pdf_path)
