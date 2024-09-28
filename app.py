import os
from flask import Flask, render_template, request, redirect, url_for
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from chromadb import Client
from chromadb.config import Settings
from PyPDF2 import PdfReader

app = Flask(__name__)

# Configuración de la API Key de OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("La clave API de OpenAI no está definida")

# Configuración de variables de entorno adicionales
modelo = os.getenv('MODELO')
if modelo is None:
    raise ValueError("La variable de entorno 'MODELO' no está definida")

input_file = os.getenv('INPUT_FILE')
if input_file is None:
    raise ValueError("La variable de entorno 'INPUT_FILE' no está definida")

FUENTES = os.getenv('FUENTES')
if FUENTES is None:
    raise ValueError("La variable de entorno 'FUENTES' no está definida")

DATA = os.getenv('DATA')
if DATA is None:
    raise ValueError("La variable de entorno 'DATA' no está definida")

# Inicialización de Chroma y Embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
settings = Settings(persist_directory=DATA)
chroma_client = Client(settings)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files_txt = os.path.join(FUENTES, input_file)
        if not os.path.exists(files_txt):
            return "El archivo especificado en INPUT_FILE no se encuentra en la ruta especificada."

        with open(files_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Omitir la cabecera si existe
        if 'upload_file' in lines[0]:
            lines = lines[1:]

        for line in lines:
            if line.strip() == '':
                continue
            upload_file, tipo_doc, nombre_res, fecha, descripcion = line.strip().split(',')
            file_path = os.path.join(FUENTES, upload_file)

            if not os.path.exists(file_path):
                return f"El archivo {upload_file} no se encuentra en {FUENTES}."

            # Extracción de texto
            if upload_file.endswith('.pdf'):
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            else:
                with open(file_path, 'r', encoding='utf-8') as file_content:
                    text = file_content.read()

            # División en chunks
            chunk_size = 500
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            # Agregar chunks y metadata a Chroma
            for chunk in chunks:
                metadata = {
                    'tipo_documento': tipo_doc,
                    'nombre_resolucion': nombre_res,
                    'fecha': fecha,
                    'descripcion': descripcion
                }
                embedding = embeddings.embed_documents([chunk])[0]
                chroma_client.add(documents=[chunk], metadatas=[metadata], embeddings=[embedding])

        return redirect(url_for('home'))
    return render_template('upload.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    respuesta = ''
    if request.method == 'POST':
        pregunta = request.form['pregunta']

        # Recuperar documentos similares
        embedding_pregunta = embeddings.embed_query(pregunta)
        resultados = chroma_client.query(embedding_pregunta, n_results=3, include=['documents', 'metadatas'])

        # Construir contexto
        contexto = ""
        for doc, meta in zip(resultados['documents'], resultados['metadatas']):
            # Incluir el fragmento y su metadata en el contexto
            contexto += f"Fragmento: {doc}\n"
            contexto += f"Metadata:\n"
            contexto += f"  Tipo de Documento: {meta['tipo_documento']}\n"
            contexto += f"  Nombre de la Resolución: {meta['nombre_resolucion']}\n"
            contexto += f"  Fecha: {meta['fecha']}\n"
            contexto += f"  Descripción: {meta['descripcion']}\n"
            contexto += "\n"

        # Generar respuesta con LangChain
        llm = OpenAI(api_key=openai_api_key, model_name=modelo)  # Especificar el modelo desde la variable de entorno
        vectorstore = Chroma(client=chroma_client, embedding_function=embeddings.embed_query)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        respuesta = qa_chain.run({"question": pregunta, "context": contexto})

        return render_template('chatbot.html', pregunta=pregunta, respuesta=respuesta)
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
