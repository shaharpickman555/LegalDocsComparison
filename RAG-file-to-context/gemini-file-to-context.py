from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import dotenv
import os
from docx import Document
from pypdf import PdfReader
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
dotenv.load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_document(file_path):
    if file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'message': 'No file part'})
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'message': 'No selected file'})

    if not (file1.filename.endswith('.pdf') or file1.filename.endswith('.docx')):
        return jsonify({'message': 'Invalid file type for file 1. Please upload a PDF or DOCX file for your legal document.'}), 400
    if not (file2.filename.endswith('.pdf')):
        return jsonify({'message': 'Invalid file type for file 2. Please upload a PDF file for the reference document.'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    file1.save(file1_path)
    file2.save(file2_path)
    curr_doc = read_document(file1_path)
    loader = PyPDFLoader(file2_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    intro_prompt = (
        "As an experienced legal expert, your task is to analyze the following legal document to "
        "determine if the information presented makes sense in a legal context, in comparison to your context. "
        "Please check for any missing important sections or any terms/sections that raise suspicion and may be "
        "problematic legally.\n"
        "Highlight the main parts in important bullet points and address specific pages and paragraphs in the "
        "document accordingly.\n"
        "Let’s work this out in a step-by-step way to be sure we have the right answer.\n"
    )
    long_prompt = intro_prompt + "\n\nDocument:\n" + curr_doc
    answer = ""
    for chunk in rag_chain.stream(long_prompt):
        answer+=chunk
    return jsonify({'message': answer})

if __name__ == '__main__':
    app.run(debug=True)