import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from docx import Document
import google.generativeai as genai
from pypdf import PdfReader
import dotenv
dotenv.load_dotenv()
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
        return jsonify({'message': 'Invalid file type for file 1. Please upload a PDF or DOCX file.'}), 400
    if not (file2.filename.endswith('.pdf') or file2.filename.endswith('.docx')):
        return jsonify({'message': 'Invalid file type for file 2. Please upload a PDF or DOCX file.'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    file1.save(file1_path)
    file2.save(file2_path)
    text1 = read_document(file1_path)
    text2 = read_document(file2_path)

    intro_prompt = (
        "As an experienced legal expert, your task is to analyze and compare two legal documents. "
        "Your analysis should focus on identifying the similarities and differences regarding their legal essence.\n"
        "The similarities and differences should be presented with bullet points, prioritized by importance/significance. "
        "Highlight the main parts in the important bullets and address specific pages and paragraphs in the documents "
        "accordingly. Also point out if a document has specific term or section that the other document doesn’t have.\n"
        "Similarities:\n"
        "Bullet points detailing the similarities, arranged by importance.\n"
        "Include page and paragraph references when they are mentioned.\n"
        "(Highlight main part for crucial similarities).\n"
        "Differences:\n"
        "Bullet points detailing the differences, arranged by importance.\n"
        "Include page and paragraph references when they are mentioned.\n"
        "For each difference, state if it makes sense to have it because of the core difference essence of the two "
        "documents or whether it raises suspicion about this term/section in one document being problematic.\n"
        "(Highlight main part for crucial differences).\n"
        "Conclusion:\n"
        "Summarize the analysis.\n"
        "Let’s work this out in a step by step way to be sure we have the right answer."
    )
    long_prompt = intro_prompt + "\n\nDocument 1:\n" + text1 + "\n\nDocument 2:\n" + text2
    response = model.generate_content(long_prompt)
    return jsonify({'message': response.text})

if __name__ == '__main__':
    app.run(debug=True)
