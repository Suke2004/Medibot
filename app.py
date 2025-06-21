from flask import Flask, request, jsonify, render_template, session
# from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os, logging, json
from datetime import datetime
from model import load_model  # Importing the LLM model from model.py

# LangChain & AI Imports
from src.helper import download_huggingface_model, process_pdf_documents
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

app = Flask(__name__)
app.secret_key = "your_secret_key"
# CORS(app)

# Constants
UPLOAD_FOLDER = "uploads"
CHAT_LOG_FILE = "chat_logs.json"
ALLOWED_EXTENSIONS = {'pdf'}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()



# Load static embedding model
embeddings = download_huggingface_model()

# This will be dynamically updated as PDFs are uploaded
docsearch = None
retriever = None

# Load the LLM model
llm = load_model()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

# === Initialize DB Chain (this is static) ===
db_docsearch = PineconeVectorStore.from_existing_index(index_name="medibot", embedding=embeddings)
db_retriever = db_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
db_rag_chain = create_retrieval_chain(db_retriever, question_answer_chain)

# === PDF Chain (this is dynamic) ===
pdf_rag_chain = None

# ---------- UTILITIES ----------

# def log_chat(question, answer):
#     entry = {
#         "timestamp": datetime.now().isoformat(),
#         "question": question,
#         "answer": answer
#     }
#     if os.path.exists(CHAT_LOG_FILE):
#         with open(CHAT_LOG_FILE, "r") as f:
#             data = json.load(f)
#     else:
#         data = []
#     data.append(entry)
#     with open(CHAT_LOG_FILE, "w") as f:
#         json.dump(data, f, indent=2)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET"])
def chat():
    global db_rag_chain, pdf_rag_chain

    try:
        user_input = request.args.get("msg")
        mode = request.args.get("mode", "db")  # default mode

        if not user_input:
            return jsonify({"error": "Missing input"}), 400

        if mode == "pdf":
            if not pdf_rag_chain:
                return jsonify({"error": "No PDF uploaded yet."}), 400
            result = pdf_rag_chain.invoke({"input": user_input})
        else:
            result = db_rag_chain.invoke({"input": user_input})

        answer = result.get("answer", "No answer generated.")
        # log_chat(user_input, answer)
        return jsonify({"response": answer})

    except Exception as e:
        logging.exception("Chat error:")
        return jsonify({"error": "Internal server error"}), 500



@app.route("/upload", methods=["POST"])
def upload_pdf():
    global pdf_rag_chain

    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Create PDF Vector Store & Chain
        pdf_docsearch = process_pdf_documents(filepath, embeddings)
        pdf_retriever = pdf_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        pdf_rag_chain = create_retrieval_chain(pdf_retriever, question_answer_chain)

        return jsonify({"message": f"Uploaded & indexed: {filename}"}), 200

    return jsonify({"message": "Invalid file format"}), 400


@app.route("/admin")
def admin_dashboard():
    try:
        with open(CHAT_LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []
    return render_template("admin.html", logs=logs)

# ---------- MAIN ----------

if __name__== '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
