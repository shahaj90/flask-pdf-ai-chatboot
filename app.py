import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# LangChain and RAG-specific imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Imports for LangChain compatibility
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Any
from pydantic import Field

app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# --- Configure Gemini API ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Please set it as an environment variable.")
    exit()
genai.configure(api_key=GEMINI_API_KEY)

# --- RAG Pipeline Setup ---


class GeminiWrapper(BaseChatModel):
    # Use pydantic Field to declare the attributes
    model_name: str = Field(default="gemini-1.5-flash-latest")
    model: Any = Field(default=None)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = genai.GenerativeModel(self.model_name)

    def _generate(self, messages: list[BaseMessage], stop=None) -> ChatResult:
        # Convert LangChain messages to a string format for Gemini's API
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"

        response = self.model.generate_content(prompt)
        ai_message = AIMessage(content=response.text)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @property
    def _llm_type(self) -> str:
        return "gemini-wrapper"


def setup_rag_pipeline():
    """
    Sets up the RAG pipeline.
    """
    pdf_path = "./data/your_document.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        exit()

    if not os.path.exists("./chroma_db"):
        print("Vector store not found. Creating it from the PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        print("Generating embeddings. This may take a few minutes...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(
            chunks, embedding_model, persist_directory="./chroma_db")
        print("Vector store created successfully.")
    else:
        print("Vector store found. Loading it...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory="./chroma_db",
                              embedding_function=embedding_model)
        print("Vector store loaded.")

    prompt_template = """
    You are an AI assistant that answers questions based on the provided context only.
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    llm = GeminiWrapper()

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


qa_chain = setup_rag_pipeline()

# --- Flask Routes ---


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"User message: {user_message}")

    try:
        response = qa_chain.invoke({"query": user_message})
        bot_response = response.get(
            'result', 'Sorry, I could not generate a response.')
        print(f"Bot response: {bot_response}")
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500


if __name__ == '__main__':
    app.run(debug=True)
