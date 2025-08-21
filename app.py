import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, render_template, request, Response
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Any, Iterator
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
    model_name: str = Field(default="gemini-2.0-flash")
    model: Any = Field(default=None)

    def __init__(self, **kwargs: Any):
        super().__init__(streaming=True, **kwargs)
        self.model = genai.GenerativeModel(self.model_name)

    def _generate(self, messages: list[BaseMessage], stop=None) -> ChatResult:
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"

        response = self.model.generate_content(prompt)
        ai_message = AIMessage(content=response.text)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _stream(self, messages: list[BaseMessage], stop=None) -> Iterator[ChatGeneration]:
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"

        responses = self.model.generate_content(prompt, stream=True)
        for response in responses:
            yield ChatGeneration(message=AIMessage(content=response.text))

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
    Question: {input}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "input"])

    llm = GeminiWrapper()

    retriever = vector_store.as_retriever()

    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return qa_chain


qa_chain = setup_rag_pipeline()

# --- Flask Routes ---


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    from flask import jsonify
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"User message: {user_message}")

    def generate():
        try:
            for chunk in qa_chain.stream({"input": user_message}):
                if 'answer' in chunk:
                    # Ensure we are yielding the text content
                    if hasattr(chunk['answer'], 'content'):
                        yield chunk['answer'].content
                    else:
                        # Fallback if it's already a string
                        yield chunk['answer']
        except Exception as e:
            print(f"An error occurred during streaming: {e}")

    return Response(generate(), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
