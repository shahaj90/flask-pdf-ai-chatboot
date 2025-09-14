# Flask PDF AI Chatbot

A simple web-based chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about a PDF document. This project uses Flask for the web framework, Google's Gemini for the language model, and ChromaDB for the vector store.

## Features

*   **Chat with your PDF:** Ask questions in natural language and get answers based on the content of your PDF document.
*   **RAG Pipeline:** Utilizes a RAG pipeline to retrieve relevant information from the PDF and generate accurate answers.
*   **Easy to Use:** Simple web interface for interacting with the chatbot.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/shahaj90/flask-pdf-ai-chatbot.git
    cd flask-pdf-ai-chatbot
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    Create a `.env` file in the root directory and add your Google API key:

    ```
    GEMINI_API_KEY=your-gemini-api-key
    ```

5.  **Add your PDF document:**

    Place your PDF document in the `data` directory and name it `your_document.pdf`.

## Usage

1.  **Run the application:**

    ```bash
    python3 app.py
    ```

2.  **Open your browser:**

    Navigate to `http://127.0.0.1:5000` to access the chatbot.

## Dependencies

*   flask
*   pypdf
*   langchain==0.3.27
*   langchain-community==0.3.27
*   langchain-core==0.3.74
*   langchain-text-splitters==0.3.9
*   chromadb==1.0.20
*   sentence-transformers==5.1.0
*   google-generativeai==0.8.5
*   python-dotenv==1.1.1
*   nest-asyncio==1.6.0
