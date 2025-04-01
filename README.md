# RAG Chatbot with LangChain and Hugging Face

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain and Hugging Face models. It fetches data from a Wikipedia article, creates a vector store using ChromaDB, and generates responses with a CodeLlama model.

## Features

* **RAG Implementation:** Uses LangChain to create a RAG pipeline.
* **Wikipedia Data:** Fetches data from a specified Wikipedia article.
* **Chroma Vector Store:** Stores document embeddings for efficient retrieval.
* **Hugging Face Models:** Uses Hugging Face embeddings and a CodeLlama language model.
* **Gradio Interface:** Provides a user-friendly chat interface.
* **Customizable Parameters:** Allows adjusting max output tokens and creativeness.

## Prerequisites

* Python 3.1x
* Pip
* A Hugging Face API key (set as the `HUGGINGFACE_API_KEY` environment variable)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

    (Replace `<repository_url>` and `<repository_directory>` with your repository's URL and directory.)

2.  **Install dependencies:**

    ```bash
    pip install gradio langchain langchain-huggingface langchain-chroma python-dotenv requests sentence-transformers
    ```

3.  **Set the Hugging Face API key:**

    * Create a `.env` file in the project's root directory.
    * Add your Hugging Face API key to the `.env` file:

        ```
        HUGGINGFACE_API_KEY=your_huggingface_api_key
        ```

        (Replace `your_huggingface_api_key` with your actual API key.)

4.  **Run the application:**

    ```bash
    python app.py
    ```

## Usage

1.  Run the `app.py` script.
2.  The Gradio interface will launch in your web browser.
3.  Enter your question in the chat input.
4.  The chatbot will generate a response based on the fetched Wikipedia data.
5.  Use the sliders to adjust the `Max output tokens` and `Creativeness` of the responses

## Code Explanation

* **Data Loading:** The script loads data from a Wikipedia article using `WebBaseLoader`.
* **Text Splitting:** The loaded data is split into chunks using `RecursiveCharacterTextSplitter`.
* **Vector Store:** ChromaDB is used to create a vector store from the document chunks, using `HuggingFaceEmbeddings`.
* **Language Model:** A CodeLlama model from Hugging Face is used to generate responses.
* **RAG Chain:** LangChain's `RunnablePassthrough`, `PromptTemplate`, and `StrOutputParser` are used to create the RAG pipeline.
* **Gradio Interface:** Gradio's `ChatInterface` is used to create the user interface.

## Notes

* Remember to set your Hugging Face API key as an environment variable or in a `.env` file.
* The Chroma database is persisted to the `./chroma_langchain_db` directory.
* You can change the Wikipedia URL to use a different source.
* The `all-MiniLM-L6-v2` and `BAAI/bge-small-zh-v1.5` models are used for embeddings.
* The `codellama/CodeLlama-34b-Instruct-hf` model is used for text generation.

## .gitignore

Make sure to add the following to your `.gitignore` file to prevent unnecessary files from being committed:
