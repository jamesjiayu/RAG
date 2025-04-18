"""
Chatbot application using RAG (Retrieval-Augmented Generation) with LangChain,
HuggingFace, Chroma, and Gradio for an interactive interface.
Loads data from a web source, processes it into chunks, and stores embeddings
for contextual question answering.
"""

import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os
import json
import logging
from typing import List, Optional
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import sqlite3

# Verify SQLite version
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger.info(f"SQLite version: {sqlite3.sqlite_version}")
if sqlite3.sqlite_version_info < (3, 35, 0):
    logger.warning("SQLite version is below 3.35.0, which may cause issues with Chroma.")

# Constants
WEB_URL = "https://en.wikipedia.org/wiki/2025_Myanmar_earthquake"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "chat_history"
USER_AGENT = "my-app/1.0"
HISTORY_FILE = "chat_history.json"

# Global chat history (in-memory)
chat_history: List[dict] = []

def setup_environment() -> str:
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        logger.error("HUGGINGFACE_API_KEY not found in environment variables")
        raise ValueError("Missing HUGGINGFACE_API_KEY")
    os.environ["USER_AGENT"] = USER_AGENT
    return api_key

def load_and_split_documents(url: str) -> List[Document]:
    logger.info(f"Loading documents from {url}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} document chunks")
        if chunks:
            logger.debug(f"First chunk content: {chunks[0].page_content[:100]}...")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load or split documents: {e}")
        raise
#构造函数，用于加载一个已存在的向量存储或者创建一个空的向量存储
def create_vector_store(chunks: List[Document]) -> Chroma:
    logger.info("Initializing embeddings and vector store")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        logger.info(f"Vector store created at {PERSIST_DIR}")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise

#（class method），用于基于给定的文档创建一个新的向量存储，并自动将文档嵌入（embed）到向量存储中
def load_vector_store() -> Chroma:
    logger.info(f"Loading Chroma vector store from {PERSIST_DIR}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        logger.info(f"Chroma vector store loaded from {PERSIST_DIR}")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise

def initialize_llm(api_key: str) -> HuggingFaceEndpoint:
    logger.info(f"Initializing LLM: {LLM_MODEL}")
    try:
        llm = HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=api_key,
            max_new_tokens=256,
            temperature=0.2,
            task="text-generation",
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def create_rag_chain(vectorstore: Chroma, llm: HuggingFaceEndpoint):
    logger.info("Setting up RAG chain")
    try:
        template = """You are a friendly assistant chatbot. Use the following
            context to answer the question concisely.
            Context: {context}
            Question: {question}
            Answer: """
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": vectorstore.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        raise

def save_chat_history():
    """Save chat history to a JSON file."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(chat_history, f, indent=2)
        logger.info(f"Chat history saved to {HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")

def load_chat_history() -> List[dict]:
    """Load chat history from a JSON file."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            logger.info(f"Chat history loaded from {HISTORY_FILE}")
            return history
        return []
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return []

def respond(
    current_msg: str,
    max_tokens: int,
    temperature: float,
    vectorstore: Chroma,
    llm: HuggingFaceEndpoint,
    rag_chain,
    history_msg: List[dict],
) -> str:
    logger.info(f"Processing message: {current_msg[:50]}...")
    vectorstore.add_texts(texts=[current_msg], metadatas=[{"role": "user"}])
    history_msg.append({"role": "user", "content": current_msg})

    try:
        llm_temp = HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=llm.huggingfacehub_api_token,
            max_new_tokens=max_tokens,
            temperature=temperature,
            task="text-generation",
        )
        temp_rag_chain = (
            {
                "context": vectorstore.as_retriever() | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                "question": RunnablePassthrough(),
            }
            | PromptTemplate.from_template(
                """You are a friendly assistant chatbot. Use the following
                context to answer the question concisely.
                Context: {context}
                Question: {question}
                Answer: """
            )
            | llm_temp
            | StrOutputParser()
        )
        response = temp_rag_chain.invoke(current_msg)
        vectorstore.add_texts(texts=[response], metadatas=[{"role": "assistant"}])
        history_msg.append({"role": "assistant", "content": response})
        logger.info("Response generated successfully")
        save_chat_history(history_msg)  
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response due to an error."
    finally:
        logger.info("Execution completed")


# lambda 函数的返回值是调用 respond() 函数的结果。
# lambda 函数将 msg, tokens, temp, history 作为参数传递给 respond()，同时还传递了额外的参数 vectorstore, llm, 和 rag_chain（这些是在 main() 中定义的变量）
# def chat_handler(msg, history, tokens, temp):
#     return respond(msg, tokens, temp, vectorstore, llm, rag_chain, history)
def main():
    try:
        api_key = setup_environment()
        chunks = load_and_split_documents(WEB_URL)
        if os.path.exists(PERSIST_DIR):
            logger.info(f"Loading existing Chroma index from {PERSIST_DIR}")
            vectorstore = load_vector_store()
        else:
            vectorstore = create_vector_store(chunks)
        llm = initialize_llm(api_key)
        rag_chain = create_rag_chain(vectorstore, llm)

        # Load initial history
        initial_history = load_chat_history()

        chatbot = gr.ChatInterface(
            fn=lambda msg, history, tokens, temp: respond(
                msg, tokens, temp, vectorstore, llm, rag_chain, history
            ),
            type="messages",
            additional_inputs=[
                gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=256,
                    step=1,
                    label="Max output tokens",
                ),
                gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Creativeness"
                ),
            ],
            description="Chat with the bot using RAG powered by LangChain and Hugging Face.",
            examples=initial_history if initial_history else None,  # Display loaded history
        )
        chatbot.launch()
    except Exception as e:
        logger.error(f"Failed to launch chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
