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
import logging
from typing import List, Optional
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# from langchain_openai import OpenAIEmbeddings

import sqlite3

print(f"version: {sqlite3.sqlite_version}")
#  >=3.35.0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Replace standard sqlite3 with pysqlite3 for Chroma compatibility


WEB_URL = "https://en.wikipedia.org/wiki/2025_Myanmar_earthquake"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PERSIST_DIR = "./chroma_langchain_db"
# PERSIST_DIR = "./faiss_index"
COLLECTION_NAME = "chat_history"
USER_AGENT = "my-app/1.0"


def setup_environment() -> str:
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        logger.error("HUGGINGFACE_API_KEY not found in environment variables")
        raise ValueError("Missing HUGGINGFACE_API_KEY")
    os.environ["USER_AGENT"] = USER_AGENT
    # Set USER_AGENT to identify requests when using HuggingFaceEndpoint and WebBaseLoader
    return api_key


"""
A Document object: 2 main attributes:
page_content: This is a string attribute that holds the actual text content of the document
metadata: This is a dictionary attribute that stores extra information related to the document. This can include:
The source URL (source)
The title of the document (title)
Creation date (date)
Any other information you want to associate with the document.

loader.load() method typically returns a list of Document objects (List[Document]).
text splitter that breaks down longer documents into smaller chunks(List[Document])
"""


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


"""
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
"""


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
        logger.error(f"Failed to create vector stroe: {e}")
        raise


# pip install faiss-cpu
# def create_vector_store1(chunks: List[Document]) -> FAISS:
#     logger.info("Initializing embeddings and vector store")
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         vectorstore.save_local(PERSIST_DIR)
#         logger.info(f"Vector store created at {PERSIST_DIR}")
#         return vectorstore
#     except Exception as e:
#         logger.error(f"Failed to create vector stroe: {e}")
#         raise

# def load_vector_store() -> FAISS:
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
#     logger.info(f"load from {PERSIST_DIR} ")
#     return vectorstore


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

    """
    (...) LangChain Runnable (object) sequence
    rag_chain itself is a RunnableSequence object.
    Pipe Operator |,  to chain together different Runnables
    output of the Runnable on the left becomes the input of the Runnable on the right
    a sequence of two Runnables connected by the | operator
    output of the vectorstore.as_retriever() Runnable will be passed as input to the format_docs Runnable. (returns a BaseRetriever object)
    a RunnablePassthrough instance  
    takes the output of the LLM and parses it into a plain string
    StrOutputParser() (an instance of an output parser)
    
    def format_docs1(docs: List[Document])-> str:
    formatted_content = ""
    for i,doc in enumerate(docs):
        formatted_content += doc.page_content
        # Add separator if it's not the last document
        if i<len(docs)-1:
            formatted_content+="\n\n"
    return formatted_content    
    """


# # os.environ["USER_AGENT"] = "my-app/1.0"
# load_dotenv()
# api_key = os.getenv("HUGGINGFACE_API_KEY")


# def prepare_data():
#     # OpenAIEmbeddings requires an OpenAI API key. If you don’t have one, you can skip using it and stick with HuggingFaceBgeEmbeddings
#     loader = WebBaseLoader("https://en.wikipedia.org/wiki/2025_Myanmar_earthquake")
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_documents(documents)
#     print(chunks[0].page_content)
#     return chunks


# # rag_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# # vector_store = Chroma.from_documents(
# #     documents=chunks,
# #     embedding=rag_embeddings,
# #     persist_directory="./chroma_langchain_db",
# # )
# # retriever = vector_store.as_retriever()
# # chroma_client = chromadb.PersistentClient(path="/workspaces/codespaces-blank/chroma_db")


# def embedding_data(chunks):
#     # Initialize embeddings for Chroma # all-MiniLM-L6-v2
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
#     # Initialize Chroma vector store (in-memory for now)
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         collection_name="chat_history",
#         embedding=embeddings,
#         persist_directory="./chroma_langchain_db",
#     )
#     # vectorstore.add_documents(chunks) #persist_directory=os.path.abspath("./chroma_langchain_db"),  # 使用绝对路径
#     return vectorstore


# chunks = prepare_data()

# vectorstore = embedding_data(chunks)

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     huggingfacehub_api_token=api_key,
#     max_new_tokens=256,
#     temperature=0.2,
#     task="text-generation",
# )

# template = """You are a friendly assistant chatbot. Use the following context to answer the question concisely.
# Context: {context}
# Question: {question}
# Answer: """

# prompt = PromptTemplate.from_template(template)
# # prompt OBJECT # =ChatPromptTemplate.from_template(template)


# # Define the RAG chain
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {
#         "context": vectorstore.as_retriever() | format_docs,  #:retriever
#         "question": RunnablePassthrough(),
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )


def respond(
    current_msg: str,
    history_msg: List,
    max_tokens: int,
    temperature: float,
    vectorstore: Chroma,
    llm: HuggingFaceEndpoint,
    rag_chain,
) -> Optional[str]:
    logger.info(f"Processing message: {current_msg[:50]}...")
    # Update LLM parameters
    llm.max_new_tokens = max_tokens
    llm.temperature = temperature

    # Store user message in Chroma
    vectorstore.add_texts(texts=[current_msg], metadatas=[{"role": "user"}])

    try:
        # Generate response using RAG chain
        response = rag_chain.invoke(current_msg)

        # Store assistant response in Chroma
        vectorstore.add_texts(texts=[response], metadatas=[{"role": "assistant"}])
        logger.info("Response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None
    finally:
        logger.info("Execution completed")


def main():
    try:
        api_key = setup_environment()
        chunks = load_and_split_documents(WEB_URL)
        vectorstore = create_vector_store(chunks)
        llm = initialize_llm(api_key)
        rag_chain = create_rag_chain(vectorstore, llm)
        chatbot = gr.ChatInterface(
            fn=lambda msg, history, tokens, temp: respond(
                msg, history, tokens, temp, vectorstore, llm, rag_chain
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
        )
        chatbot.launch(share=True)
    except Exception as e:
        logger.error(f"Failed to launch chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
