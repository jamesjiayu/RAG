"""
Chatbot application using RAG (Retrieval-Augmented Generation) with LangChain,
HuggingFace, FAISS, and Gradio for an interactive interface.
Loads data from a web source, processes it into chunks, and stores embeddings
for contextual question answering.
"""



# __import__("pysqlite3")
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# import sqlite3
# print(f"version: {sqlite3.sqlite_version}")
#  >=3.35.0 if Chroma
# from langchain_chroma import Chroma

import os
import sys
import logging
from typing import List, Optional, Dict, Any
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

#from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAI, OpenAIEmbeddings  # For optional OpenAI fallback
import sqlite3
print(f"version: {sqlite3.sqlite_version}")
#  >=3.35.0 # Replace standard sqlite3 with pysqlite3 for Chroma compatibility



logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WEB_URL = "https://en.wikipedia.org/wiki/2025_Myanmar_earthquake"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# "facebook/rag-sequence-nq"   Or "google/flan-t5-large"
# PERSIST_DIR = "./chroma_langchain_db"
PERSIST_DIR = "./faiss_index"
COLLECTION_NAME = "chat_history"
USER_AGENT = "my-app/1.0"


# Set USER_AGENT to identify requests when using HuggingFaceEndpoint and WebBaseLoader
#MAX_MESSAGES_BEFORE_PRUNE = 100 #for faiss

def setup_environment() -> str:
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        logger.error("HUGGINGFACE_API_KEY not found in environment variables")
        raise ValueError("Missing HUGGINGFACE_API_KEY")
    os.environ["USER_AGENT"] = USER_AGENT
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


def create_rag_chain(vectorstore: FAISS, llm: HuggingFaceEndpoint):
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

def respond(
    current_msg: str,
    max_tokens: int,
    temperature: float,
    vectorstore: FAISS,
    llm: HuggingFaceEndpoint,
    rag_chain,
    #history_msg: Optional[List[dict]] = None,
) -> str:
    logger.info(f"Processing message: {current_msg[:50]}...")
    llm.max_new_tokens = max_tokens
    llm.temperature = temperature
    vectorstore = load_vector_store()
    # Store user message in Chroma. FAISS.save_local() 将其持久化到磁盘后，必须使用 FAISS.load_local() 明确地将其重新加载到内存中以执行像 add_texts 或检索这样的操作
    vectorstore.add_texts(texts=[current_msg], metadatas=[{"role": "user"}])

    try:
        # Generate response using RAG chain
        response = rag_chain.invoke(current_msg)

        # Store assistant response in vectorstore
        vectorstore.add_texts(texts=[response], metadatas=[{"role": "assistant"}])
        # vectorstore.save_local(PERSIST_DIR) #Optionally save periodically
        logger.info("Response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response due to an error."
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
            fn=lambda msg, tokens, temp: respond(
                msg, tokens, temp, vectorstore, llm, rag_chain
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
        chatbot.launch()
        vectorstore.save_local(PERSIST_DIR)
    except Exception as e:
        logger.error(f"Failed to launch chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




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
