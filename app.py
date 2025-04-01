__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import gradio as gr
import os

os.environ["USER_AGENT"] = "my-app/1.0"  
# Set USER_AGENT to identify requests when using HuggingFaceEndpoint and WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

# import requests
# from langchain.document_loaders import WebBaseLoader #deprecated
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate  # vs  PromptTemplate

# OpenAIEmbeddings requires an OpenAI API key. If you don’t have one, you can skip using it and stick with HuggingFaceBgeEmbeddings
loader = WebBaseLoader("https://en.wikipedia.org/wiki/2025_Myanmar_earthquake")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(chunks[0].page_content)

# rag_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=rag_embeddings,
#     persist_directory="./chroma_langchain_db",
# )
# retriever = vector_store.as_retriever()
# chroma_client = chromadb.PersistentClient(path="/workspaces/codespaces-blank/chroma_db")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # "codellama/CodeLlama-34b-Instruct-hf",
    huggingfacehub_api_token=api_key,
    max_new_tokens=256,
    temperature=0.2,
    task="text-generation",
)

# Initialize embeddings for Chroma
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)  # all-MiniLM-L6-v2

# Initialize Chroma vector store (in-memory for now)
vectorstore = Chroma.from_documents(
    documents=chunks,
    collection_name="chat_history",
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
)
# vectorstore.add_documents(chunks)
# Define a prompt template
template = """You are a friendly assistant chatbot. Use the following context to answer the question concisely.

Context: {context}

Question: {question}

Answer: """
prompt = PromptTemplate.from_template(
    template
)  # prompt =ChatPromptTemplate.from_template(template)


# Define the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": vectorstore.as_retriever() | format_docs,  #:retriever
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def respond(current_msg, history_msg, max_tokens, temperature):
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

        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        logger.info("Execution completed.")


# Gradio Interface
chatbot = gr.ChatInterface(
    fn=respond,
    type="messages",
    additional_inputs=[
        gr.Slider(
            minimum=1, maximum=2048, value=256, step=1, label="Max output tokens"
        ),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Creativeness"),
    ],
    description="Chat with the bot using RAG powered by LangChain and Hugging Face.",
)

chatbot.launch(share=True)
