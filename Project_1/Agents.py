import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import FAISS
from pathlib import Path
import json


load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = os.getenv("OPENAI_BASE_URL")


llm = ChatOpenAI(
    model_name = "openai/gpt-oss-20b:free",
    # model_name = "deepseek/deepseek-chat-v3.1:free",
    api_key=openrouter_key,
    base_url=openrouter_base_url
)


emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}  # normalize for cosine similarity
)

faiss_index = None

def detect_file_type(file_path: str):
    ext = Path(file_path).suffix.lower()
    return ext


#PDF LOADER
def Pdf_loader(filepath:str):
    pdf = PyPDFLoader(filepath)
    pdf_docs = pdf.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pdf_docs)
    pdf_index = FAISS.from_documents(all_splits, emb)
    pdf_index.save_local("pdf-index")

    


#TEXT LOADER
def Text_loader(filepath:str):
    text = TextLoader(filepath)
    text_docs = text.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(text_docs)
    text_index = FAISS.from_documents(all_splits, emb)
    text_index.save_local("text-index")


#MD LOADER
def Md_loader(filepath):
    md = UnstructuredMarkdownLoader(filepath)
    md_docs = md.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(md_docs)    
    md_index = FAISS.from_documents(all_splits, emb)
    md_index.save_local("md-index")



# memory = InMemorySaver()
# config = {"configurable": {"thread_id": "1"}}

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def Load_Docs(filename: str):

    global faiss_index
    # Detect file type once
    file_type = detect_file_type(filename)

    if file_type == ".pdf":
        if not os.path.exists("pdf-index"):
            Pdf_loader(filename)
        faiss_index = FAISS.load_local("pdf-index", emb, allow_dangerous_deserialization=True )
    elif file_type == ".txt":
        if not os.path.exists("text-index"):
            Text_loader(filename)
        faiss_index = FAISS.load_local("text-index", emb, allow_dangerous_deserialization=True )
    elif file_type == ".md":
        if not os.path.exists("md-index"):
            Md_loader(filename)
        faiss_index = FAISS.load_local("md-index", emb, allow_dangerous_deserialization=True )
    else:
        raise ValueError("Unsupported file format")

    return faiss_index

def retrieve(state: State):
    retrieved_docs = faiss_index.similarity_search(state["question"])
    return{"context": retrieved_docs}


def generate(state: State):

    context = state["context"]
    question = state["question"]

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    prompt =f"""You are an intelligent research assistant designed to provide clear, accurate, and well-structured answers.

            Use the given context to create a complete, factual, and professional response to the question.

            Guidelines:
            - Focus only on the provided context; add reasoning only when necessary.
            - Keep the tone formal and coherent.
            - Present information in organized paragraphs with clear explanations.
            - Conclude with a concise summary of the key points.

            Context:
            {docs_content}

            Question:
            {question}

            Answer:
            """
    
    response = llm.invoke(prompt)
    return {"answer": response.content}


Graph_builder = StateGraph(State)

Graph_builder.add_node("Retriver", retrieve)
Graph_builder.add_node("Generator", generate)

Graph_builder.add_edge(START, "Retriver")
Graph_builder.add_edge("Retriver", "Generator")
Graph_builder.add_edge("Generator", END)

graph = Graph_builder.compile()


