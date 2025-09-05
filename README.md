# RAG-PDF-Assistant
A simple Retrieval-Augmented Generation (RAG) workflow using LangChain, LangGraph, HuggingFace embeddings, and OpenRouter LLMs to query PDF documents in natural language.
RAG Pipeline with LangGraph and OpenRouter
This repository contains a Retrieval-Augmented Generation (RAG) pipeline built with LangGraph and powered by OpenRouter. The project demonstrates how to build a robust and modular system for querying custom PDF documents.

Key Features
PDF Processing: Utilizes PyPDFLoader to ingest PDF documents, which are then split into chunks using RecursiveCharacterTextSplitter.

Vector Storage: Documents are embedded using HuggingFaceEmbeddings and stored in an InMemoryVectorStore for efficient retrieval.

LLM Integration: Leverages ChatOpenAI via OpenRouter to access powerful language models for generating answers.

State Management with LangGraph: A LangGraph state machine orchestrates the entire process, from retrieving relevant document chunks to generating a final, context-aware response.

How It Works
The pipeline follows a clear, multi-step process:

Ingestion: A PDF file is loaded and processed.

Chunking & Embedding: The document is broken into smaller chunks and converted into vector embeddings.

Retrieval: When a question is asked, the system retrieves the most relevant chunks from the vector store.

Generation: The retrieved context and the user's question are passed to the LLM, which generates a comprehensive answer.

Prerequisites
Python 3.x

A .env file with your OPENROUTER_API_KEY and OPENAI_BASE_URL.

Run the script:
The main script processes the PDF and allows you to interact with the RAG system by asking questions.

Technologies Used
LangGraph: For building the stateful, cyclic graphs.

LangChain: For document loading, text splitting, and vector store management.

OpenRouter: A unified API for accessing various LLMs.

HuggingFace: For text embedding models.
