import os
import time
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE=1000
CHUNK_OVERLAP = 150
TOP_K = 4

def load_pdfs(file_paths):
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

FAISS_INDEX_PATH = "faiss_index"

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization = True
    )

def index_exists():
    from pathlib import Path
    return Path(FAISS_INDEX_PATH).exists()

def build_qa_chain(vectorstore):
    llm = ChatGroq(
        model = LLM_MODEL,
        temperature = 0,
        groq_api_key = os.getenv("GROQ_API_KEY"),
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template = """You are FinSight, a financial doceument analyst.
        Answer using ONLY the context below. Never make up fidures or dates.
        Always cite the document name and page number.
        If the answer is not in the context, say so clearly.
        Context:
        {context}

        Question: {question}
        Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents = True,
    )

def build_pipeline(file_paths):
    t0 = time.time()
    docs = load_pdfs(file_paths)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain = build_qa_chain(vectorstore)

    stats= {
        "num_documents": len(file_paths),
        "num_pages": len(docs),
        "num_chunks": len(chunks),
        "index_time_s": round(time.time() - t0, 2),
        "index_saved": True,
    }
    return chain, stats

def compute_relevance_score(question, source_documents, vectorstore=None):
    """
    Compute relevance as the proportion of retrieved chunks
    that contain keywords from the question.
    Simple, fast, and interpretable — no re-embedding needed.
    """
    if not source_documents:
        return 0.0

    keywords = [
        w.lower() for w in question.split()
        if len(w) > 3 and w.lower() not in
        {"what", "when", "where", "which", "does", "this", "that", "with", "from", "have", "will"}
    ]

    if not keywords:
        return 0.0

    matched_chunks = 0
    for doc in source_documents:
        content = doc.page_content.lower()
        if any(keyword in content for keyword in keywords):
            matched_chunks += 1

    return matched_chunks / len(source_documents)