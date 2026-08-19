"""
Vector store setup for the Syllabus & Exam Assistant.

Mirrors the grounded-RAG pipeline from Assignment 2 (Exercises 3 & 4):
RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> FAISS.
The only difference is chunk_size, tuned a bit larger here so each
chunk keeps a whole syllabus section (e.g. the full grading breakdown)
together instead of splitting it mid-sentence.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from .data import SYLLABUS_DOCUMENTS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_syllabus_vectorstore(chunk_size: int = 400, chunk_overlap: int = 60) -> FAISS:
    """Splits syllabus documents and indexes them into an in-memory FAISS store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(SYLLABUS_DOCUMENTS)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def format_docs(docs) -> str:
    """Joins retrieved chunks into one grounded context block, tagged by course code."""
    if not docs:
        return ""
    blocks = [f"[{d.metadata.get('course_code', 'UNKNOWN')}] {d.page_content}" for d in docs]
    return "\n\n".join(blocks)
