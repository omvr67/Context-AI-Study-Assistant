"""
Vector store setup for the Syllabus & Exam Assistant.

Mirrors the grounded-RAG pipeline from Assignment 2 (Exercises 3 & 4):
RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> FAISS.
chunk_size is tuned a bit larger than the assignment default so each
chunk keeps a whole syllabus section (e.g. the full grading breakdown)
together instead of splitting it mid-sentence.

On top of the base pipeline, this module also indexes any locally
saved custom syllabi (see custom_courses.py) at startup, and exposes
add_course_to_vectorstore / remove_course_from_vectorstore so the
/courses POST and DELETE endpoints can update the live FAISS index
in place -- no full rebuild needed for either operation.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from .custom_courses import load_custom_courses
from .data import SYLLABUS_DOCUMENTS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def build_syllabus_vectorstore() -> tuple[FAISS, dict[str, list[str]]]:
    """Builds the FAISS store from the hardcoded syllabi plus any locally
    saved custom syllabi (see custom_courses.py).

    Returns (vectorstore, course_chunk_ids) where course_chunk_ids maps
    each course_code to the FAISS ids of its chunks, so a single course
    can later be removed with vectorstore.delete(ids) instead of a full
    rebuild.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = _splitter()
    course_chunk_ids: dict[str, list[str]] = {}
    vectorstore: FAISS | None = None

    def _index(documents: list[Document]) -> None:
        nonlocal vectorstore
        chunks = splitter.split_documents(documents)
        if not chunks:
            return
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            # from_documents doesn't hand back ids directly, but the wrapper
            # builds this map in the same order the chunks were embedded.
            new_ids = list(vectorstore.index_to_docstore_id.values())
        else:
            new_ids = vectorstore.add_documents(chunks)
        for chunk, chunk_id in zip(chunks, new_ids):
            code = chunk.metadata.get("course_code", "UNKNOWN")
            course_chunk_ids.setdefault(code, []).append(chunk_id)

    _index(SYLLABUS_DOCUMENTS)
    for course in load_custom_courses():
        _index([Document(
            page_content=course["content"],
            metadata={
                "course_code": course["course_code"],
                "course_name": course["course_name"],
                "custom": True,
            },
        )])

    if vectorstore is None:
        # Should never happen -- SYLLABUS_DOCUMENTS is never empty -- but keep
        # startup from crashing outright if data.py is ever emptied out.
        vectorstore = FAISS.from_documents(
            [Document(page_content="No syllabi indexed yet.", metadata={"course_code": "NONE"})],
            embeddings,
        )

    return vectorstore, course_chunk_ids


def add_course_to_vectorstore(vectorstore: FAISS, course_code: str, course_name: str, content: str) -> list[str]:
    """Chunks + embeds one new syllabus and adds it to the live index.
    Returns the new chunk ids so the caller can track them for later removal.
    """
    doc = Document(
        page_content=content,
        metadata={"course_code": course_code, "course_name": course_name, "custom": True},
    )
    chunks = _splitter().split_documents([doc])
    return vectorstore.add_documents(chunks)


def remove_course_from_vectorstore(vectorstore: FAISS, chunk_ids: list[str]) -> None:
    """Removes previously-tracked chunk ids from the live index in place."""
    if chunk_ids:
        vectorstore.delete(chunk_ids)


def format_docs(docs) -> str:
    """Joins retrieved chunks into one grounded context block, tagged by course code."""
    if not docs:
        return ""
    blocks = [f"[{d.metadata.get('course_code', 'UNKNOWN')}] {d.page_content}" for d in docs]
    return "\n\n".join(blocks)
