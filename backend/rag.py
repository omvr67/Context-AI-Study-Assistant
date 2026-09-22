"""
Vector store setup for the Syllabus & Exam Assistant.

Mirrors the grounded-RAG pipeline from Assignment 2 (Exercises 3 & 4):
RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> FAISS.
chunk_size is tuned a bit larger than the assignment default so each
chunk keeps a whole syllabus section (e.g. the full grading breakdown)
together instead of splitting it mid-sentence.

On top of the base pipeline, this module also indexes any locally
saved custom syllabi (see custom_courses.py) at startup, and exposes
add_course_to_vectorstore / add_pdf_course_to_vectorstore /
remove_course_from_vectorstore so the /courses and /courses/upload
endpoints can update the live FAISS index in place -- no full rebuild
needed for any of them.

v1.1 additions:
  - page-aware indexing for PDF uploads: each chunk keeps the page number
    of the page it was split from, so format_docs can cite "CS301 p.4"
    instead of just the course code (see custom_courses.py's "pages" field).
  - retrieve_relevant_chunks(): a relevance-floor + per-course-cap wrapper
    around similarity search, used by search_syllabus so tangentially
    related chunks get dropped instead of passed to the LLM as if they were
    a real match, and so a whole-store search can't be dominated by one
    course's chunks.
  - get_course_full_text(): reads a course's full, un-chunked text straight
    from its source of truth (data.py / custom_syllabi.json) rather than
    via chunk search -- used by tools (build_ai_study_plan) that need the
    whole document, since chunk search can fragment or miss a topic list.

v1.2 addition:
  - add_notebook_doc_to_vectorstore(): indexes a private, per-device
    notebook PDF into this same store, tagged with device_id + notebook=True
    so it can only ever surface through search_notebook's own metadata
    filter (see backend/tools.py and backend/notebook_store.py) -- never
    through search_syllabus, and never for a different device.

Solutions addition:
  - add_course_solutions_to_vectorstore() / add_notebook_solutions_to_vectorstore():
    index an attached answer-key PDF under the same course_code/doc_id as
    its source material (so it's retrieved together, by the same existing
    search_syllabus/search_notebook calls), tagged material="solutions" so
    format_docs can mark it distinctly in citations. build_syllabus_vectorstore
    now returns a third dict, course_solutions_chunk_ids, tracking these
    chunks separately from a course's own so one can be replaced without
    touching the other.
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

# --- retrieval tuning -------------------------------------------------------
DEFAULT_K = 4
FETCH_K = 12  # cast a wider net before filtering, so the relevance floor has something to actually filter
# similarity_search_with_relevance_scores() normalizes FAISS's raw L2 distance into
# a rough 0..1 "higher is better" score. It's an approximation, not a calibrated
# probability -- but it's enough to reliably separate "on-topic chunk" from
# "nothing useful matched" for a store this size, which plain top-k similarity
# search can't do (top-k always returns k results even when none are relevant).
RELEVANCE_FLOOR = 0.22


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _custom_course_documents(course: dict) -> list[Document]:
    """Builds the Document(s) for one saved custom course. PDF-sourced
    courses get one Document per page (so page metadata survives into every
    chunk split from it); older/pasted-text courses get a single Document,
    same as before. If an answer-key PDF has been attached (course["solutions"]),
    its pages are appended too, tagged material="solutions" so
    build_syllabus_vectorstore can route their chunk ids into a separate
    tracking dict (course_solutions_chunk_ids) from the course's own
    material -- see that function's docstring for why that separation
    matters.
    """
    pages = course.get("pages")
    if pages:
        docs = [
            Document(
                page_content=p["text"],
                metadata={
                    "course_code": course["course_code"],
                    "course_name": course["course_name"],
                    "custom": True,
                    "source": course.get("source", "pdf"),
                    "page": p["page"],
                },
            )
            for p in pages
        ]
    else:
        docs = [Document(
            page_content=course["content"],
            metadata={
                "course_code": course["course_code"],
                "course_name": course["course_name"],
                "custom": True,
                "source": course.get("source", "text"),
            },
        )]

    solutions_pages = course.get("solutions")
    if solutions_pages:
        docs.extend(
            Document(
                page_content=p["text"],
                metadata={
                    "course_code": course["course_code"],
                    "course_name": course["course_name"],
                    "custom": True,
                    "material": "solutions",
                    "page": p["page"],
                },
            )
            for p in solutions_pages
        )
    return docs


def build_syllabus_vectorstore() -> tuple[FAISS, dict[str, list[str]], dict[str, list[str]]]:
    """Builds the FAISS store from the hardcoded syllabi plus any locally
    saved custom syllabi (see custom_courses.py), including any attached
    solutions PDFs.

    Returns (vectorstore, course_chunk_ids, course_solutions_chunk_ids):
    course_chunk_ids maps each course_code to the FAISS ids of its own
    chunks, and course_solutions_chunk_ids maps it to the ids of any
    attached answer-key chunks -- kept in a *separate* dict (routed there
    automatically inside _index, based on each chunk's material metadata)
    so a solutions PDF can later be replaced with vectorstore.delete() on
    just its own ids, without touching -- or needing to re-embed -- the
    course's actual material. Either can be removed with
    remove_course_from_vectorstore(vectorstore, ids) instead of a full
    rebuild.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = _splitter()
    course_chunk_ids: dict[str, list[str]] = {}
    course_solutions_chunk_ids: dict[str, list[str]] = {}
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
            target = course_solutions_chunk_ids if chunk.metadata.get("material") == "solutions" else course_chunk_ids
            target.setdefault(code, []).append(chunk_id)

    _index(SYLLABUS_DOCUMENTS)
    for course in load_custom_courses():
        _index(_custom_course_documents(course))

    if vectorstore is None:
        # Should never happen -- SYLLABUS_DOCUMENTS is never empty -- but keep
        # startup from crashing outright if data.py is ever emptied out.
        vectorstore = FAISS.from_documents(
            [Document(page_content="No syllabi indexed yet.", metadata={"course_code": "NONE"})],
            embeddings,
        )

    return vectorstore, course_chunk_ids, course_solutions_chunk_ids


def add_course_to_vectorstore(vectorstore: FAISS, course_code: str, course_name: str, content: str) -> list[str]:
    """Chunks + embeds one new pasted-text syllabus and adds it to the live
    index. Returns the new chunk ids so the caller can track them for later
    removal.
    """
    doc = Document(
        page_content=content,
        metadata={"course_code": course_code, "course_name": course_name, "custom": True, "source": "text"},
    )
    chunks = _splitter().split_documents([doc])
    return vectorstore.add_documents(chunks)


def add_pdf_course_to_vectorstore(
    vectorstore: FAISS, course_code: str, course_name: str, pages: list[dict]
) -> list[str]:
    """Chunks + embeds a page-aware PDF upload and adds it to the live
    index, keeping each resulting chunk's originating page number in its
    metadata (see format_docs) so an answer can cite "CS301 p.4" instead of
    just the course code. Returns the new chunk ids.
    """
    page_docs = [
        Document(
            page_content=p["text"],
            metadata={
                "course_code": course_code,
                "course_name": course_name,
                "custom": True,
                "source": "pdf",
                "page": p["page"],
            },
        )
        for p in pages
    ]
    chunks = _splitter().split_documents(page_docs)
    return vectorstore.add_documents(chunks)


def add_notebook_doc_to_vectorstore(
    vectorstore: FAISS, device_id: str, doc_id: str, title: str, pages: list[dict]
) -> list[str]:
    """Chunks + embeds one private notebook PDF and adds it to the live
    index, tagged with device_id + notebook=True so search_notebook's
    metadata filter -- and only that filter -- can ever surface it (see
    tools.make_notebook_tool). Keeps each chunk's page number, same as
    add_pdf_course_to_vectorstore, so notebook citations can read "p.N"
    too. Returns the new chunk ids so the caller can track them for later
    removal, same convention as the course-upload path.
    """
    page_docs = [
        Document(
            page_content=p["text"],
            metadata={
                "course_code": title,
                "course_name": title,
                "device_id": device_id,
                "doc_id": doc_id,
                "notebook": True,
                "page": p["page"],
            },
        )
        for p in pages
    ]
    chunks = _splitter().split_documents(page_docs)
    return vectorstore.add_documents(chunks)


def add_course_solutions_to_vectorstore(
    vectorstore: FAISS, course_code: str, course_name: str, pages: list[dict]
) -> list[str]:
    """Chunks + embeds a solutions/answer-key PDF attached to an existing
    custom course and adds it to the live index, tagged material="solutions"
    (see format_docs) so a citation -- and the agent's own reasoning -- can
    tell an answer key apart from the original material, even though both
    share the same course_code and are retrieved together by
    search_syllabus's ordinary course_code filter. Returns the new chunk
    ids; callers should track these separately from the course's own (see
    backend/main.py's course_solutions_chunk_ids) so a later re-upload can
    remove just the old solutions chunks before adding a replacement.
    """
    page_docs = [
        Document(
            page_content=p["text"],
            metadata={
                "course_code": course_code,
                "course_name": course_name,
                "custom": True,
                "material": "solutions",
                "page": p["page"],
            },
        )
        for p in pages
    ]
    chunks = _splitter().split_documents(page_docs)
    return vectorstore.add_documents(chunks)


def add_notebook_solutions_to_vectorstore(
    vectorstore: FAISS, device_id: str, doc_id: str, title: str, pages: list[dict]
) -> list[str]:
    """Notebook equivalent of add_course_solutions_to_vectorstore: chunks +
    embeds a solutions PDF attached to one of a device's own notebook docs,
    tagged material="solutions" plus the same device_id + notebook=True +
    doc_id metadata the source notebook doc itself carries, so
    search_notebook's existing device_id filter covers the attached
    solutions automatically -- no separate tool or filter change needed.
    """
    page_docs = [
        Document(
            page_content=p["text"],
            metadata={
                "course_code": title,
                "course_name": title,
                "device_id": device_id,
                "doc_id": doc_id,
                "notebook": True,
                "material": "solutions",
                "page": p["page"],
            },
        )
        for p in pages
    ]
    chunks = _splitter().split_documents(page_docs)
    return vectorstore.add_documents(chunks)


def remove_course_from_vectorstore(vectorstore: FAISS, chunk_ids: list[str]) -> None:
    """Removes previously-tracked chunk ids from the live index in place."""
    if chunk_ids:
        vectorstore.delete(chunk_ids)


def retrieve_relevant_chunks(
    vectorstore: FAISS, query: str, course_code: str = "", k: int = DEFAULT_K
) -> tuple[list[Document], float]:
    """Retrieves up to k chunks for a query, on top of plain top-k similarity search:

      - optional metadata filtering by course_code (unchanged from before),
      - a minimum relevance floor so chunks that merely happen to be the
        "least bad" match get dropped instead of handed to the LLM as if
        they were a real answer -- the direct cause of confidently-wrong
        "grounded" answers when nothing on-topic actually exists,
      - a light per-course cap so, when searching across all courses, one
        course's chunks can't fill every slot and crowd out a real match
        from a different course.

    Returns (docs, best_score) -- best_score is the top raw relevance score
    seen (0.0 if nothing cleared the floor), so callers can tell "nothing
    relevant" apart from "found stuff" without re-deriving it from the
    returned docs.
    """
    search_filter = {"course_code": course_code} if course_code else None
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=FETCH_K, filter=search_filter)
    except Exception:
        # Some vectorstore configurations don't support relevance-score
        # normalization -- fall back to plain top-k rather than failing the
        # whole search, just without the relevance floor.
        docs = vectorstore.similarity_search(query, k=k, filter=search_filter)
        return docs, (1.0 if docs else 0.0)

    scored = [(doc, score) for doc, score in scored if score >= RELEVANCE_FLOOR]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    picked: list[Document] = []
    per_course_count: dict[str, int] = {}
    max_per_course = k if course_code else max(2, k - 1)
    for doc, _score in scored:
        code = doc.metadata.get("course_code", "UNKNOWN")
        if per_course_count.get(code, 0) >= max_per_course:
            continue
        picked.append(doc)
        per_course_count[code] = per_course_count.get(code, 0) + 1
        if len(picked) >= k:
            break

    best_score = scored[0][1] if scored else 0.0
    return picked, best_score


def get_course_full_text(course_code: str) -> str | None:
    """Returns the full, un-chunked syllabus text for a course code, straight
    from its source of truth (data.py for built-ins, custom_syllabi.json for
    saved/uploaded ones), or None if the code isn't indexed anywhere.

    Used by tools that need the whole document -- e.g. build_ai_study_plan's
    topic extraction -- rather than a handful of retrieved chunks, since
    chunk search can fragment or entirely miss a "Week N: Topic" list
    depending on how it happens to land relative to chunk boundaries.
    """
    code = course_code.strip().upper()
    for doc in SYLLABUS_DOCUMENTS:
        if doc.metadata.get("course_code") == code:
            return doc.page_content
    for course in load_custom_courses():
        if course["course_code"] == code:
            return course.get("content", "")
    return None


def format_docs(docs) -> str:
    """Joins retrieved chunks into one grounded context block, tagged by
    course code and, for PDF-sourced chunks, page number (e.g.
    "[CS301|p.4] ..."). agent.py's _parse_sources knows this exact format
    and splits the "|p.N" suffix back out for the API response.

    A chunk from an attached answer-key PDF (material="solutions") gets
    " Solutions" appended to its display code (e.g. "[CS301 Solutions|p.2]
    ...") so it reads as distinct from the original material in both the
    agent's own context and the citations shown to the student, even
    though both share the same underlying course_code/doc_id and are
    retrieved together.
    """
    if not docs:
        return ""
    blocks = []
    for d in docs:
        code = d.metadata.get("course_code", "UNKNOWN")
        if d.metadata.get("material") == "solutions":
            code = f"{code} Solutions"
        page = d.metadata.get("page")
        tag = f"{code}|p.{page}" if page else code
        blocks.append(f"[{tag}] {d.page_content}")
    return "\n\n".join(blocks)
