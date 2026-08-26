"""
PDF ingestion for the Syllabus & Exam Assistant.

Validates and extracts text from uploaded syllabus/course-material PDFs so
they can be chunked, embedded, and added to the same FAISS index the
hardcoded and pasted-text syllabi already live in (see rag.py).

Kept deliberately dependency-light: pypdf is a pure-Python, actively
maintained PDF text extractor with no heavy native/ML dependencies, so it
fits the same "small and inspectable" footprint as the rest of the
backend. It does not do OCR -- scanned/image-only PDFs will extract little
or no text, which is treated as a validation failure rather than silently
indexing an empty document that would make search_syllabus look "grounded"
in content that isn't actually there.
"""
import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB -- generous for a syllabus/chapter, small enough to stay fast
MIN_EXTRACTABLE_CHARS = 40  # below this, treat the PDF as unreadable/scanned rather than indexing near-nothing
MIN_PAGE_CHARS = 15  # skip near-blank pages (dividers, cover pages) rather than indexing noise


class PDFValidationError(ValueError):
    """Raised for any PDF that fails validation or yields no usable text."""


def validate_pdf_header(data: bytes) -> None:
    """Cheap sanity check before handing bytes to pypdf: real PDFs start
    with the %PDF- magic bytes. Catches empty uploads, oversized files, and
    non-PDF files masquerading with a .pdf extension before doing any real
    parsing work.
    """
    if not data:
        raise PDFValidationError("Uploaded file is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise PDFValidationError(f"PDF is too large (max {MAX_PDF_BYTES // (1024 * 1024)} MB).")
    if not data.lstrip(b"\x00\t\n\r ")[:5].startswith(b"%PDF-"):
        raise PDFValidationError("File does not look like a valid PDF (missing %PDF header).")


def extract_pdf_pages(data: bytes) -> list[dict]:
    """Extracts per-page text from PDF bytes.

    Returns a list of {"page": 1-indexed page number, "text": cleaned text}
    for every page that yields non-trivial text. Raises PDFValidationError
    if the file can't be parsed, is encrypted without a usable empty
    password, or yields effectively no extractable text at all (most
    likely a scanned/image-only PDF, which this lightweight extractor
    can't OCR).
    """
    validate_pdf_header(data)

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as e:
        raise PDFValidationError(f"Couldn't parse PDF: {e}") from e
    except Exception as e:  # pypdf can raise a few non-PdfReadError types on malformed files
        raise PDFValidationError(f"Couldn't parse PDF: {e}") from e

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # only handles PDFs encrypted with an empty owner/user password
        except Exception:
            pass
        if reader.is_encrypted:
            raise PDFValidationError("This PDF is password-protected -- please upload an unlocked copy.")

    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(text) >= MIN_PAGE_CHARS:
            pages.append({"page": i, "text": text})

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars < MIN_EXTRACTABLE_CHARS:
        raise PDFValidationError(
            "Couldn't extract readable text from this PDF -- it may be a scanned image "
            "rather than real text, which this uploader can't OCR."
        )
    return pages
