"""Pydantic request/response models for the FastAPI backend.

Same style as the Task/TaskItem models in the Demystifying APIs
notebook: plain BaseModel subclasses with type-hinted fields.

v1.1: added ChatRequest.mode, SourceChunk.page (page-aware citations for
PDF-sourced material), ChatResponse.mode (echoes back which mode actually
applied), and CourseInfo.source/page_count so the frontend can show a
course as PDF- vs text-sourced.

v1.2: added ChatRequest.device_id (identifies which device's private
notebook, if any, this turn's search_notebook tool should be scoped to --
see backend/notebook_store.py and backend/tools.py's make_notebook_tool)
and NotebookDocInfo, the /notebook equivalent of CourseInfo.

Solutions addition: CourseInfo.has_solutions / NotebookDocInfo.has_solutions
report whether an answer-key PDF has been attached via
POST /courses/{course_code}/solutions or POST /notebook/{doc_id}/solutions.

Flashcards addition: Flashcard/FlashcardDeck/FlashcardsRequest support
POST /flashcards, triggered client-side by a bare "/flashcards" command
typed into the composer (see frontend/app.js) rather than a chat message --
it never goes through ChatRequest/the agent's tool loop at all.
FlashcardDeck is what the endpoint actually returns, with the title/source
filled in server-side.

v1.3: added the "depth" mode (Deep Explainer). Round 2 of v1.3 removed the
older eli5/teach modes and the mode-bar UI entirely -- ChatRequest.mode now
only ever carries "depth" or "exam", and both are one-shot "/depth"/"/exam"
commands typed straight into the composer, never a sticky selection. That
same round also reworked flashcards to be conversation-driven instead of
course/notebook-driven: FlashcardsRequest dropped target/device_id for a
bare session_id, GeneratedFlashcards was removed (nothing built decks from
a resolved course/notebook target anymore), and GeneratedTopicFlashcards
took its place as the schema handed to the model's structured-output mode
(see agent.py's generate_flashcards_from_conversation). It also added
ChatRequest.notebook_doc_ids, so a chat turn can be pinned to exactly the
notebook PDF(s) the student selected in the sidebar (see the field's own
comment below, and tools.py's make_notebook_tool).

Quiz addition (v1.3 round 3): QuizQuestion/GeneratedTopicQuiz/QuizDeck/
QuizRequest support POST /quiz (a bare "/quiz" command, same
conversation-driven shape as /flashcards) and POST /retry-quiz (a bare
"/retry" command). QuizAnswer/QuizResultsRequest support POST
/quiz/results, which the frontend calls once a quiz is finished (grading
already done client-side against each question's correct_index) purely so
the session has real graded misses on record -- see agent.py's
record_quiz_results()/generate_retry_quiz() -- for /retry-quiz to draw on
later. This is the "real multiple-choice quiz mode with explicit
right/wrong grading" the original retry-exam request was deferred behind,
rather than inferring correctness from free-form chat.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str
    course_code: str | None = None
    mode: str | None = None  # None/"normal", "exam", "depth" -- see agent.py's MODE_PROMPTS
    device_id: str | None = None
    # When non-empty, this turn is pinned to exactly these notebook doc_ids
    # (see notebook_store.py) -- search_notebook is restricted to them and
    # search_syllabus is left out of the toolset entirely for the call, so
    # the answer is grounded in only what the student selected. Empty/None
    # leaves today's default behavior (search_notebook covers the whole
    # device's notebook, search_syllabus is still available) untouched.
    notebook_doc_ids: list[str] | None = None


class SourceChunk(BaseModel):
    course_code: str
    snippet: str
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    grounded: bool = False
    sources: list[SourceChunk] = []
    tools_used: list[str] = []
    mode: str | None = None


class CourseInfo(BaseModel):
    course_code: str
    course_name: str
    custom: bool = False
    source: str = "text"
    page_count: int | None = None
    has_solutions: bool = False


class AddCourseRequest(BaseModel):
    course_code: str
    course_name: str
    content: str


class NotebookDocInfo(BaseModel):
    doc_id: str
    title: str
    page_count: int
    has_solutions: bool = False


class ResetResponse(BaseModel):
    status: str
    session_id: str
    summary: str | None = None


class Flashcard(BaseModel):
    front: str
    back: str


class GeneratedTopicFlashcards(BaseModel):
    """Schema passed to the model's structured-output mode for the
    conversation-driven path -- see agent.py's
    generate_flashcards_from_conversation(). There's no resolved
    course/notebook title known ahead of time here (unlike the old,
    now-removed course/notebook-grounded path): the topic itself has to be
    inferred from the conversation, so the model reports it alongside the
    cards rather than the caller supplying it.
    """
    topic: str
    cards: list[Flashcard]


class FlashcardDeck(BaseModel):
    """Response shape for POST /flashcards."""
    title: str
    source: str  # "conversation" -- see generate_flashcards_from_conversation
    cards: list[Flashcard]


class FlashcardsRequest(BaseModel):
    # v1.3 round 2: flashcards are generated from what's actually been
    # discussed in this chat session (see agent.py's recent_transcript() /
    # generate_flashcards_from_conversation()), not from a named course or
    # notebook doc -- so all this endpoint needs is which session to read.
    session_id: str


# --- Quiz mode + retry-exam ---------------------------------------------
# The multiple-choice, explicitly-graded counterpart to flashcards: built
# the same conversation-driven way (see GeneratedTopicFlashcards above),
# but grading a quiz answer is a definite right/wrong against
# correct_index rather than "did the student seem to understand this
# flashcard" -- which is exactly what makes a real retry-exam possible:
# generate_retry_quiz() (agent.py) can draw on actual graded misses
# instead of trying to infer them from free-form chat.


class QuizQuestion(BaseModel):
    question: str
    options: list[str]  # exactly 4
    correct_index: int  # 0-3
    explanation: str  # shown to the student right after they answer, right or wrong


class GeneratedTopicQuiz(BaseModel):
    """Schema passed to the model's structured-output mode for both quiz
    generation paths -- see agent.py's generate_quiz_from_conversation()
    (fresh quiz) and generate_retry_quiz() (review of past misses). Same
    "the model reports its own topic" shape as GeneratedTopicFlashcards,
    for the same reason: there's no resolved course/notebook target to
    title the deck from ahead of time.
    """
    topic: str
    questions: list[QuizQuestion]


class QuizDeck(BaseModel):
    """Response shape for POST /quiz and POST /retry-quiz."""
    title: str
    source: str  # "conversation" | "retry"
    questions: list[QuizQuestion]


class QuizRequest(BaseModel):
    # Same reasoning as FlashcardsRequest: both /quiz and /retry-quiz only
    # need to know which session's transcript/quiz history to read.
    session_id: str


class QuizAnswer(BaseModel):
    """One graded question from a quiz the student just took -- see
    POST /quiz/results. Grading itself already happened client-side (the
    frontend has correct_index from the QuizDeck it rendered); this is
    just the record of what happened, kept server-side so a later
    /retry-quiz has real misses to draw on instead of nothing.
    """
    question: str
    options: list[str]
    correct_index: int
    chosen_index: int


class QuizResultsRequest(BaseModel):
    session_id: str
    topic: str
    answers: list[QuizAnswer]
