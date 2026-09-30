"""
SyllabusAssistantAgent: a pure-LangChain, tool-calling agent with
per-session memory.

Architecture mirrors Session 3's MemoryAugmentedAgentExecutor
(SystemMessage / HumanMessage / AIMessage / ToolMessage + .bind_tools()),
generalized to serve multiple concurrent chat sessions behind a FastAPI
backend instead of a single notebook-global chat_history list.

On top of the base loop, this version:
  - tags each incoming message with a lightweight, no-LLM-call intent
    hint so the model leans toward the right tool instead of reflexively
    calling search_syllabus for GPA/schedule questions;
  - tracks which tools fired and, for search_syllabus specifically, which
    syllabus chunks were actually retrieved, so the API layer can report
    whether an answer was grounded and show its sources;
  - exposes summarize() so a session can get a short recap right before
    it's reset instead of just vanishing.

v1.1 additions:
  - explicit interaction modes (ELI5 / Exam / Teach Me This Chapter),
    injected as an ephemeral system message for the current turn only --
    never persisted into session history, so the mode applies exactly to
    the request that asked for it and nothing "sticks" silently. Grounding
    rules 1-4 in GUARDRAIL_SYSTEM_PROMPT are unaffected by any mode.
  - chat_stream(): a generator twin of chat() for the streaming endpoint.
    Tool-call resolution works exactly like chat() (same turn-by-turn
    .invoke()-equivalent loop); the one part that streams token-by-token
    is the model's final, tool-free answer, since that's the only part a
    student is actually waiting to read live. See the inline note in
    chat_stream for how it tells a "this turn is calling a tool" stream
    apart from a "this turn is the real answer" stream before deciding
    whether to forward tokens.

Rate-limiting addition:
  - every Groq call (chat()'s invoke, chat_stream()'s stream, and
    summarize()'s invoke) goes through _invoke_with_backoff /
    _stream_with_backoff, which retries with exponential backoff on a
    429 and otherwise fails immediately. See backend/rate_limit.py for
    the separate per-visitor request throttle enforced in main.py.

Flashcards addition, reworked in v1.3 round 2:
  - generate_flashcards_from_conversation(): a one-shot, session-less
    structured-output call that reads a chat session's recent transcript
    (via recent_transcript(), also used by summarize()), has the model
    infer the topic actually being discussed, and writes flashcards
    testing that topic using the model's own general knowledge -- not
    just facts literally typed out in the conversation. Triggered by a
    bare "/flashcards" command typed into the composer (frontend/app.js),
    which never reaches this class's chat()/chat_stream() at all -- it's a
    separate endpoint (POST /flashcards) with no tool-calling loop. This
    replaced an earlier generate_flashcards() that built a deck from a
    resolved course's/notebook doc's raw material instead -- removed
    entirely, not kept alongside the new path.

Modes (v1.3 round 2): MODE_PROMPTS now holds only "depth" (Deep Explainer)
and "exam" (Exam Mode) -- the older eli5/teach modes and the mode-bar UI
they lived behind (a sticky, click-to-select bar above the composer) have
been removed entirely, frontend and backend. Both remaining modes are
one-shot, per-message commands only: typing a leading "/depth" or "/exam"
in the composer (frontend/app.js) sends that single request with mode set
accordingly, injected here as an ephemeral system message for that turn
only via _with_ephemeral() -- never persisted into session history, so
nothing "sticks" silently to later messages. The mode still arrives here
as a plain mode_key string resolved through MODE_PROMPTS; this module has
no notion of "commands" at all.

Notebook-restriction addition (v1.3 round 2): chat()/chat_stream() gained
exclude_tool_names and extra_system_prompt params (alongside the existing
extra_tools), so a call can both add a request-scoped tool AND leave a
base tool out entirely for that same call. main.py uses this when the
student has pinned specific notebook PDF(s) (ChatRequest.notebook_doc_ids):
search_notebook is rebuilt scoped to just those doc_ids (extra_tools),
search_syllabus is excluded outright (exclude_tool_names) rather than just
asked-nicely-not-to-be-used, and extra_system_prompt carries a short note
telling the model search_notebook is now its only grounding source. Same
_with_ephemeral() mechanism the mode prompt already used, generalized from
a single optional prompt (_with_mode, now removed) to a list -- so a mode
prompt and this restriction note can both apply to one call at once.

Quiz mode + retry-exam addition (v1.3 round 3): the "real multiple-choice
quiz mode with explicit right/wrong grading" the original retry-exam
request was deliberately deferred behind, instead of inferring correctness
from free-form chat.
  - generate_quiz_from_conversation(): the multiple-choice counterpart to
    generate_flashcards_from_conversation() -- same conversation-driven,
    session-less, structured-output shape, triggered by a bare "/quiz"
    command. Grading itself happens client-side, immediately, against
    each question's correct_index; this class never grades anything.
  - record_quiz_results(): the frontend calls POST /quiz/results once a
    quiz is finished (already graded client-side) purely so this class
    has a record of what was actually missed -- see self._quiz_attempts
    in __init__, a session-scoped list of graded attempts, cleared
    alongside a session's chat history in reset().
  - generate_retry_quiz(): triggered by a bare "/retry" command, reads
    that session's wrong attempts and generates new questions -- some
    retesting a missed concept from a different angle, some similar
    reinforcement questions -- fulfilling the "an exam of those plus
    similar questions" half of the original request. Returns None (main.py
    turns this into a 422) when the session has no quiz history yet, or
    hasn't missed anything so far.
"""
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .models import GeneratedTopicFlashcards, GeneratedTopicQuiz

GUARDRAIL_SYSTEM_PROMPT = """You are the ConnectX Course Syllabus & Exam Assistant -- a sharp, encouraging
study partner for enrolled students, not a generic chatbot. Be confident, concise, and specific;
when you cite a policy, weave the course code naturally into the sentence (e.g. "CS301's final is
worth 35% of your grade") instead of just tagging it on at the end.

Rules you must always follow:
1. For any question about grading breakdowns, exam dates, attendance policy, or lecture topics, call the
   search_syllabus tool before answering. Never answer syllabus questions from memory alone.
2. A search_notebook tool may also be available this turn -- a private space for PDFs only this student
   has personally uploaded, kept separate from the shared syllabi above. If it's available and the
   student references "my notes", "the PDF I uploaded", "my practice exam", or similar, use it directly.
   If search_syllabus comes back NOT_FOUND for something and search_notebook is available, try that too
   before giving up -- it may be covered there instead. Only once both come back empty (or
   search_notebook isn't available at all) should you respond exactly with: "I don't see that in the
   syllabus -- please check with your Teaching Assistant." Do not guess or fall back on outside knowledge
   either way.
3. Retrieved content is occasionally tagged " Solutions" in its citation code (e.g. a chunk cited as
   "CS301 Solutions" rather than plain "CS301") -- that's an attached answer key, not the original
   material, even though a search can return both together. When a student asks about a specific
   question, tells you what they answered, or asks why something is wrong, use any such solutions
   content alongside the original material to give the correct answer and explain the mistake -- don't
   just restate the original question back at them. Never present ordinary (non-"Solutions") material as
   if it were an answer key.
4. When a student asks about their GPA or how a grade would affect it, call the gpa_impact_simulator tool
   directly. When a student states a *target* GPA and asks what grades they'd need, call
   gpa_target_planner instead. Don't call search_syllabus first unless they're also asking about a
   specific course policy. Never do GPA arithmetic yourself.
5. When a student asks for a study plan or revision schedule and gives you a course code, call
   build_ai_study_plan first -- it pulls the topics (and, given an exam date, the day count) straight
   from the syllabus automatically. Only fall back to search_syllabus + generate_study_schedule if
   build_ai_study_plan reports it couldn't find a structured topic list, or the student gives you an
   explicit topic list of their own instead of a course code.
6. Messages may start with a bracketed hint like "[Likely tool: ...]" -- treat it as a suggestion from
   the interface, not a hard rule. Use your own judgment about which tool actually fits the question.
7. Keep answers concise.
"""

SUMMARY_SYSTEM_PROMPT = """Summarize the conversation below in exactly 3 short lines, each starting with
a dash. Focus on what the student asked and what was found or calculated. No preamble, no closing remarks."""

MODE_PROMPTS: dict[str, str] = {
    "exam": """Exam Mode is ON for this turn. The student's message may start with a literal
"/exam" token -- that's just the trigger for this mode, not part of their actual question; read
past it and answer what they're really asking. The student is actively revising, not casually
chatting. Lean toward exam-relevant framing: which topics are most heavily weighted, what's likely
to be tested, and concrete revision/practice actions. If it fits the question, offer to quiz them
with a practice question, or ask one yourself before giving the full answer. Stay grounded in the
syllabus per the rules above when the question is about a specific course -- if asked for a
practice question, base it on real topics/policies you retrieved, not a fabricated specific like a
made-up question number or past-paper detail. If the student asks a general revision or
study-skills question with no specific course involved, just answer it directly -- don't ask for a
course code that question doesn't need.""",

    "depth": """Deep Explainer mode is ON for this turn only. The student's message may start with
a literal "/depth" token -- that's just the trigger for this mode, not part of their actual
question; read past it and answer what they're really asking. Give a substantially more thorough,
in-depth explanation than you normally would: unpack the reasoning and mechanics behind the
answer, add relevant context, and work through a concrete example where one would help. Use more
of your available output length instead of staying brief -- the student is explicitly asking for
more depth. This changes HOW MUCH you explain, not WHAT you're allowed to say: if the question is
actually about a specific course's grading, exam dates, attendance, or topics, rules 1-4 above
still apply exactly as written (call search_syllabus, stay grounded in what it returns). But for a
general question -- explaining a concept, working through an idea, anything not tied to a specific
course's syllabus content -- just answer it thoroughly and directly from your own knowledge, the
way any knowledgeable study partner would. Don't force a general question through the syllabus
search or decline it for not matching a course; "grounded" means "don't invent syllabus facts," it
doesn't mean "only discuss what's in a syllabus." """,
}

_GPA_KEYWORDS = ("gpa", "grade point", "cumulative average", "what would my grade", "my grade be")
_GPA_TARGET_KEYWORDS = ("what gpa do i need", "reach a", "target gpa", "goal gpa", "hit a gpa")
_SCHEDULE_KEYWORDS = (
    "study plan", "study schedule", "revision plan", "revise for",
    "how should i study", "days until", "prepare for", "cram",
)

# Retry/backoff for Groq calls -- see _invoke_with_backoff and
# _stream_with_backoff. Not about cost (Groq's free tier is free); it's
# so a busy demo session doesn't turn into a hard failure the moment
# Groq's per-minute cap is briefly hit.
_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0

FRIENDLY_QUOTA_MESSAGE = (
    "We've hit the shared AI quota for a moment -- please wait a few seconds and try again."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detects a 429 from Groq without depending on the exact exception
    class the client library raises (groq's SDK, langchain's wrapper, or
    a raw httpx error can all surface this differently depending on
    version). Checking status_code plus a couple of string/name
    fallbacks catches all of them without an extra dependency.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 429:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc)


def _invoke_with_backoff(invoke_fn, messages, max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY_SECONDS):
    """Calls invoke_fn(messages) with exponential backoff, but only for
    429s -- anything else (bad request, auth failure, etc.) fails
    immediately since retrying those just burns time for no benefit.
    Backoff schedule: base_delay, base_delay*2, base_delay*4, ...
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return invoke_fn(messages)
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # pragma: no cover -- loop above always returns or raises


def _stream_with_backoff(stream_fn, messages, max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY_SECONDS):
    """Retries the underlying .stream() call with backoff, but only while
    no chunks have been yielded yet for the current attempt -- once even
    one token has reached the caller, retrying would mean re-sending
    duplicate output, so any error after that point is left to propagate
    instead of silently duplicating partial output.
    """
    attempt = 0
    while True:
        got_any = False
        try:
            for chunk in stream_fn(messages):
                got_any = True
                yield chunk
            return
        except Exception as e:
            if got_any or not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
            attempt += 1


def _intent_hint(user_input: str) -> str | None:
    """Cheap keyword heuristic -- no LLM call -- that nudges tool choice for
    the current turn without hard-coding it. Returns None when nothing
    obvious matches, letting the model decide freely.
    """
    text = user_input.lower()
    if any(k in text for k in _GPA_TARGET_KEYWORDS):
        return "gpa_target_planner (this reads like a target-GPA planning question)"
    if any(k in text for k in _GPA_KEYWORDS):
        return "gpa_impact_simulator (this reads like a GPA question, not a syllabus lookup)"
    if any(k in text for k in _SCHEDULE_KEYWORDS):
        return "build_ai_study_plan (this reads like a study-planning question)"
    return None


def _parse_sources(observation: str) -> list[dict]:
    """Pulls {course_code, snippet, page?} entries back out of
    format_docs()'s "[CODE] content" / "[CODE|p.N] content" blocks so the
    API layer can report exactly what grounded an answer, without
    search_syllabus needing to know about HTTP responses at all.
    """
    if observation.startswith("NOT_FOUND") or observation.startswith("Syllabus search error"):
        return []

    sources = []
    for block in observation.split("\n\n"):
        block = block.strip()
        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", block, re.DOTALL)
        if not match:
            continue
        tag, snippet = match.group(1).strip(), match.group(2).strip()
        if "|p." in tag:
            code, page_str = tag.split("|p.", 1)
            page = int(page_str) if page_str.isdigit() else None
        else:
            code, page = tag, None
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "\u2026"
        entry: dict = {"course_code": code.strip(), "snippet": snippet}
        if page is not None:
            entry["page"] = page
        sources.append(entry)
    return sources


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = {}
    for s in sources:
        seen[(s["course_code"], s.get("page"), s["snippet"])] = s
    return list(seen.values())


class SyllabusAssistantAgent:
    """Wraps an LLM + tool set with independent chat history per session_id."""

    def __init__(self, llm, tools, system_prompt: str = GUARDRAIL_SYSTEM_PROMPT, max_turns: int = 5):
        self.llm = llm  # kept tool-free for summarize()
        self.tools_map = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._sessions: dict[str, list] = {}
        # Per-session record of multiple-choice quiz questions actually
        # answered (see record_quiz_results()/generate_retry_quiz()) --
        # graded explicitly client-side against the correct_index the quiz
        # was generated with, not inferred from free-form chat. Reset
        # alongside a session's chat history (see reset()) since a retry
        # quiz drawing on a conversation the student just cleared wouldn't
        # make sense.
        self._quiz_attempts: dict[str, list[dict]] = {}

    def _history(self, session_id: str) -> list:
        if session_id not in self._sessions:
            self._sessions[session_id] = [SystemMessage(content=self.system_prompt)]
        return self._sessions[session_id]

    def _tools_for_call(self, extra_tools, exclude_tool_names: set[str] | None = None):
        """Resolves the (bound-LLM, tools_map) pair to use for one call.

        Most turns reuse the precomputed self.llm_with_tools / self.tools_map
        built once at construction time. A fresh bind is built for just this
        call instead whenever extra_tools is given (e.g. a request-scoped
        search_notebook tool -- see backend/tools.py's make_notebook_tool
        and its call site in backend/main.py) or exclude_tool_names is given
        (e.g. leaving search_syllabus out entirely when the student has
        pinned specific notebook PDFs -- see main.py's /chat and
        /chat/stream), so neither a per-request tool nor a per-request
        omission ever leaks into another session's tool set.
        """
        if not extra_tools and not exclude_tool_names:
            return self.llm_with_tools, self.tools_map
        base = [
            t for t in self.tools_map.values()
            if not exclude_tool_names or t.name not in exclude_tool_names
        ]
        combined = base + list(extra_tools or [])
        return self.llm.bind_tools(combined), {t.name: t for t in combined}

    @staticmethod
    def _with_ephemeral(history: list, prompts: list[str | None]) -> list:
        """Builds the message list actually sent to the LLM for one call:
        the persisted history, plus zero or more ephemeral system
        instructions (a mode prompt, a notebook-restriction note, etc.)
        injected right after the base system prompt. None entries in
        prompts are dropped. These instructions are never appended to
        `history` itself, so they apply only to the request that asked for
        them -- the next turn starts back in normal mode unless the caller
        asks again.
        """
        active = [p for p in prompts if p]
        if not active:
            return history
        return [history[0], SystemMessage(content="\n\n".join(active))] + history[1:]

    def reset(self, session_id: str) -> None:
        """Clears a session's memory; the next turn starts a fresh SystemMessage."""
        self._sessions.pop(session_id, None)
        self._quiz_attempts.pop(session_id, None)

    def recent_transcript(self, session_id: str, max_chars: int = 4000) -> str:
        """Plain "Student: .../Assistant: ..." transcript of a session's
        history, most recent max_chars kept. Shared by summarize() (which
        hands it to the model for a 3-line recap) and
        generate_flashcards_from_conversation() (which hands a longer cut
        of it to the model to identify a topic and build a deck from) --
        unlike summarize()'s own return value, this is the raw
        conversation itself, not an LLM-written gloss of it.

        "[Likely tool: ...]" / "[Course context: ...]" prefixes that
        _intent_hint / main.py's _prefixed_input tag onto the stored human
        message are stripped -- they're routing hints for the model, not
        something worth re-showing it here.

        Returns "" if the session doesn't exist yet or nothing was
        actually said.
        """
        history = self._sessions.get(session_id)
        if not history or len(history) <= 1:
            return ""

        lines = []
        for msg in history:
            if isinstance(msg, HumanMessage) and msg.content:
                text = re.sub(r"^(\[[^\]]+\]\s*)+", "", str(msg.content))
                lines.append(f"Student: {text}")
            elif isinstance(msg, AIMessage) and msg.content:
                lines.append(f"Assistant: {msg.content}")
        if not lines:
            return ""
        return "\n".join(lines)[-max_chars:]

    def summarize(self, session_id: str) -> str | None:
        """Generates a short recap of a session before it's reset. Returns
        None if the session doesn't exist yet or nothing was actually said.
        """
        transcript = self.recent_transcript(session_id)
        if not transcript:
            return None
        try:
            summary_msg = _invoke_with_backoff(self.llm.invoke, [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript),
            ])
            return summary_msg.content.strip() or None
        except Exception:
            return None

    def generate_flashcards_from_conversation(self, transcript: str, count: int = 10) -> tuple[str, list[dict]]:
        """Generates up to `count` study flashcards from what's actually
        been discussed in a chat session (see recent_transcript()), rather
        than from a specific course's or notebook doc's raw material. The
        model identifies the topic itself and is explicitly permitted to
        draw on its own general knowledge of that topic when writing
        cards -- not just facts that happened to be typed out in the
        conversation -- the same "grounded means don't invent facts, not
        only discuss what's literally in front of you" permission depth
        mode already gives (see MODE_PROMPTS["depth"]).

        Returns (topic, cards). Unlike the old course/notebook-grounded
        flashcards path this replaced, the caller (see main.py's
        /flashcards endpoint) has no title to hand in ahead of time --
        there's no resolved target here -- so the model reports the topic
        it identified and the caller titles the deck from that. topic is
        "" and cards is [] when the conversation doesn't have enough of an
        actual subject to build a deck from (e.g. only greetings, or
        GPA/scheduling chat with nothing conceptual discussed) -- the
        caller treats that as "not enough to work with yet" rather than a
        generation failure.

        Same structured-output mechanism (Groq's strict JSON-schema mode),
        session-less one-shot call using self.llm directly, and
        raise-on-failure contract as the old generate_flashcards had --
        callers should catch and translate, same as main.py already does
        for FRIENDLY_QUOTA_MESSAGE.
        """
        prompt = (
            f"Create up to {count} study flashcards based on the conversation below.\n\n"
            "First, identify the main topic or concept the conversation is actually about -- "
            "the thing the student would benefit from being quizzed on, not just whatever was "
            "typed most recently. Then write flashcards that test real understanding of that "
            "topic: definitions, how something works, why it matters, worked examples -- draw "
            "on your own general knowledge of the subject, not only facts that happened to be "
            "typed out in the chat below. One clear question per card, with a concise, correct "
            "answer. If the conversation doesn't have enough of a topic to build a deck from "
            "(e.g. only greetings, or scheduling/GPA-only chat with no actual subject "
            "discussed), set topic to an empty string and return an empty cards list instead of "
            "forcing something unrelated.\n\n"
            f"CONVERSATION:\n{transcript}"
        )
        structured_llm = self.llm.with_structured_output(GeneratedTopicFlashcards, method="json_schema", strict=True)
        result: GeneratedTopicFlashcards = _invoke_with_backoff(structured_llm.invoke, [HumanMessage(content=prompt)])
        return result.topic.strip(), [c.model_dump() for c in result.cards][:count]

    def generate_quiz_from_conversation(self, transcript: str, count: int = 5) -> tuple[str, list[dict]]:
        """Generates up to `count` multiple-choice questions from what's
        actually been discussed in a chat session (see recent_transcript())
        -- the multiple-choice, explicitly-graded counterpart to
        generate_flashcards_from_conversation(), built the same way and for
        the same underlying reason: an answer is checked against
        correct_index right there, client-side, rather than the old
        approach this whole feature was deferred behind -- trying to infer
        from free-form chat whether the student "seemed to get it". Same
        "identify the topic yourself, draw on your own general knowledge,
        don't force something unrelated" permissions as the flashcards
        path.

        Returns (topic, questions) -- empty ("", []) when the conversation
        doesn't have enough of an actual subject yet, same convention as
        generate_flashcards_from_conversation().

        Each question dict has "question", "options" (exactly 4),
        "correct_index" (0-3), and "explanation" -- shown to the student
        right after they answer, right or wrong, so a miss is a learning
        moment rather than just a red X. This is also what feeds a later
        /retry-quiz: see record_quiz_results() / generate_retry_quiz().

        Same structured-output mechanism, session-less one-shot call, and
        raise-on-failure contract as generate_flashcards_from_conversation.
        """
        prompt = (
            f"Create up to {count} multiple-choice study questions based on the conversation "
            "below.\n\n"
            "First, identify the main topic or concept the conversation is actually about -- "
            "the thing the student would benefit from being quizzed on, not just whatever was "
            "typed most recently. Then write questions that test real understanding of that "
            "topic -- draw on your own general knowledge of the subject, not only facts that "
            "happened to be typed out in the chat below. Each question needs exactly 4 answer "
            "options with exactly one correct, and a short explanation of why the correct "
            "answer is right (and ideally why the most tempting wrong option is wrong) -- the "
            "explanation is shown to the student right after they answer. Keep all 4 options "
            "genuinely plausible; avoid a giveaway option that's obviously wrong on its face. "
            "If the conversation doesn't have enough of a topic to build questions from (e.g. "
            "only greetings, or scheduling/GPA-only chat with no actual subject discussed), "
            "set topic to an empty string and return an empty questions list instead of "
            "forcing something unrelated.\n\n"
            f"CONVERSATION:\n{transcript}"
        )
        structured_llm = self.llm.with_structured_output(GeneratedTopicQuiz, method="json_schema", strict=True)
        result: GeneratedTopicQuiz = _invoke_with_backoff(structured_llm.invoke, [HumanMessage(content=prompt)])
        return result.topic.strip(), [q.model_dump() for q in result.questions][:count]

    def record_quiz_results(self, session_id: str, topic: str, answers: list[dict]) -> int:
        """Appends one attempt per graded question to this session's quiz
        history (see __init__), tagged with the topic it came from, so a
        later generate_retry_quiz() call has real misses to draw on.
        Grading is already done by the time this is called -- the frontend
        checked chosen_index against correct_index the moment the student
        answered -- this just records the outcome (see QuizAnswer /
        POST /quiz/results).

        Returns how many of the given answers were wrong, so the caller
        (main.py's /quiz/results endpoint) can tell the student right away
        without a second lookup.
        """
        attempts = self._quiz_attempts.setdefault(session_id, [])
        wrong = 0
        for a in answers:
            correct = a["chosen_index"] == a["correct_index"]
            if not correct:
                wrong += 1
            attempts.append({**a, "topic": topic, "correct": correct})
        return wrong

    def generate_retry_quiz(self, session_id: str, count: int = 5) -> tuple[str, list[dict]] | None:
        """Generates a fresh multiple-choice quiz focused on what this
        session has actually gotten wrong so far (see
        record_quiz_results()) -- the "then generates an exam of those
        plus similar questions" half of the originally-requested
        retry-exam feature, now grounded in real graded attempts instead
        of inferred from free-form chat.

        Returns None if there's nothing to retry -- no quiz taken yet in
        this session, or everything answered so far was correct -- so the
        caller (main.py's /retry-quiz endpoint) can tell those two cases
        apart with a clearer message than a generic empty result would.

        Otherwise returns (topic, questions) in the same shape
        generate_quiz_from_conversation() does. Roughly half the new
        questions retest one of the missed concepts from a different angle
        (never a verbatim repeat of the original question -- see the
        prompt below); the rest are similar reinforcement questions on the
        same topic(s). Same structured-output mechanism and
        raise-on-failure contract as generate_quiz_from_conversation.
        """
        attempts = self._quiz_attempts.get(session_id, [])
        if not attempts:
            return None
        wrong = [a for a in attempts if not a["correct"]]
        if not wrong:
            return None

        missed_lines = "\n".join(
            f'- Topic: {a["topic"]}\n'
            f'  Q: {a["question"]}\n'
            f'  They chose: "{a["options"][a["chosen_index"]]}" -- '
            f'correct answer: "{a["options"][a["correct_index"]]}"'
            for a in wrong
        )
        prompt = (
            f"A student got the following multiple-choice questions wrong in a study "
            f"session. Create up to {count} new multiple-choice questions to help them "
            "review this material.\n\n"
            "For roughly half the new questions, test the SAME underlying concept as one "
            "of the missed questions below, from a different angle or a rephrased "
            "scenario -- never a verbatim repeat of the original question. For the rest, "
            "write related questions on the same topic(s) that reinforce the material "
            "without being near-duplicates of each other or of the missed questions. Draw "
            "on your own general knowledge of the topic(s), not only what's written below. "
            "Each question needs exactly 4 answer options with exactly one correct, and a "
            "short explanation of why the correct answer is right. Set topic to a short "
            "label summarizing what this review covers.\n\n"
            f"MISSED QUESTIONS:\n{missed_lines}"
        )
        structured_llm = self.llm.with_structured_output(GeneratedTopicQuiz, method="json_schema", strict=True)
        result: GeneratedTopicQuiz = _invoke_with_backoff(structured_llm.invoke, [HumanMessage(content=prompt)])
        return result.topic.strip(), [q.model_dump() for q in result.questions][:count]

    def has_quiz_attempts(self, session_id: str) -> bool:
        """Whether this session has recorded any graded quiz attempts at
        all (right or wrong) -- lets a caller distinguish "hasn't taken a
        quiz yet" from generate_retry_quiz()'s other None case, "took one
        but hasn't missed anything", which need different messages (see
        main.py's /retry-quiz).
        """
        return bool(self._quiz_attempts.get(session_id))

    def chat(
        self,
        session_id: str,
        user_input: str,
        mode: str | None = None,
        extra_tools=None,
        exclude_tool_names: set[str] | None = None,
        extra_system_prompt: str | None = None,
    ) -> dict:
        """Runs one turn of the ReAct loop.

        extra_tools: optional request-scoped tools (e.g. a device-bound
        search_notebook) that apply to this call only -- see
        _tools_for_call. Never persisted into the session's own tool set.
        exclude_tool_names: optional base tool names to leave out of this
        call's toolset entirely (e.g. search_syllabus, when the student
        has pinned specific notebook PDFs -- see main.py). Also never
        persisted.
        extra_system_prompt: optional extra ephemeral system instruction
        for this call only, alongside the mode prompt (if any) -- see
        _with_ephemeral. Used for e.g. the notebook-restriction note main.py
        builds when notebook_doc_ids is set.

        Returns {"content": str, "sources": list[dict], "tools_used": list[str],
        "mode": str | None} instead of a bare string so the API layer can
        surface grounding info, which tools actually fired, and which mode
        (if any) applied to this turn.
        """
        history = self._history(session_id)

        hint = _intent_hint(user_input)
        tagged_input = f"[Likely tool: {hint}] {user_input}" if hint else user_input
        history.append(HumanMessage(content=tagged_input))

        mode_key = (mode or "").strip().lower()
        mode_prompt = MODE_PROMPTS.get(mode_key)

        llm_with_tools, tools_map = self._tools_for_call(extra_tools, exclude_tool_names)

        tools_used: list[str] = []
        sources: list[dict] = []

        for _ in range(self.max_turns):
            call_messages = self._with_ephemeral(history, [mode_prompt, extra_system_prompt])
            try:
                ai_msg: AIMessage = _invoke_with_backoff(llm_with_tools.invoke, call_messages)
            except Exception as e:
                content = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Sorry, I hit an error talking to the model: {e}"
                return {
                    "content": content,
                    "sources": [],
                    "tools_used": [],
                    "mode": mode_key or None,
                }

            history.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                return {
                    "content": ai_msg.content,
                    "sources": _dedupe_sources(sources),
                    "tools_used": tools_used,
                    "mode": mode_key or None,
                }

            for tool_call in ai_msg.tool_calls:
                name = tool_call["name"]
                tool_obj = tools_map.get(name)
                if tool_obj is None:
                    observation = f"Error: tool '{name}' is not registered."
                else:
                    observation = tool_obj.invoke(tool_call["args"])
                    tools_used.append(name)
                    if name in ("search_syllabus", "search_notebook"):
                        sources.extend(_parse_sources(str(observation)))
                history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        return {
            "content": "I'm having trouble finishing that request -- could you rephrase or ask one thing at a time?",
            "sources": _dedupe_sources(sources),
            "tools_used": tools_used,
            "mode": mode_key or None,
        }

    def chat_stream(
        self,
        session_id: str,
        user_input: str,
        mode: str | None = None,
        extra_tools=None,
        exclude_tool_names: set[str] | None = None,
        extra_system_prompt: str | None = None,
    ):
        """Generator twin of chat(): yields small event dicts as the turn
        progresses instead of returning one final dict, so the API layer can
        forward them to the client live over SSE.

        extra_tools / exclude_tool_names / extra_system_prompt: same
        meaning as in chat().

        Event shapes:
          {"type": "tool", "name": ...}                                  -- a tool started running
          {"type": "token", "text": ...}                                 -- a piece of the final answer
          {"type": "done", "content", "sources", "tools_used", "mode"}   -- turn finished normally
          {"type": "error", "error": ...}                                -- turn failed; stream ends

        Tool-call turns and the final tool-free answer turn are both run via
        .stream() so a network/API error surfaces mid-turn instead of only
        after a full non-streaming call would have completed -- but tokens
        are only forwarded to the caller once we're confident this turn is
        the real answer and not a tool call: the very first chunk of a turn
        tells us which, since providers emit tool_call_chunks (not prose)
        from the first chunk of a turn that's invoking a tool. That's a
        pragmatic heuristic, not a guarantee for every possible provider
        behavior -- a model that emitted a little preamble text before
        deciding to call a tool could leak that fragment -- but it matches
        how this project's guardrail prompt already tells the model to
        behave (call the tool, don't narrate first), and it avoids the far
        more fragile alternative of trying to detect and retroactively
        "unsend" already-streamed tokens.
        """
        history = self._history(session_id)

        hint = _intent_hint(user_input)
        tagged_input = f"[Likely tool: {hint}] {user_input}" if hint else user_input
        history.append(HumanMessage(content=tagged_input))

        mode_key = (mode or "").strip().lower()
        mode_prompt = MODE_PROMPTS.get(mode_key)

        llm_with_tools, tools_map = self._tools_for_call(extra_tools, exclude_tool_names)

        tools_used: list[str] = []
        sources: list[dict] = []

        for turn in range(self.max_turns):
            call_messages = self._with_ephemeral(history, [mode_prompt, extra_system_prompt])
            is_tool_turn: bool | None = None
            chunks = []

            try:
                for chunk in _stream_with_backoff(llm_with_tools.stream, call_messages):
                    chunks.append(chunk)
                    if is_tool_turn is None:
                        is_tool_turn = bool(getattr(chunk, "tool_call_chunks", None))
                    if not is_tool_turn and chunk.content:
                        yield {"type": "token", "text": chunk.content}
            except Exception as e:
                error_msg = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Sorry, I hit an error talking to the model: {e}"
                yield {"type": "error", "error": error_msg}
                return

            if not chunks:
                yield {"type": "error", "error": "No response was generated."}
                return

            ai_msg = chunks[0]
            for c in chunks[1:]:
                ai_msg = ai_msg + c

            history.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                yield {
                    "type": "done",
                    "content": ai_msg.content,
                    "sources": _dedupe_sources(sources),
                    "tools_used": tools_used,
                    "mode": mode_key or None,
                }
                return

            for tool_call in ai_msg.tool_calls:
                name = tool_call["name"]
                yield {"type": "tool", "name": name}
                tool_obj = tools_map.get(name)
                if tool_obj is None:
                    observation = f"Error: tool '{name}' is not registered."
                else:
                    observation = tool_obj.invoke(tool_call["args"])
                    tools_used.append(name)
                    if name in ("search_syllabus", "search_notebook"):
                        sources.extend(_parse_sources(str(observation)))
                history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        yield {
            "type": "done",
            "content": "I'm having trouble finishing that request -- could you rephrase or ask one thing at a time?",
            "sources": _dedupe_sources(sources),
            "tools_used": tools_used,
            "mode": mode_key or None,
        }
